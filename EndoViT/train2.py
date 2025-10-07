# train_classifier_binary_fullviz.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np


# ======== Dataset ========
class FeatureDataset(Dataset):
    def __init__(self, weight_dir):
        self.weight_dir = weight_dir
        self.files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".pt")])
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fname):
        if fname.startswith("B"):
            return 1
        elif fname.startswith("A") or fname.startswith("C"):
            return 0
        else:
            raise ValueError(f"Unknown class: {fname}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = os.path.join(self.weight_dir, self.files[idx])
        feature = torch.load(fpath)
        cls_token = feature[0, :]
        label = torch.tensor(self.labels[idx])
        return cls_token, label


# ======== Load Data ========
train_dir = "dataset_split/train/weights"
val_dir = "dataset_split/val/weights"
train_dataset = FeatureDataset(train_dir)
val_dataset = FeatureDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ======== Model ========
import datetime

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden1=256, hidden2=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 建立不覆蓋的輸出資料夾
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"🟢 本次訓練結果將儲存至: {save_dir}")

# ======== Train ========
num_epochs = 100
train_losses, val_losses = [], []
precisions, recalls, f1_scores, maps = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # 將 A(0) 和 C(2) 合併成同一類 0；B(1) 為類別 1
        y = torch.where(y == 1, torch.tensor(1, device=y.device), torch.tensor(0, device=y.device))
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # ===== Validation =====
    model.eval()
    preds, trues = [], []
    all_logits = []
    val_total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            y = torch.where(y == 1, torch.tensor(1, device=y.device), torch.tensor(0, device=y.device))
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss = F.cross_entropy(logits, y)
            val_total_loss += val_loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
            all_logits.append(torch.softmax(logits, dim=1).cpu())

    preds, trues = np.array(preds), np.array(trues)
    all_logits = torch.cat(all_logits)
    val_loss_epoch = val_total_loss / len(val_loader.dataset)
    val_losses.append(val_loss_epoch)

    # Metrics
    precision = precision_score(trues, preds, average="binary", zero_division=0)
    recall = recall_score(trues, preds, average="binary", zero_division=0)
    f1 = f1_score(trues, preds, average="binary", zero_division=0)

    # mAP@50
    aps = []
    for i in range(2):
        y_true = (torch.tensor(trues) == i).int().numpy()
        y_score = all_logits[:, i].numpy()
        aps.append(average_precision_score(y_true, y_score))
    map50 = np.mean(aps)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    maps.append(map50)

    print(f"Epoch {epoch+1:02d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss_epoch:.4f} | P={precision:.3f} | R={recall:.3f} | F1={f1:.3f} | mAP@50={map50:.3f}")

# ===== Save Model =====
torch.save(model.state_dict(), os.path.join(save_dir, "mlp_classifier_2class.pt"))

# ===== Confusion Matrices =====
cm = confusion_matrix(trues, preds)
labels = ["A+C", "B"]

# 數量矩陣
disp1 = ConfusionMatrixDisplay(cm, display_labels=labels)
disp1.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_count.png"), dpi=300)
plt.close()

# 百分比矩陣
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
disp2.plot(cmap="Oranges", values_format=".2f")
plt.title("Confusion Matrix (Percentage)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ===== PR Curve =====
plt.figure()
for i, label in enumerate(labels):
    y_true = (torch.tensor(trues) == i).int().numpy()
    y_score = all_logits[:, i].numpy()
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.plot(r, p, label=f"{label} (AP={aps[i]:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_curve.png"), dpi=300)
plt.close()

# ===== Training Curves =====
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "train_val_loss_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(epochs, precisions, label="Precision")
plt.plot(epochs, recalls, label="Recall")
plt.plot(epochs, f1_scores, label="F1 Score")
plt.plot(epochs, maps, label="mAP@50")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Metrics Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "metrics_curve.png"), dpi=300)
plt.close()

print(f"\n✅ 兩分類訓練完成！所有結果與兩種 Confusion Matrix 已儲存於: {save_dir}")