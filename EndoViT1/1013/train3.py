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
    roc_curve,
    auc,
)
import numpy as np
import datetime

# ======== Dataset ========
class FeatureDataset(Dataset):
    def __init__(self, weight_dir):
        self.weight_dir = weight_dir
        self.files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".pt")])
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fname):
        if fname.startswith("A"):
            return 0
        elif fname.startswith("B"):
            return 1
        elif fname.startswith("C"):
            return 2
        else:
            raise ValueError(f"Unknown class: {fname}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = os.path.join(self.weight_dir, self.files[idx])
        feature = torch.load(fpath)       # [197,768]
        cls_token = feature[0, :]         # [768]
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
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden1=256, hidden2=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),     
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),     
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
class_weights = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float).to(device)

# ======== Output Dir ========
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/run3_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"本次訓練結果將儲存至: {save_dir}")

# ======== Train ========
num_epochs = 300
train_losses, val_losses = [], []
precisions, recalls, f1_scores, maps = [], [], [], []
train_accs, val_accs = [], []  

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ===== Validation =====
    model.eval()
    preds, trues = [], []
    all_logits = []
    val_total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss = F.cross_entropy(logits, y, weight=class_weights)
            val_total_loss += val_loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
            all_logits.append(torch.softmax(logits, dim=1).cpu())
            correct += (pred == y).sum().item()
            total += y.size(0)

    preds, trues = np.array(preds), np.array(trues)
    all_logits = torch.cat(all_logits)
    val_loss_epoch = val_total_loss / len(val_loader.dataset)
    val_acc = correct / total
    val_losses.append(val_loss_epoch)
    val_accs.append(val_acc)

    # ===== Metrics =====
    precision = precision_score(trues, preds, average="macro", zero_division=0)
    recall = recall_score(trues, preds, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    aps = []
    for i in range(3):
        y_true = (torch.tensor(trues) == i).int().numpy()
        y_score = all_logits[:, i].numpy()
        aps.append(average_precision_score(y_true, y_score))
    map50 = np.mean(aps)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    maps.append(map50)

    print(f"Epoch {epoch+1:02d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss_epoch:.4f} | "
          f"TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | P={precision:.3f} | "
          f"R={recall:.3f} | F1={f1:.3f} | mAP@50={map50:.3f}")

# ===== Save Model =====
torch.save(model.state_dict(), os.path.join(save_dir, "mlp_classifier_bn.pt"))

# ===== Confusion Matrices =====
cm = confusion_matrix(trues, preds)
labels = ["A", "B", "C"]

# 數量矩陣
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_count.png"), dpi=300)
plt.close()

# 百分比矩陣
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(cm_normalized, display_labels=labels).plot(cmap="Oranges", values_format=".2f")
plt.title("Confusion Matrix (Percentage)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ===== Precision–Recall Curve =====
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

# ===== ROC Curve =====
plt.figure()
for i, label in enumerate(labels):
    y_true = (torch.tensor(trues) == i).int().numpy()
    y_score = all_logits[:, i].numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_curve.png"), dpi=300)
plt.close()

# ===== Training Curves =====
epochs = np.arange(1, num_epochs + 1)

# Loss 曲線 (Train vs Val)
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

# Accuracy 曲線 
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training / Validation Accuracy Curve")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
plt.close()

# Precision / Recall / F1 / mAP 曲線
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

print(f"\n訓練完成！所有結果、ROC、Accuracy曲線與Confusion Matrix已儲存於: {save_dir}")