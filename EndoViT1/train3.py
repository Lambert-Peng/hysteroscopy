import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

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
        feature = torch.load(fpath)      # [197,768]
        cls_token = feature[0, :]       # [768]
        label = torch.tensor(self.labels[idx])
        return cls_token, label, self.files[idx]


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


# ======== Config ========
train_dir = "1101/data/train/weights"
val_dir = "1101/data/val/weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(FeatureDataset(train_dir), batch_size=32, shuffle=True)
val_loader = DataLoader(FeatureDataset(val_dir), batch_size=32, shuffle=False)

model = MLPClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
class_weights = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float).to(device)

# ======== Save Dir ========
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/train3_run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"本次訓練結果將儲存至: {save_dir}")

# ======== Train Loop ========
num_epochs = 300
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    train_losses.append(total_loss / len(train_loader.dataset))
    train_accs.append(correct / total)

    # ===== Validation =====
    model.eval()
    preds, trues, logits_list, fnames = [], [], [], []
    val_total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y, names in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            val_total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
            logits_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            fnames.extend(names)

    val_losses.append(val_total_loss / len(val_loader.dataset))
    val_accs.append(correct / total)

    print(f"Epoch {epoch+1:03d} | TrainLoss={train_losses[-1]:.4f} | ValLoss={val_losses[-1]:.4f} | "
          f"TrainAcc={train_accs[-1]:.3f} | ValAcc={val_accs[-1]:.3f}")

# ===== Save Model =====
model_path = os.path.join(save_dir, "mlp_classifier.pt")
torch.save(model.state_dict(), model_path)

# ===== Save Validation Results for Evaluation =====
logits_all = np.concatenate(logits_list, axis=0)
df = pd.DataFrame({
    "filename": fnames,
    "true_label": trues,
    "pred_label": preds,
    "prob_class0": logits_all[:, 0],
    "prob_class1": logits_all[:, 1],
    "prob_class2": logits_all[:, 2],
})

excel_path = os.path.join(save_dir, "val_predictions.xlsx")
df.to_excel(excel_path, index=False)

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


print(f"\n模型與預測資料已儲存：\nModel: {model_path}\nData: {excel_path}")