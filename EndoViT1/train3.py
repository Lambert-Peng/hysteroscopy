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

# ======== Early Stopping 設定 ========
num_epochs = 300
patience = 100  # 若連續 100 個 epoch 未改善則提前停止
best_val_loss = float("inf")
no_improve_epochs = 0
best_model_path = os.path.join(save_dir, "best_model.pt")

train_losses, val_losses, train_accs, val_accs = [], [], [], []

# ======== Train Loop ========
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

    val_loss = val_total_loss / len(val_loader.dataset)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:03d} | TrainLoss={train_losses[-1]:.4f} | ValLoss={val_loss:.4f} | "
          f"TrainAcc={train_accs[-1]:.3f} | ValAcc={val_acc:.3f}")

    # ===== Early Stopping 檢查 =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Val loss 改善，已儲存最佳模型至 {best_model_path}")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(f"早停觸發：連續 {patience} 個 epoch 無改善，結束訓練。")
        break

# ===== 使用最佳模型 =====
model.load_state_dict(torch.load(best_model_path))
print(f"\n已載入最佳模型 (ValLoss={best_val_loss:.4f})")

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

# ===== Save Curves =====
epochs = np.arange(1, len(train_losses) + 1)
def plot_loss(train_losses, val_losses, optimizer):
    # 從 optimizer 取得 learning rate
    learning_rate = optimizer.param_groups[0]['lr']
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Rate: {learning_rate:.6f} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

plot_loss(train_losses, val_losses, optimizer)

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

print(f"\n模型與預測資料已儲存：\n最佳模型: {best_model_path}\n驗證結果: {excel_path}")
