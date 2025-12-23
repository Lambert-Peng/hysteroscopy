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
from pathlib import Path
import torch_directml  # <--- AMD 顯卡支援
from torch.optim.lr_scheduler import CosineAnnealingLR

# ======== 1. Dataset (三分類 A=0, B=1, C=2) ========
class RecursiveFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        
        if len(self.files) == 0:
            raise RuntimeError(f"在 {root_dir} 找不到任何 .pt 檔案！")
            
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fpath):
        filename = fpath.name
        if filename.startswith("A"):
            return 0
        elif filename.startswith("B"):
            return 1
        elif filename.startswith("C"):
            return 2
        else:
            raise ValueError(f"Unknown class in file: {filename}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        feature = torch.load(fpath)  # 不指定 CPU
        feature = feature.float()
        # 形狀處理
        # [形狀修正] 確保只取 [768]
        if feature.dim() == 3:
            # feature[0] 取出 [197, 768]
            # .mean(dim=0) 對 197 個 token 取平均 -> 變成 [768]
            feature = feature[0].mean(dim=0)
            
        # 情況 2: 形狀是 [197, 768] (標準形狀)
        elif feature.dim() == 2:
            # 直接對 dim=0 (197 那一維) 取平均 -> 變成 [768]
            feature = feature.mean(dim=0)
            
        # 情況 3: 形狀是 [151296] (被攤平了)
        elif feature.dim() == 1 and feature.shape[0] > 768:
            # 先用 view 恢復成 [N, 768]，再取平均
            # -1 會自動算出 197 (或任何 token 數)
            feature = feature.view(-1, 768).mean(dim=0)
             
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label, fpath.name

# ======== 2. Model (三分類) ========
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
         return self.linear(x)

# ======== 3. Config & Setup ========
# 請修改路徑
train_dir = "dataset/train/weights"
val_dir = "dataset/val/weights"

# 設定 AMD 裝置
device = torch_directml.device()
print(f"Using device: {device}")

train_dataset = RecursiveFeatureDataset(train_dir)
val_dataset = RecursiveFeatureDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 800
patience = 80
best_val_loss = float("inf")
no_improve_epochs = 0

model = MLPClassifier(num_classes=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float).to(device)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/train3_mean_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"訓練結果將儲存至: {save_dir}")

# ======== 4. Train Loop ========

best_model_path = os.path.join(save_dir, "best_model_3class.pt")

train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("開始訓練 (三分類: A=0, B=1, C=2)...")
for epoch in range(num_epochs):
    # Train
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
    
    train_loss = total_loss / len(train_dataset)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    scheduler.step()

    # Validation
    model.eval()
    val_total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, weight=class_weights)
            val_total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    val_loss = val_total_loss / len(val_dataset)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  >>> 最佳模型已儲存 (Loss: {best_val_loss:.4f})")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print("Early stopping triggered.")
        break

# ======== 5. 繪製訓練曲線 ========
epochs_range = np.arange(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Acc')
plt.plot(epochs_range, val_accs, label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(save_dir, "training_curves.png"))
plt.close()

# ======== 6. 產生 Excel (驗證集預測結果) ========
print("\n正在生成 Excel 預測結果...")
model.load_state_dict(torch.load(best_model_path)) # 載入最佳權重
model.eval()

all_data = {
    "filename": [], "true_label": [], "pred_label": [],
    "prob_A": [], "prob_B": [], "prob_C": []
}

with torch.no_grad():
    for x, y, names in val_loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1) # [Batch, 3]
        preds = logits.argmax(1)

        all_data["filename"].extend(names)
        all_data["true_label"].extend(y.numpy())
        all_data["pred_label"].extend(preds.cpu().numpy())
        all_data["prob_A"].extend(probs[:, 0].cpu().numpy()) # Class 0
        all_data["prob_B"].extend(probs[:, 1].cpu().numpy()) # Class 1
        all_data["prob_C"].extend(probs[:, 2].cpu().numpy()) # Class 2

df = pd.DataFrame(all_data)
excel_path = os.path.join(save_dir, "val_predictions_3class.xlsx")
df.to_excel(excel_path, index=False)
print(f"預測結果已儲存至: {excel_path}")
print("請接著執行 evaluate3_class.py")