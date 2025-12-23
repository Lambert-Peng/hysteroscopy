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
import torch_directml
from torch.optim.lr_scheduler import CosineAnnealingLR


# ======== 1. Dataset (修正版: 取第一個 Token) ========
class RecursiveFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        
        if len(self.files) == 0:
            raise RuntimeError(f"在 {root_dir} 找不到任何 .pt 檔案！")
            
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fpath):
        filename = fpath.name
        # A 或 C -> Class 0 (Abnormal)
        # B     -> Class 1 (Normal)
        if filename.startswith("A") or filename.startswith("C"):
            return 0  
        elif filename.startswith("B"):
            return 1  
        else:
            raise ValueError(f"Unknown class in file: {filename}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        feature = torch.load(fpath)  # 不指定 CPU
        feature = feature.float() 
        
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
        return feature, label, fpath.name # 回傳檔名以便存檔

# ======== 2. Model ========
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden1=256, hidden2=64, num_classes=2, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),                 # 使用 GELU 激活函數 (ViT 的標準配備)
            nn.Dropout(dropout),       # 防止過擬合
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),                 # 使用 GELU 激活函數 (ViT 的標準配備)
            nn.Dropout(dropout), 
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ======== 3. Config & Setup ========
TRAIN_DIR = "dataset/train/weights"
VAL_DIR = "dataset/val/weights"

device = torch_directml.device()
print(f"Using device: {device}")

train_dataset = RecursiveFeatureDataset(TRAIN_DIR)
val_dataset = RecursiveFeatureDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 300
patience = 80
best_val_loss = float("inf")
no_improve_epochs = 0

model = MLPClassifier(num_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 權重設定 (A+C vs B)
class_weights = torch.tensor([1.0, 1.8], dtype=torch.float).to(device)

# 儲存設定
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/train2m_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"訓練結果將儲存至: {save_dir}")

best_model_path = os.path.join(save_dir, "best_model_binary.pt")

# ======== 4. Training Loop ========


train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("開始訓練...")
for epoch in range(num_epochs):
    # --- Train ---
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

    train_avg_loss = total_loss / len(train_dataset)
    train_acc = correct / total
    train_losses.append(train_avg_loss)
    train_accs.append(train_acc)

    scheduler.step()

    # --- Validation ---
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

    val_avg_loss = val_total_loss / len(val_dataset)
    val_acc = correct / total
    val_losses.append(val_avg_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1:03d} | Train Loss: {train_avg_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_avg_loss:.4f} Acc: {val_acc:.3f}")

    # --- Early Stopping ---
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  >>> 最佳模型已儲存 (Loss: {best_val_loss:.4f})")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# 繪製曲線
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.savefig(os.path.join(save_dir, "training_curves.png"))
plt.close()

# ======== 5. [關鍵新增] 載入最佳模型並輸出 Excel ========
print("\n正在生成驗證集預測結果 (Excel)...")

# 重新載入最佳權重
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_filenames = []
all_trues = []
all_preds = []
all_prob_0 = []
all_prob_1 = []

with torch.no_grad():
    for x, y, filenames in val_loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1) # 轉成機率
        preds = logits.argmax(1)
        
        # 收集數據
        all_filenames.extend(filenames)
        all_trues.extend(y.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_prob_0.extend(probs[:, 0].cpu().numpy()) # Class 0 (A+C)
        all_prob_1.extend(probs[:, 1].cpu().numpy()) # Class 1 (B)

# 建立 DataFrame
df_out = pd.DataFrame({
    "filename": all_filenames,
    "true_label": all_trues,
    "pred_label": all_preds,
    "prob_class0": all_prob_0,
    "prob_class1": all_prob_1
})

excel_path = os.path.join(save_dir, "val_predictions.xlsx")
df_out.to_excel(excel_path, index=False)
print(f"預測結果已儲存至: {excel_path}")
print("現在你可以執行 evaluate2.py 進行評估了！")