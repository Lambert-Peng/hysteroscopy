import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import torch_directml
from torch.optim.lr_scheduler import CosineAnnealingLR


# ================= 1. Dataset =================
class RecursiveFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"在 {root_dir} 找不到任何 .pt 檔案！")
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fpath):
        filename = fpath.name
        if filename.startswith("A"): return 0
        elif filename.startswith("B"): return 1
        elif filename.startswith("C"): return 2
        else: raise ValueError(f"Unknown class: {filename}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        feature = torch.load(fpath)
        feature = feature.float()
        if feature.dim() == 3: feature = feature[0].mean(dim=0)
        elif feature.dim() == 2: feature = feature.mean(dim=0)
        elif feature.dim() == 1 and feature.shape[0] > 768: feature = feature.view(-1, 768).mean(dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label, fpath.name

# ================= 2. Model =================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.linear(x)

# ================= 3. Setup =================
train_dir = "dataset/train/weights"
val_dir = "dataset/val/weights"

device = torch_directml.device()
print(f"Using device: {device}")

train_dataset = RecursiveFeatureDataset(train_dir)
val_dataset = RecursiveFeatureDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

IGNORE_CLASS_INDEX = 1
model = MLPClassifier(num_classes=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Weight A(0): 1.06, B(1): 0.0, C(2): 0.95
ac_class_weights = torch.tensor([1.06, 0.0, 0.95]).to(device)
criterion = nn.CrossEntropyLoss(weight=ac_class_weights, ignore_index=IGNORE_CLASS_INDEX)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/train_AC_specific_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "best_model_AC.pt")

# ================= 4. Training Loop =================
num_epochs = 100
patience = 30
no_improve_epochs = 0

best_val_loss = float("inf")
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print(f"開始訓練 AC 專用模型 (忽略 Class B)...")

for epoch in range(num_epochs):
    model.train()
    total_loss, correct_ac, total_ac = 0, 0, 0
    
    for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)
        valid_mask = (y != IGNORE_CLASS_INDEX)
        if valid_mask.sum() == 0: continue
            
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        preds = out.argmax(1)
        total_loss += loss.item() * valid_mask.sum().item()
        correct_ac += (preds[valid_mask] == y[valid_mask]).sum().item()
        total_ac += valid_mask.sum().item()

    scheduler.step()
    if total_ac == 0: total_ac = 1 
    train_loss = total_loss / total_ac
    train_acc = correct_ac / total_ac
    train_losses.append(train_loss); train_accs.append(train_acc)
    
    # --- Validation ---
    model.eval()
    val_total_loss, val_correct_ac, val_total_ac = 0, 0, 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            valid_mask = (y != IGNORE_CLASS_INDEX)
            if valid_mask.sum() == 0: continue
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss): continue
            preds = out.argmax(1)
            val_total_loss += loss.item() * valid_mask.sum().item()
            val_correct_ac += (preds[valid_mask] == y[valid_mask]).sum().item()
            val_total_ac += valid_mask.sum().item()

    if val_total_ac == 0: val_total_ac = 1
    val_loss = val_total_loss / val_total_ac
    val_acc = val_correct_ac / val_total_ac
    val_losses.append(val_loss); val_accs.append(val_acc)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("  >>> 最佳模型已儲存")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# ================= 5. 繪製訓練曲線 =================
epochs_range = np.arange(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.title('Loss Curve (AC Only)'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Acc')
plt.plot(epochs_range, val_accs, label='Val Acc')
plt.title('Accuracy Curve (AC Only)'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_curves.png"))
plt.close()

# ================= 6. [新增] 產生 Excel 與 評估圖表 =================
print("\n正在生成預測結果並進行評估...")

# 載入最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()

results = []
y_true_ac = [] # 只存 A 和 C 的真實標籤
y_pred_ac = [] # 只存 A 和 C 的預測標籤
y_scores_ac = [] # 只存 A 和 C 的機率 (用來畫 ROC)

with torch.no_grad():
    for x, y, filenames in val_loader:
        x = x.to(device)
        logits = model(x)
        # 即使模型輸出 3 類，我們也只看 A 和 C 的 logits
        # 手動 Softmax (只取 A(0) 和 C(2))
        logit_A = logits[:, 0]
        logit_C = logits[:, 2]
        exp_A = torch.exp(logit_A)
        exp_C = torch.exp(logit_C)
        sum_exp = exp_A + exp_C + 1e-8
        
        prob_A = (exp_A / sum_exp).cpu().numpy()
        prob_C = (exp_C / sum_exp).cpu().numpy()
        
        # 為了相容，我們還是存一個 prob_B 但設為 0
        prob_B = np.zeros_like(prob_A)
        
        batch_size = len(filenames)
        for i in range(batch_size):
            label_idx = y[i].item()
            
            # 存 Excel (所有檔案都要存，方便之後 Combined 使用)
            results.append({
                "filename": filenames[i],
                "true_label": label_idx,
                "pred_label": 0 if prob_A[i] > prob_C[i] else 2, # 簡單判斷
                "prob_A": prob_A[i],
                "prob_B": 0.0,
                "prob_C": prob_C[i]
            })
            
            # 存評估數據 (只存 A 和 C 的樣本)
            if label_idx != IGNORE_CLASS_INDEX:
                y_true_ac.append(label_idx) # 0 或 2
                # 為了方便畫二分類 ROC，我們把 A 當 0，C 當 1
                mapped_label = 0 if label_idx == 0 else 1 
                y_pred_ac.append(mapped_label) 
                y_scores_ac.append(prob_C[i]) # 取 C 的機率當作 Positive Score

# 1. 輸出 Excel
excel_path = os.path.join(save_dir, "val_predictions_AC.xlsx")
pd.DataFrame(results).to_excel(excel_path, index=False)
print(f"預測結果已儲存至: {excel_path}")
