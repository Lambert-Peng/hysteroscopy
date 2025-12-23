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

# ==========================
#       MIL Dataset
# ==========================
class MILDataset(Dataset):
    def __init__(self, root):
        self.bags = []   # bag folder paths
        self.labels = [] # A/C = 0, B = 1

        for cls in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue

            for bag in sorted(os.listdir(cls_path)):
                bag_path = os.path.join(cls_path, bag)
                if os.path.isdir(bag_path):
                    self.bags.append(bag_path)
                    self.labels.append(self._cls_to_label(cls))

    def _cls_to_label(self, cls_name):
        return 0 if cls_name in ["A", "C"] else 1

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag_path = self.bags[idx]
        files = sorted([f for f in os.listdir(bag_path) if f.endswith(".pt")])

        feats = []
        for f in files:
            fpath = os.path.join(bag_path, f)
            feature = torch.load(fpath)       # (tokens, dim)
            cls_token = feature[0, :]         # CLS token
            feats.append(cls_token)

        feats = torch.stack(feats)            # (N_instances, 768)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, label, bag_path


# ==========================
#    Attention MIL Model
# ==========================
class AttentionMIL(nn.Module):
    def __init__(self, in_dim=768, num_classes=2, hidden_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: (batch=1, N_instances, in_dim)
        A = self.attention(x)            # (1, N, 1)
        A = torch.softmax(A, dim=1)      # attention weights

        bag_feat = torch.sum(A * x, dim=1)  # (1, in_dim)
        logits = self.classifier(bag_feat)  # (1, num_classes)

        return logits, A.squeeze(-1)


# ==========================
#        Config
# ==========================
train_dir = "dataset/train/weights"
val_dir   = "dataset/val/weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(MILDataset(train_dir), batch_size=1, shuffle=True)
val_loader   = DataLoader(MILDataset(val_dir), batch_size=1, shuffle=False)

model = AttentionMIL().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(device)

# ==========================
#        Save Dir
# ==========================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/train2_1layer_MIL_run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"訓練結果儲存於：{save_dir}")

# ==========================
#     Training Loop
# ==========================
num_epochs = 800
patience = 50
best_val_loss = float("inf")
no_improve_epochs = 0
best_model_path = os.path.join(save_dir, "best_model.pt")

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # ----------------------
    #       TRAIN
    # ----------------------
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, attn = model(x)  # (1,2)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += 1  # 因為 batch=1

    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)

    # ----------------------
    #     VALIDATION
    # ----------------------
    model.eval()
    val_total_loss, correct, total = 0, 0, 0
    preds, trues, prob_list, bag_names = [], [], [], []

    with torch.no_grad():
        for x, y, bag_path in val_loader:
            x, y = x.to(device), y.to(device)
            logits, attn = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            val_total_loss += loss.item()
            pred = logits.argmax(1)

            correct += (pred == y).sum().item()
            total += 1

            preds.append(pred.item())
            trues.append(y.item())
            prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            bag_names.append(bag_path)

    val_loss = val_total_loss / len(val_loader)
    val_acc = correct / total

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:03d} | TrainLoss={train_losses[-1]:.4f} | ValLoss={val_loss:.4f} | "
          f"TrainAcc={train_accs[-1]:.3f} | ValAcc={val_acc:.3f}")

    # ----------------------
    #    Early Stopping
    # ----------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Val Loss 改善 → 已儲存最佳模型至 {best_model_path}")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print("Early stopping!")
        break

# ==========================
#     Load Best Model
# ==========================
model.load_state_dict(torch.load(best_model_path))
print(f"已載入最佳模型 (ValLoss={best_val_loss:.4f})")

# ==========================
#     Save Excel
# ==========================
probs = np.concatenate(prob_list, axis=0)

df = pd.DataFrame({
    "bag": bag_names,
    "true_label": trues,
    "pred_label": preds,
    "prob_class0": probs[:, 0],
    "prob_class1": probs[:, 1],
})

excel_path = os.path.join(save_dir, "val_predictions.xlsx")
df.to_excel(excel_path, index=False)

# ==========================
#     Plot Curves
# ==========================
epochs = np.arange(1, len(train_losses) + 1)

def plot_loss(train_losses, val_losses):
    lr = optimizer.param_groups[0]['lr']
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Learning Rate: {lr:.6f} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

plot_loss(train_losses, val_losses)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
plt.close()

print(f"\n訓練完成！\n最佳模型: {best_model_path}\n驗證結果: {excel_path}")