import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch_directml
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ================= 設定區 =================
# [關鍵修改] 使用 Top-K 策略
TOP_K = 3  # 取分數最高的前 3 張算平均
BAG_THRESHOLD = 0.45  # 只要 Top-3 平均 > 0.40 就判異常 (為了不漏診，設低一點)

# 模型路徑 (請確認路徑正確)
BINARY_MODEL_PATH = "1210/results/best/train2_1layer_mean_20251210_010556/best_model_binary.pt"
AC_MODEL_PATH = "results/train_AC_specific_20251223_170912/best_model_AC.pt"
VAL_DATA_DIR = "dataset/val/weights"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"results/bag_topk_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch_directml.device()

# ================= 類別定義 (保持不變) =================
class RecursiveFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        self.labels = [self._get_label(f) for f in self.files]
    def _get_label(self, fpath):
        filename = fpath.name
        if filename.startswith("A"): return 0
        elif filename.startswith("B"): return 1
        elif filename.startswith("C"): return 2
        return 0
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fpath = self.files[idx]
        feature = torch.load(fpath).float()
        if feature.dim() == 3: feature = feature[0].mean(dim=0)
        elif feature.dim() == 2: feature = feature.mean(dim=0)
        elif feature.dim() == 1 and feature.shape[0] > 768: feature = feature.view(-1, 768).mean(dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label, fpath.name

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.linear(x)

def load_model(path, num_classes):
    print(f"Loading: {path}")
    model = MLPClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_bag_id(filename):
    parts = str(filename).split('_')
    if len(parts) >= 2: return f"{parts[0]}_{parts[1]}"
    return str(filename)

# ================= 主程式 =================
def main():
    model_bin = load_model(BINARY_MODEL_PATH, num_classes=2) 
    model_ac  = load_model(AC_MODEL_PATH, num_classes=3)     

    dataset = RecursiveFeatureDataset(VAL_DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("正在掃描所有影像...")
    all_data = []
    
    with torch.no_grad():
        for features, label, filenames in loader:
            features = features.to(device)
            # Binary
            logits_bin = model_bin(features)
            probs_bin = F.softmax(logits_bin, dim=1).cpu().numpy()
            # AC
            logits_ac = model_ac(features)
            logit_A, logit_C = logits_ac[:, 0], logits_ac[:, 2]
            exp_A, exp_C = torch.exp(logit_A), torch.exp(logit_C)
            sum_exp = exp_A + exp_C + 1e-8
            prob_A = (exp_A / sum_exp).cpu().numpy()
            prob_C = (exp_C / sum_exp).cpu().numpy()
            
            batch_size = len(filenames)
            for i in range(batch_size):
                all_data.append({
                    "bag_id": extract_bag_id(filenames[i]),
                    "true_label": label[i].item(),
                    "bin_prob_abnormal": probs_bin[i, 0], # 假設 0 是異常 (A+C)
                    "ac_prob_A": prob_A[i],
                    "ac_prob_C": prob_C[i]
                })

    df = pd.DataFrame(all_data)
    
    # 3. Bag Level 決策 (Top-K 邏輯)
    print(f"開始 Bag Level 決策 (Top-{TOP_K} Mean, Threshold={BAG_THRESHOLD})...")
    
    grouped = df.groupby('bag_id')
    bag_results = []
    
    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # [關鍵邏輯] Top-K Mean Pooling
        # 取出該病人所有影像的異常分數，排序，取前 K 高
        scores = group['bin_prob_abnormal'].values
        if len(scores) >= TOP_K:
            top_k_scores = np.sort(scores)[-TOP_K:] # 取最後 K 個 (最大的)
            bag_score = top_k_scores.mean()
        else:
            # 如果影像少於 K 張，就取全部平均 (或最大)
            bag_score = scores.mean()

        # 篩檢決策
        if bag_score > BAG_THRESHOLD:
            # --- 判為異常，進入 Stage 2 (AC 鑑別) ---
            # 這裡我們只取 "被判定為異常機率高" 的那些影像來做 AC 分類，會更準
            # 或者簡單一點，還是用 Mean
            mean_prob_A = group['ac_prob_A'].mean()
            mean_prob_C = group['ac_prob_C'].mean()
            
            if mean_prob_A > mean_prob_C:
                final_pred = 0 # A
            else:
                final_pred = 2 # C
        else:
            # --- 判為正常 B ---
            final_pred = 1
        
        bag_results.append({
            "bag_id": bag_id,
            "true_label": true_label,
            "pred_label": final_pred,
            "bag_score": bag_score
        })

    df_bag = pd.DataFrame(bag_results)
    df_bag.to_excel(os.path.join(SAVE_DIR, "bag_level_predictions.xlsx"), index=False)
    
    # 4. 繪圖
    y_true = df_bag['true_label']
    y_pred = df_bag['pred_label']
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"Top-{TOP_K} Strategy 結果 (Thr={BAG_THRESHOLD})")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    
    class_names = ["A", "B", "C"]
    
    # 百分比版 CM
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap="Oranges", values_format=".2f")
    plt.title(f"Bag CM (Top-{TOP_K}, Thr={BAG_THRESHOLD})")
    plt.savefig(os.path.join(SAVE_DIR, "Bag_CM_TopK_Percent.png"))
    print(f"圖表已儲存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()