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
# 閾值：病人層級的異常分數超過多少才進入第二關？
# 建議 0.3 或 0.35 (因為 Mean Pooling 會把分數拉平均，不用設太高)
BAG_THRESHOLD = 0.35

# 模型路徑
BINARY_MODEL_PATH = "1210/results/best/train2_1layer_mean_20251210_010556/best_model_binary.pt"
AC_MODEL_PATH = "results/train_AC_specific_20251223_170912/best_model_AC.pt"

# 資料路徑
VAL_DATA_DIR = "dataset/val/weights"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 輸出設定
SAVE_DIR = f"results/bag_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch_directml.device()

# ================= 類別定義 =================
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
    # 1. 載入模型
    model_bin = load_model(BINARY_MODEL_PATH, num_classes=2) 
    model_ac  = load_model(AC_MODEL_PATH, num_classes=3)     

    dataset = RecursiveFeatureDataset(VAL_DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. 收集所有 Instance 的預測 (先不決定，只存機率)
    print("正在掃描所有影像...")
    all_data = []
    
    with torch.no_grad():
        for features, label, filenames in loader:
            features = features.to(device)
            
            # Binary Model
            logits_bin = model_bin(features)
            probs_bin = F.softmax(logits_bin, dim=1).cpu().numpy()
            
            # AC Model
            logits_ac = model_ac(features)
            # 手動 Softmax (只取 A 和 C)
            logit_A = logits_ac[:, 0]
            logit_C = logits_ac[:, 2]
            exp_A = torch.exp(logit_A)
            exp_C = torch.exp(logit_C)
            sum_exp = exp_A + exp_C + 1e-8
            prob_A = (exp_A / sum_exp).cpu().numpy()
            prob_C = (exp_C / sum_exp).cpu().numpy()
            
            batch_size = len(filenames)
            for i in range(batch_size):
                all_data.append({
                    "bag_id": extract_bag_id(filenames[i]),
                    "true_label": label[i].item(),
                    "bin_prob_abnormal": probs_bin[i, 0], # Class 0 is Abnormal (A+C) in train2 logic? Check!
                    # ⚠️ 注意: 請確認 train2 的 label 定義。
                    # 通常: A/C -> 0, B -> 1
                    # 所以 prob[0] 是異常機率。如果你的設定相反，請這裡改成 probs_bin[i, 1]
                    
                    "ac_prob_A": prob_A[i],
                    "ac_prob_C": prob_C[i]
                })

    df = pd.DataFrame(all_data)
    
    # 3. Bag Level 決策 (關鍵邏輯)
    print(f"開始 Bag Level 決策 (Threshold={BAG_THRESHOLD})...")
    
    grouped = df.groupby('bag_id')
    bag_results = []
    
    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # Step A: 計算兩種統計量
        bag_mean_score = group['bin_prob_abnormal'].mean()
        bag_max_score = group['bin_prob_abnormal'].max()

        # Step B: 混合篩檢 (Hybrid)
        # 門檻設定：
        # Mean 設 0.35 (保護 B)
        # Max 設 0.65 (專抓 C，設高一點避免氣泡誤判)
        if (bag_mean_score > 0.35) or (bag_max_score > 0.55):
            # --- 判為異常，進入 Stage 2 (AC 鑑別) ---
            # 這裡用 Mean 比較穩，因為已經確定是病人異常了
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
        })

    df_bag = pd.DataFrame(bag_results)
    
    # 儲存結果
    excel_path = os.path.join(SAVE_DIR, "bag_level_predictions.xlsx")
    df_bag.to_excel(excel_path, index=False)
    
    # 4. 繪製混淆矩陣
    y_true = df_bag['true_label']
    y_pred = df_bag['pred_label']
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"Bag-First Strategy 結果 (Thr={BAG_THRESHOLD})")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    
    # 繪圖
    class_names = ["A", "B", "C"]
    
    # 數量版
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Bag Level CM (Count) - Thr={BAG_THRESHOLD}")
    plt.savefig(os.path.join(SAVE_DIR, "Bag_CM_Count.png"))
    
    # 百分比版
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap="Oranges", values_format=".2f")
    plt.title(f"Bag Level CM (Percent) - Thr={BAG_THRESHOLD}")
    plt.savefig(os.path.join(SAVE_DIR, "Bag_CM_Percent.png"))
    
    print(f"圖表已儲存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()