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
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ================= 設定區 (Critical Tuning) =================
# [關鍵參數] 二分類篩檢閾值
# 設為 0.3 代表：只要模型覺得異常機率 > 30%，就判為異常 (進入 AC 分類)
# 如果發現 A/C 還是被判成 B，請繼續調低這個值 (e.g., 0.2, 0.15)
BINARY_THRESHOLD = 0.30 

# 模型路徑
BINARY_MODEL_PATH = "1210/results/best/train2_1layer_mean_20251210_010556/best_model_binary.pt"
AC_MODEL_PATH = "results/train_AC_specific_20251223_170912/best_model_AC.pt"
TRICLASS_MODEL_PATH = "1210/results/best/train3_1layer_mean_20251210_021143/best_model_3class.pt"

VAL_DATA_DIR = "dataset/val/weights"
AC_MODEL_DIR = os.path.dirname(AC_MODEL_PATH)
SAVE_DIR = os.path.join(AC_MODEL_DIR, f"combined_eval_thr_{BINARY_THRESHOLD}")
os.makedirs(SAVE_DIR, exist_ok=True)
EXCEL_OUTPUT = os.path.join(SAVE_DIR, "combined_predictions.xlsx")

print(f"結果將儲存至: {SAVE_DIR}")
CLASS_NAMES = ["A", "B", "C"]
IDX_A, IDX_B, IDX_C = 0, 1, 2 

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

# ================= 繪圖函式 =================
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    # 數量版
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format='d', ax=ax)
    plt.title(f"{title} (Counts)")
    plt.savefig(os.path.join(SAVE_DIR, f"{filename}_counts.png"), dpi=300)
    plt.close()
    # 百分比版
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["A", "B", "C"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp_norm.plot(cmap="Oranges", values_format=".2f", ax=ax)
    plt.title(f"{title} (Percentage)")
    plt.savefig(os.path.join(SAVE_DIR, f"{filename}_percentage.png"), dpi=300)
    plt.close()

def plot_multiclass_roc(y_true, y_score, title, filename):
    n_classes = 3
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle=':', lw=4,
             label=f'Macro-average (AUC = {macro_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{title} ROC Curve')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, f"{filename}_ROC.png"), dpi=300)
    plt.close()
    return macro_auc

# ================= 主程式 =================
def main():
    model_bin = load_model(BINARY_MODEL_PATH, num_classes=2) 
    model_ac  = load_model(AC_MODEL_PATH, num_classes=3)     
    model_tri = load_model(TRICLASS_MODEL_PATH, num_classes=3)

    dataset = RecursiveFeatureDataset(VAL_DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    results = []
    y_true_all, y_probs_m1, y_probs_m2, y_probs_old = [], [], [], []

    print(f"開始推論 (使用閾值: {BINARY_THRESHOLD})...")
    with torch.no_grad():
        for features, label, filenames in loader:
            features = features.to(device)
            y_true_all.extend(label.numpy())
            
            # --- A. 二分類模型 ---
            logits_bin = model_bin(features)
            probs_bin = F.softmax(logits_bin, dim=1).cpu().numpy()
            p_AC_stage1 = probs_bin[:, 0] # P(Abnormal)
            p_B_stage1  = probs_bin[:, 1] # P(Normal)
            
            # --- B. AC 專用模型 ---
            logits_ac = model_ac(features).cpu().numpy()
            logit_A, logit_C = logits_ac[:, IDX_A], logits_ac[:, IDX_C]
            exp_A, exp_C = np.exp(logit_A), np.exp(logit_C)
            sum_exp = exp_A + exp_C + 1e-8
            p_A_cond_m1 = exp_A / sum_exp
            p_C_cond_m1 = exp_C / sum_exp

            # --- C. 舊模型 ---
            logits_tri = model_tri(features)
            probs_tri = F.softmax(logits_tri, dim=1).cpu().numpy()
            y_probs_old.extend(probs_tri)
            
            p_A_tri, p_C_tri = probs_tri[:, IDX_A], probs_tri[:, IDX_C]
            sum_tri = p_A_tri + p_C_tri + 1e-8
            p_A_cond_m2 = p_A_tri / sum_tri
            p_C_cond_m2 = p_C_tri / sum_tri

            # --- D. 結合計算 (關鍵邏輯修改) ---
            batch_size = len(filenames)
            for i in range(batch_size):
                
                # ==== Method 1: Hierarchical with Threshold ====
                # 只有當 "異常機率" > 閾值 時，才允許判成 A 或 C
                if p_AC_stage1[i] > BINARY_THRESHOLD:
                    # 強制判為異常，看 AC 模型覺得是誰
                    # 這裡我們手動將 B 的機率壓低，重新分配給 A 和 C
                    final_pB = 0.0 # 強制排除 B
                    final_pA = p_A_cond_m1[i]
                    final_pC = p_C_cond_m1[i]
                else:
                    # 判為正常
                    final_pB = 1.0
                    final_pA = 0.0
                    final_pC = 0.0
                
                y_probs_m1.append([final_pA, final_pB, final_pC])
                
                # ==== Method 2: Ensemble (保留原樣或也加閾值) ====
                m2_pB = p_B_stage1[i]
                m2_pA = p_AC_stage1[i] * p_A_cond_m2[i]
                m2_pC = p_AC_stage1[i] * p_C_cond_m2[i]
                y_probs_m2.append([m2_pA, m2_pB, m2_pC])
                
                results.append({
                    "filename": filenames[i],
                    "true_label": label[i].item(),
                    
                    # --- 預測類別 (你現在有的) ---
                    "M1_pred": np.argmax([final_pA, final_pB, final_pC]),
                    "M2_pred": np.argmax([m2_pA, m2_pB, m2_pC]),

                    # --- [必須補上] 預測機率 (為了下一步 Mean Pooling) ---
                    # 這裡我們儲存 Method 1 的機率，因為這是你的主力方法
                    "prob_A": final_pA,
                    "prob_B": final_pB,
                    "prob_C": final_pC
                })

    y_true = np.array(y_true_all)
    probs_m1 = np.array(y_probs_m1)
    probs_m2 = np.array(y_probs_m2)
    probs_old = np.array(y_probs_old)
    pred_m1 = np.argmax(probs_m1, axis=1)
    pred_m2 = np.argmax(probs_m2, axis=1)
    pred_old = np.argmax(probs_old, axis=1)

    # 繪圖
    print("\n>>> 正在繪製圖表...")
    plot_confusion_matrix(y_true, pred_old, "Baseline", "Baseline")
    plot_confusion_matrix(y_true, pred_m1, f"Method 1 (Thr={BINARY_THRESHOLD})", "Method1")
    plot_confusion_matrix(y_true, pred_m2, "Method 2", "Method2")
    
    auc_old = plot_multiclass_roc(y_true, probs_old, "Baseline", "Baseline")
    auc_m1 = plot_multiclass_roc(y_true, probs_m1, f"Method 1 (Thr={BINARY_THRESHOLD})", "Method1")
    auc_m2 = plot_multiclass_roc(y_true, probs_m2, "Method 2", "Method2")

    print("\n" + "="*60)
    print(f"閾值 Threshold 設定為: {BINARY_THRESHOLD}")
    print(f"{'Method':<25} | {'Acc':<8} | {'AUC':<8}")
    print("-" * 60)
    print(f"{'Baseline':<25} | {accuracy_score(y_true, pred_old):.4f}   | {auc_old:.4f}")
    print(f"{'Method 1 (Thresholding)':<25} | {accuracy_score(y_true, pred_m1):.4f}   | {auc_m1:.4f}")
    print(f"{'Method 2 (Ensemble)':<25} | {accuracy_score(y_true, pred_m2):.4f}   | {auc_m2:.4f}")
    print("="*60)
    pd.DataFrame(results).to_excel(EXCEL_OUTPUT, index=False)

if __name__ == "__main__":
    main()