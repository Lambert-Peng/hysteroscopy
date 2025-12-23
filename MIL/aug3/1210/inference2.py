import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score, 
    matthews_corrcoef, f1_score, recall_score,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. 設定區域
# ==========================================
# 請修改 Excel 路徑
EXCEL_PATH = r"results/train2m_20251211_222442/val_predictions.xlsx" 

# 設定目標特異度 (Specificity)
TARGET_SPECIFICITY = 0.9 

# 設定 Top-k 的 k 值 (取前 k 高的分數平均)
TOP_K = 3

# ==========================================
# 2. 核心功能函式
# ==========================================
def extract_bag_id(filename):
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return filename

def get_top_k_mean(series, k=3):
    """取前 k 高的分數算平均"""
    return series.nlargest(k).mean()

def evaluate_and_plot_method(method_name, y_true, y_scores, save_dir, target_spec=0.9):
    """
    針對單一方法 (如 Max, Mean) 進行完整評估並畫圖
    """
    print(f"\n>>> 正在評估: {method_name} Pooling ...")
    
    # 1. ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 2. 找最佳閾值 (基於 Target Specificity)
    target_fpr = 1 - target_spec
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[-1]
        thresh = thresholds[best_idx]
    else:
        thresh = 0.5
        
    # 3. 計算指標
    y_pred = (y_scores >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv= tp / (tp + fp) if (tp + fp) > 0 else 0
    npv= tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # 印出數據
    print(f"  [結果] Threshold: {thresh:.4f}")
    print(f"  [結果] AUC: {roc_auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}, Acc: {acc:.4f}")
    
    # --- 繪圖 2: Confusion Matrix ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "A+C"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"MIL Confusion Matrix ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_{method_name}.png"), dpi=300)
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=["B", "A+C"]).plot(cmap="Oranges", values_format=".2f")
    plt.title(f"MIL Confusion Matrix ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_Percentage_{method_name}.png"), dpi=300)
    plt.close()
    # --- 繪圖 3: Score Distribution ---
    plt.figure(figsize=(6, 4))
    plt.hist(y_scores[y_true==0], bins=20, alpha=0.5, label='Normal (B)', color='green', edgecolor='k')
    plt.hist(y_scores[y_true==1], bins=20, alpha=0.5, label='Abnormal (A+C)', color='red', edgecolor='k')
    plt.axvline(thresh, color='blue', linestyle='--', linewidth=2, label=f'Thr={thresh:.2f}')
    plt.title(f"Score Distribution ({method_name})")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f"Hist_{method_name}.png"), dpi=300)
    plt.close()
    
    return {
        "Method": method_name,
        "AUC": roc_auc,
        "Sens": sens,
        "Spec": spec,
        "Acc": acc,
        "Threshold": thresh,
        "fpr": fpr,
        "tpr": tpr
    }

# ==========================================
# 3. 主程式
# ==========================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到檔案 {EXCEL_PATH}")
        return

    save_dir = os.path.dirname(EXCEL_PATH)
    mil_save_dir = os.path.join(save_dir, "MIL_Comparison_All")
    os.makedirs(mil_save_dir, exist_ok=True)
    
    print(f"讀取: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # 1. 資料分組
    df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')

    for bag_id, group in grouped:
        original_label = group['true_label'].iloc[0]
        is_abnormal = 1 if original_label == 0 else 0
        
        # 取得該 Bag 所有預測分數
        probs = group['prob_class0']
        
        bag_results.append({
            'bag_id': bag_id,
            'true_label': is_abnormal,
            'prob_max': probs.max(),                   # Max Pooling
            'prob_mean': probs.mean(),                 # Mean Pooling
            'prob_topk': get_top_k_mean(probs, TOP_K)  # Top-k Pooling
        })

    df_bag = pd.DataFrame(bag_results)
    df_bag.to_excel(os.path.join(mil_save_dir, "bag_predictions_all_methods.xlsx"), index=False)
    
    y_true = df_bag['true_label'].values
    
    # 2. 執行三種方法的評估
    results = []
    results.append(evaluate_and_plot_method("Max", y_true, df_bag['prob_max'].values, mil_save_dir, TARGET_SPECIFICITY))
    results.append(evaluate_and_plot_method("Mean", y_true, df_bag['prob_mean'].values, mil_save_dir, TARGET_SPECIFICITY))
    results.append(evaluate_and_plot_method(f"Top-{TOP_K}", y_true, df_bag['prob_topk'].values, mil_save_dir, TARGET_SPECIFICITY))
    
    # 3. 畫綜合比較圖 (Overlay ROC)
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'purple']
    for i, res in enumerate(results):
        plt.plot(res['fpr'], res['tpr'], color=colors[i], lw=2, 
                 label=f"{res['Method']} (AUC={res['AUC']:.3f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of MIL Aggregation Methods')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(mil_save_dir, "Compare_ROC_Overlay.png"), dpi=300)
    plt.close()
    
    # 4. 印出總表
    print("\n" + "="*60)
    print(f"{'Method':<10} | {'AUC':<8} | {'Sens':<8} | {'Spec':<8} | {'Acc':<8} | {'Thr':<8}")
    print("-" * 60)
    for res in results:
        print(f"{res['Method']:<10} | {res['AUC']:.4f}   | {res['Sens']:.4f}   | {res['Spec']:.4f}   | {res['Acc']:.4f}   | {res['Threshold']:.4f}")
    print("="*60)
    print(f"\n全部完成！圖表已儲存至: {mil_save_dir}")

if __name__ == "__main__":
    main()