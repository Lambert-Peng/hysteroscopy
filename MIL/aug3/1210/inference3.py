import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
    f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. 設定區域
# ==========================================
# 請修改成 train3_class 產出的 Excel 路徑
EXCEL_PATH = r"results/best/train3_mean_20251209_235525/val_predictions_3class.xlsx"

TOP_K = 3
CLASS_NAMES = ["A", "B", "C"] # 對應 prob_0, prob_1, prob_2

# ==========================================
# 2. 輔助函式
# ==========================================
def extract_bag_id(filename):
    parts = filename.split('_')
    if len(parts) >= 2: return f"{parts[0]}_{parts[1]}"
    else: return filename

def get_top_k_mean(series, k=3):
    return series.nlargest(k).mean()

def evaluate_and_plot_method(method_name, y_true, y_pred_probs, save_dir):
    """
    評估三分類 MIL 結果並繪製對應圖表 (與 inference2 保持一致)
    y_pred_probs: shape (N, 3) -> [score_A, score_B, score_C]
    """
    print(f"\n>>> 評估方法: {method_name} Pooling")
    
    # 1. 決定最終類別 (Argmax)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 2. 計算基礎指標
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. 計算 Macro-Average ROC (用來畫出一條代表該方法的曲線)
    # 將標籤轉為 One-hot
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    # 計算每一類的 FPR/TPR
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_pred_probs[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    print(f"  [結果] Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, Macro AUC: {macro_auc:.4f}")
    
    # --- 繪圖 2: Confusion Matrix (數量版 - 藍色) ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"MIL Confusion Matrix ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_{method_name}.png"), dpi=300)
    plt.close()

    # --- 繪圖 3: Confusion Matrix (百分比版 - 橘色) ---
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # 避免除以 0
    cm_norm = cm.astype("float") / row_sums
    
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(cmap="Oranges", values_format=".2f")
    plt.title(f"MIL Confusion Matrix ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_Percentage_{method_name}.png"), dpi=300)
    plt.close()
    
    return {
        "Method": method_name, 
        "Acc": acc, 
        "F1": f1, 
        "AUC": macro_auc,
        "fpr": all_fpr,
        "tpr": mean_tpr
    }

# ==========================================
# 3. 主程式
# ==========================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到檔案 {EXCEL_PATH}")
        return

    save_dir = os.path.dirname(EXCEL_PATH)
    mil_save_dir = os.path.join(save_dir, "MIL_3Class_Comparison")
    os.makedirs(mil_save_dir, exist_ok=True)
    
    print(f"讀取: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')

    # --- MIL Aggregation ---
    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # 取得三欄機率 (A, B, C)
        # 注意: train3_class 輸出的欄位是 prob_A, prob_B, prob_C
        p0, p1, p2 = group['prob_A'], group['prob_B'], group['prob_C']
        
        # 1. Max Pooling: 對每一類取最大值
        max_scores = [p0.max(), p1.max(), p2.max()]
        
        # 2. Mean Pooling: 對每一類取平均值
        mean_scores = [p0.mean(), p1.mean(), p2.mean()]
        
        # 3. Top-K Pooling: 對每一類取前 K 大平均
        topk_scores = [get_top_k_mean(p0, TOP_K), get_top_k_mean(p1, TOP_K), get_top_k_mean(p2, TOP_K)]
        
        bag_results.append({
            'bag_id': bag_id,
            'true_label': true_label,
            'max_scores': max_scores,
            'mean_scores': mean_scores,
            'topk_scores': topk_scores
        })

    # 輸出 Bag Level Excel
    # 這裡將 max_scores 拆開成三個欄位以便 Excel 閱讀
    export_data = []
    for r in bag_results:
        row = {'bag_id': r['bag_id'], 'true_label': r['true_label']}
        row.update({f'max_prob_{c}': r['max_scores'][i] for i, c in enumerate(CLASS_NAMES)})
        row.update({f'mean_prob_{c}': r['mean_scores'][i] for i, c in enumerate(CLASS_NAMES)})
        row.update({f'topk_prob_{c}': r['topk_scores'][i] for i, c in enumerate(CLASS_NAMES)})
        export_data.append(row)
    
    pd.DataFrame(export_data).to_excel(os.path.join(mil_save_dir, "bag_predictions_all_methods.xlsx"), index=False)
    
    # 準備數據進行評估
    y_true = np.array([r['true_label'] for r in bag_results])
    max_probs = np.array([r['max_scores'] for r in bag_results])
    mean_probs = np.array([r['mean_scores'] for r in bag_results])
    topk_probs = np.array([r['topk_scores'] for r in bag_results])
    
    # 執行三種方法的評估
    results = []
    print("\n>>> 開始評估 MIL 聚合方法...")
    results.append(evaluate_and_plot_method("Max", y_true, max_probs, mil_save_dir))
    results.append(evaluate_and_plot_method("Mean", y_true, mean_probs, mil_save_dir))
    results.append(evaluate_and_plot_method(f"Top-{TOP_K}", y_true, topk_probs, mil_save_dir))
    
    # --- 繪圖 4: 綜合比較圖 (Overlay ROC) ---
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'purple']
    for i, res in enumerate(results):
        plt.plot(res['fpr'], res['tpr'], color=colors[i], lw=2, 
                 label=f"{res['Method']} (AUC={res['AUC']:.3f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of MIL Methods (Macro-Average ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(mil_save_dir, "Compare_ROC_Overlay.png"), dpi=300)
    plt.close()

    # --- 終端機比較表 ---
    print("\n" + "="*50)
    print(f"MIL 三分類結果比較 (Bag Level)")
    print("="*50)
    print(f"{'Method':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Macro AUC':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['Method']:<10} | {res['Acc']:.4f}     | {res['F1']:.4f}     | {res['AUC']:.4f}")
    print("="*50)
    print(f"圖表與結果已儲存至: {mil_save_dir}")

if __name__ == "__main__":
    main()