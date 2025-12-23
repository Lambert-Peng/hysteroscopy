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
# 請修改 Excel 路徑 (inference2.py 使用的輸入檔)
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\1210\results\best\train2_1layer_mean_20251210_010556\val_predictions.xlsx"

# 設定目標特異度 (Specificity) - 用來決定最佳閾值
TARGET_SPECIFICITY = 0.9 

# 設定 Top-k 的 k 值
TOP_K = 3

# ==========================================
# 2. 核心功能函式
# ==========================================
def extract_bag_id(filename):
    parts = str(filename).split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def get_top_k_mean(series, k=3):
    """取前 k 高的分數算平均"""
    return series.nlargest(k).mean()

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    計算二分類的詳細指標 (Normal vs Abnormal)
    """
    # y_true/pred: 0=Normal, 1=Abnormal
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
    
    metrics_list = []
    
    for i, class_label in enumerate(class_names):
        # 針對該類別 (One-vs-Rest)
        # 如果是 Class 0 (Normal): TP=cm[0,0], FP=cm[1,0], FN=cm[0,1], TN=cm[1,1]
        # 如果是 Class 1 (Abnormal): TP=cm[1,1], FP=cm[0,1], FN=cm[1,0], TN=cm[0,0]
        
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        def safe_div(n, d): return n / d if d > 0 else 0
        
        precision = safe_div(tp, tp + fp)       # PPV
        recall    = safe_div(tp, tp + fn)       # Sensitivity
        specificity = safe_div(tn, tn + fp)     # Specificity
        npv       = safe_div(tn, tn + fn)       # NPV
        f1        = safe_div(2 * precision * recall, precision + recall)
        
        # MCC
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = safe_div(numerator, denominator)
        
        metrics_list.append({
            "Class": class_label,
            "Precision (PPV)": round(precision, 4),
            "Recall (Sens)": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Specificity": round(specificity, 4),
            "NPV": round(npv, 4),
            "MCC": round(mcc, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
        
    df_metrics = pd.DataFrame(metrics_list).set_index("Class")
    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    
    return df_metrics, df_cm

def save_metrics_plot(df_metrics, title, save_path):
    """將指標表格畫成圖片儲存"""
    fig, ax = plt.subplots(figsize=(10, 3)) # 調整尺寸
    ax.axis('off')
    ax.axis('tight')
    
    table_data = df_metrics.reset_index().values
    column_labels = df_metrics.reset_index().columns
    
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(title, pad=20, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_and_plot_method(method_name, y_true, y_scores, save_dir, target_spec=0.9, excel_writer=None):
    """
    針對單一方法進行評估、繪圖、並寫入 Excel
    """
    print(f"\n>>> 正在評估: {method_name} Pooling ...")
    
    # 1. 找最佳閾值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    target_fpr = 1 - target_spec
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[-1]
        thresh = thresholds[best_idx]
    else:
        thresh = 0.5
        
    # 2. 產生預測結果
    y_pred = (y_scores >= thresh).astype(int)
    
    # 3. 計算詳細指標 (Per-class)
    class_names = ["Normal (B)", "Abnormal (A+C)"]
    df_metrics, df_cm = calculate_per_class_metrics(y_true, y_pred, class_names)
    
    # 印出 Abnormal 類的關鍵指標
    rec_ab = df_metrics.loc["Abnormal (A+C)", "Recall (Sens)"]
    spec_ab = df_metrics.loc["Abnormal (A+C)", "Specificity"]
    acc = accuracy_score(y_true, y_pred)
    print(f"  [結果] Thr: {thresh:.4f}, AUC: {roc_auc:.4f}")
    print(f"  [結果] Sens (A+C): {rec_ab:.4f}, Spec: {spec_ab:.4f}, Acc: {acc:.4f}")

    # 4. 儲存視覺化圖表
    save_metrics_plot(df_metrics, f"{method_name} Metrics (Thr={thresh:.4f})", 
                      os.path.join(save_dir, f"Metrics_Viz_{method_name}.png"))

    # 5. 寫入 Excel (如果 writer 存在)
    if excel_writer:
        sheet_name = method_name[:31] # Excel sheet name limit
        
        # 寫入 CM
        df_cm.to_excel(excel_writer, sheet_name=sheet_name, startrow=1, startcol=0)
        worksheet = excel_writer.sheets[sheet_name]
        worksheet.write_string(0, 0, f"Confusion Matrix (Thr={thresh:.4f})")
        
        # 寫入 Metrics
        df_metrics.to_excel(excel_writer, sheet_name=sheet_name, startrow=6, startcol=0)
        worksheet.write_string(5, 0, "Per-Class Metrics")

    # 6. 繪製標準圖表 (CM & Hist)
    # CM Image
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "A+C"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"MIL CM ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_{method_name}.png"), dpi=300)
    plt.close()
    
    # CM Percentage Image
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=["B", "A+C"]).plot(cmap="Oranges", values_format=".2f")
    plt.title(f"MIL CM % ({method_name})")
    plt.savefig(os.path.join(save_dir, f"CM_Percentage_{method_name}.png"), dpi=300)
    plt.close()

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(y_scores[y_true==0], bins=20, alpha=0.5, label='Normal (B)', color='green', edgecolor='k')
    plt.hist(y_scores[y_true==1], bins=20, alpha=0.5, label='Abnormal (A+C)', color='red', edgecolor='k')
    plt.axvline(thresh, color='blue', linestyle='--', linewidth=2, label=f'Thr={thresh:.2f}')
    plt.title(f"Score Dist ({method_name})")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f"Hist_{method_name}.png"), dpi=300)
    plt.close()
    
    return {
        "Method": method_name,
        "AUC": roc_auc,
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

    # 1. 資料分組與標籤修正
    df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')

    for bag_id, group in grouped:
        original_label = group['true_label'].iloc[0]
        
        # [重要] 修正標籤邏輯：1(B) 為正常(0)，其他(0/A, 2/C) 為異常(1)
        # 這樣才能正確計算 A+C 的檢出率
        if original_label == 1:
            is_abnormal = 0 # Normal
        else:
            is_abnormal = 1 # Abnormal (A or C)
        
        probs = group['prob_class0'] # 假設 class0 是異常機率 (請依實際模型調整)
        
        bag_results.append({
            'bag_id': bag_id,
            'true_label': is_abnormal,
            'prob_max': probs.max(),
            'prob_mean': probs.mean(),
            'prob_topk': get_top_k_mean(probs, TOP_K)
        })

    df_bag = pd.DataFrame(bag_results)
    # 存檔備份
    df_bag.to_excel(os.path.join(mil_save_dir, "bag_predictions_all_methods.xlsx"), index=False)
    
    y_true = df_bag['true_label'].values
    
    # 2. 執行評估並寫入 Excel
    excel_output = os.path.join(mil_save_dir, "MIL_Detailed_Metrics_Binary.xlsx")
    results = []
    
    print(f"正在生成詳細報表: {excel_output}")
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        results.append(evaluate_and_plot_method("Max", y_true, df_bag['prob_max'].values, mil_save_dir, TARGET_SPECIFICITY, writer))
        results.append(evaluate_and_plot_method("Mean", y_true, df_bag['prob_mean'].values, mil_save_dir, TARGET_SPECIFICITY, writer))
        results.append(evaluate_and_plot_method(f"Top-{TOP_K}", y_true, df_bag['prob_topk'].values, mil_save_dir, TARGET_SPECIFICITY, writer))

    # 3. 畫 ROC Overlay
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
    
    print(f"\n全部完成！報表與圖表已儲存至: {mil_save_dir}")

if __name__ == "__main__":
    main()