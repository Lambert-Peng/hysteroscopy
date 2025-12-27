import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os

# ================= 設定區 =================
# 請修改為 inference_combined.py 產出的 Excel 路徑
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\best\combined_eval_thr_0.3\combined_predictions.xlsx"

# 閾值 (要跟 inference 時一致)
BINARY_THRESHOLD = 0.3

# ================= 輔助函式 =================
def extract_bag_id(filename):
    parts = str(filename).split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    計算多類別的詳細指標 (One-vs-Rest)
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]) # 0:A, 1:B, 2:C
    
    metrics_list = []
    
    for i, class_label in enumerate(class_names):
        # One-vs-Rest 邏輯
        # TP: 對角線該類別的值
        tp = cm[i, i]
        # FN: 該類別的 Row Sum - TP
        fn = cm[i, :].sum() - tp
        # FP: 該類別的 Col Sum - TP
        fp = cm[:, i].sum() - tp
        # TN: 總數 - (TP + FP + FN)
        tn = cm.sum() - (tp + fp + fn)
        
        # 避免除以零
        def safe_div(n, d): return n / d if d > 0 else 0
        
        precision = safe_div(tp, tp + fp)       # PPV
        recall    = safe_div(tp, tp + fn)       # Sensitivity / TPR
        specificity = safe_div(tn, tn + fp)     # TNR
        npv       = safe_div(tn, tn + fn)       # NPV
        f1        = safe_div(2 * precision * recall, precision + recall)
        
        # MCC Calculation
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # 繪製表格
    table_data = df_metrics.reset_index().values
    column_labels = df_metrics.reset_index().columns
    
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ================= 主程式 =================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到檔案 {EXCEL_PATH}")
        return
        
    print(f"正在進行 Bag Level 詳細評估: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    save_dir = os.path.dirname(EXCEL_PATH)
    
    # 1. 提取 Bag ID
    if 'bag_id' not in df.columns:
        df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')
    
    print(f"共發現 {len(grouped)} 位病人 (Bags)")

    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # --- 策略 1: Max Pooling ---
        preds = group['M1_pred'].values
        if np.any(preds != 1): # 有任何一張不是 B
            sum_prob_A = group['prob_A'].sum()
            sum_prob_C = group['prob_C'].sum()
            final_pred_max = 0 if sum_prob_A > sum_prob_C else 2
        else:
            final_pred_max = 1 # B
            
        # --- 策略 2: Mean Pooling ---
        mean_pA = group['prob_A'].mean()
        mean_pC = group['prob_C'].mean()
        # 這裡我們用 A+C 的總和來判斷異常 (M1 邏輯)
        # 注意: 這裡需要確保 prob 欄位是 Method 1 的結果
        mean_p_Abnormal = mean_pA + mean_pC
        
        if mean_p_Abnormal > BINARY_THRESHOLD:
            final_pred_mean = 0 if mean_pA > mean_pC else 2
        else:
            final_pred_mean = 1 # B

        bag_results.append({
            'bag_id': bag_id,
            'true_label': true_label,
            'pred_max': final_pred_max,
            'pred_mean': final_pred_mean
        })
        
    df_bag = pd.DataFrame(bag_results)
    class_names = ["A", "B", "C"]
    
    # --- 計算詳細指標 ---
    metrics_max, cm_max_df = calculate_per_class_metrics(df_bag['true_label'], df_bag['pred_max'], class_names)
    metrics_mean, cm_mean_df = calculate_per_class_metrics(df_bag['true_label'], df_bag['pred_mean'], class_names)
    
    # --- 顯示與儲存結果 ---
    print("\n" + "="*60)
    print("【Max Pooling 詳細指標】")
    print(metrics_max)
    print("\n【Mean Pooling 詳細指標】")
    print(metrics_mean)
    print("="*60)
    
    # 1. 儲存為 Excel (包含 CM 和 Metrics)
    excel_output = os.path.join(save_dir, "MIL_Detailed_Metrics.xlsx")
    
    # [修改點] 這裡加上 engine='xlsxwriter'
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        
        # Max Pooling Sheet
        cm_max_df.to_excel(writer, sheet_name="Max_Pooling", startrow=1, startcol=0) # 改 startrow=1 留標題空間
        
        # 使用 xlsxwriter 的 worksheet 物件
        workbook = writer.book
        worksheet_max = writer.sheets["Max_Pooling"]
        
        # 寫入標題 (xlsxwriter 是 0-based index: row 0, col 0)
        worksheet_max.write_string(0, 0, "Confusion Matrix")
        
        metrics_max.to_excel(writer, sheet_name="Max_Pooling", startrow=7, startcol=0)
        worksheet_max.write_string(6, 0, "Per-Class Metrics")
        
        # Mean Pooling Sheet
        cm_mean_df.to_excel(writer, sheet_name="Mean_Pooling", startrow=1, startcol=0)
        worksheet_mean = writer.sheets["Mean_Pooling"]
        
        worksheet_mean.write_string(0, 0, "Confusion Matrix")
        
        metrics_mean.to_excel(writer, sheet_name="Mean_Pooling", startrow=7, startcol=0)
        worksheet_mean.write_string(6, 0, "Per-Class Metrics")
        
    print(f"\n詳細數據已儲存至 Excel: {excel_output}")
    
    # 2. 儲存視覺化圖表 (圖片檔)
    save_metrics_plot(metrics_max, f"Max Pooling Metrics (Thr={BINARY_THRESHOLD})", 
                      os.path.join(save_dir, "MIL_Metrics_Viz_Max.png"))
    save_metrics_plot(metrics_mean, f"Mean Pooling Metrics (Thr={BINARY_THRESHOLD})", 
                      os.path.join(save_dir, "MIL_Metrics_Viz_Mean.png"))
                      
    print(f"視覺化圖表已儲存: MIL_Metrics_Viz_Max.png, MIL_Metrics_Viz_Mean.png")

    # 3. 繪製原本的混淆矩陣圖 (保持原本功能)
    cm_max = confusion_matrix(df_bag['true_label'], df_bag['pred_max'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_max, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Bag CM (Max) - Counts")
    plt.savefig(os.path.join(save_dir, "MIL_CM_Max_Count.png"))
    plt.close()

if __name__ == "__main__":
    main()