import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, roc_curve, auc
)
import os

# ================= 設定區 =================
# 請修改為 inference_combined.py 產出的 Excel 路徑
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\best\combined_eval_thr_0.3\combined_predictions.xlsx"

# 閾值 (要跟 inference 時一致，用於決定 B vs A+C)
BINARY_THRESHOLD = [0.64, 0.4, 0.45]

# Top-K 設定
TOP_K = 3

# ================= 輔助函式 =================
def extract_bag_id(filename):
    parts = str(filename).split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def get_top_k_mean(scores, k=3):
    """計算前 K 高分的平均值"""
    if len(scores) == 0: return 0.0
    scores = np.array(scores)
    if len(scores) < k:
        return np.mean(scores)
    return np.mean(np.sort(scores)[-k:])

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """計算多類別的詳細指標"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]) # 0:A, 1:B, 2:C
    metrics_list = []
    
    for i, class_label in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        def safe_div(n, d): return n / d if d > 0 else 0
        
        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        npv       = safe_div(tn, tn + fn)
        f1        = safe_div(2 * precision * recall, precision + recall)
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = safe_div(numerator, denominator)
        
        metrics_list.append({
            "Class": class_label,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Spec": round(specificity, 4),
            "NPV": round(npv, 4),
            "MCC": round(mcc, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
        
    df_metrics = pd.DataFrame(metrics_list).set_index("Class")
    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    return df_metrics, df_cm

def save_metrics_plot(df_metrics, title, save_path):
    """將指標表格存為圖片"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df_metrics.reset_index().values, colLabels=df_metrics.reset_index().columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm_both(y_true, y_pred, class_names, method_name, save_dir):
    """同時畫 Count 和 Percentage 混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # 1. Count
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format='d', ax=ax)
    plt.title(f"Bag CM Count ({method_name})")
    plt.savefig(os.path.join(save_dir, f"MIL_CM_Count_{method_name}.png"), dpi=300)
    plt.close()

    # 2. Percentage
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(cmap="Oranges", values_format=".2f", ax=ax)
    plt.title(f"Bag CM Percent ({method_name})")
    plt.savefig(os.path.join(save_dir, f"MIL_CM_Percent_{method_name}.png"), dpi=300)
    plt.close()

# ================= 主程式 =================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到檔案 {EXCEL_PATH}")
        return
        
    print(f"正在進行 Bag Level 評估 (含 Top-{TOP_K} & ROC): {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    save_dir = os.path.dirname(EXCEL_PATH)
    
    if 'bag_id' not in df.columns:
        df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')
    
    # 用來畫 ROC 的資料 (Binary: Normal=0, Abnormal=1)
    roc_data = {
        'y_true_bin': [],
        'score_max': [],
        'score_mean': [],
        'score_topk': []
    }

    print(f"處理 {len(grouped)} 位病人...")

    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # 準備 Instance Level 的異常機率 (A+C)
        # 注意：假設 prob_A, prob_B, prob_C 已經存在
        # 如果是二分類模型，通常看 1 - prob_B 或是 prob_A + prob_C
        p_abnormal_instances = group['prob_A'].values + group['prob_C'].values
        
        # --- 1. Max Pooling ---
        bag_score_max = np.max(p_abnormal_instances)
        if bag_score_max > BINARY_THRESHOLD[0]:
            # 判異常，決定是 A 還是 C (用總和比較)
            final_pred_max = 0 if group['prob_A'].sum() > group['prob_C'].sum() else 2
        else:
            final_pred_max = 1 # B
            
        # --- 2. Mean Pooling ---
        bag_score_mean = np.mean(p_abnormal_instances)
        if bag_score_mean > BINARY_THRESHOLD[1]:
            final_pred_mean = 0 if group['prob_A'].mean() > group['prob_C'].mean() else 2
        else:
            final_pred_mean = 1 # B
            
        # --- 3. Top-K Pooling (新增) ---
        bag_score_topk = get_top_k_mean(p_abnormal_instances, k=TOP_K)
        
        if bag_score_topk > BINARY_THRESHOLD[2]:
            # 策略：取異常分數最高的前 K 張，看它們平均是偏 A 還是偏 C
            # 這裡簡單起見，如果判異常，還是用整包的 Mean 來分 A/C，
            # 或是只用 Top-K 的 instance 來分。這裡使用整包平均以保持穩定。
            mean_pA = group['prob_A'].mean()
            mean_pC = group['prob_C'].mean()
            final_pred_topk = 0 if mean_pA > mean_pC else 2
        else:
            final_pred_topk = 1 # B

        bag_results.append({
            'bag_id': bag_id,
            'true_label': true_label,
            'pred_max': final_pred_max,
            'pred_mean': final_pred_mean,
            'pred_topk': final_pred_topk
        })
        
        # 收集 ROC 數據
        # True Label: B(1) -> 0, A(0)/C(2) -> 1
        is_abnormal_gt = 0 if true_label == 1 else 1
        roc_data['y_true_bin'].append(is_abnormal_gt)
        roc_data['score_max'].append(bag_score_max)
        roc_data['score_mean'].append(bag_score_mean)
        roc_data['score_topk'].append(bag_score_topk)
        
    df_bag = pd.DataFrame(bag_results)
    class_names = ["A", "B", "C"]
    
    # --- 計算並儲存詳細指標 (Max, Mean, Top-K) ---
    methods = {
        "Max": df_bag['pred_max'],
        "Mean": df_bag['pred_mean'],
        f"Top-{TOP_K}": df_bag['pred_topk']
    }
    
    excel_output = os.path.join(save_dir, "MIL_Detailed_Metrics_Comparison.xlsx")
    print(f"\n生成報表: {excel_output}")
    
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        for method_name, preds in methods.items():
            print(f"處理 {method_name} Pooling...")
            metrics, cm_df = calculate_per_class_metrics(df_bag['true_label'], preds, class_names)
            
            # 存 Excel
            sheet_name = method_name[:31]
            cm_df.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0)
            worksheet = writer.sheets[sheet_name]
            worksheet.write_string(0, 0, f"Confusion Matrix ({method_name})")
            
            metrics.to_excel(writer, sheet_name=sheet_name, startrow=7, startcol=0)
            worksheet.write_string(6, 0, "Per-Class Metrics")
            
            # 存圖片 (Metrics Table & CM)
            save_metrics_plot(metrics, f"{method_name} Metrics", 
                              os.path.join(save_dir, f"MIL_Metrics_Viz_{method_name}.png"))
            plot_cm_both(df_bag['true_label'], preds, class_names, method_name, save_dir)

    # --- [新增] 繪製 ROC 比較圖 ---
    print("\n正在繪製綜合 ROC 比較圖 (Binary: Normal vs Abnormal)...")
    plt.figure(figsize=(8, 6))
    
    y_true_bin = np.array(roc_data['y_true_bin'])
    plot_configs = [
        ('Max', roc_data['score_max'], 'red'),
        ('Mean', roc_data['score_mean'], 'blue'),
        (f'Top-{TOP_K}', roc_data['score_topk'], 'green')
    ]
    
    for label, scores, color in plot_configs:
        fpr, tpr, _ = roc_curve(y_true_bin, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison: MIL Aggregation Strategies')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    roc_save_path = os.path.join(save_dir, "MIL_Comparison_ROC.png")
    plt.savefig(roc_save_path, dpi=300)
    plt.close()
    
    print(f"ROC 圖已儲存至: {roc_save_path}")
    print("全部完成！")

if __name__ == "__main__":
    main()