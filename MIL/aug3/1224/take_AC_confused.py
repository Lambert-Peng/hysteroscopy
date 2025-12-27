import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import shutil
from tqdm import tqdm

# ================= 設定區 (請務必修改這裡) =================
# 1. Excel 路徑 (inference_combine.py 產出的檔案)
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\best\combined_eval_thr_0.3\combined_predictions.xlsx"

# 2. 原始圖片的「根目錄」 (程式會去這裡抓圖)
SOURCE_IMG_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\hysteroscopy_dataset"

# 3. 輸出結果與圖片的資料夾 (預設放在 Excel 同級目錄下的 Top3_Errors)
# 您可以不用改，程式會自動建立
BASE_SAVE_DIR = os.path.dirname(EXCEL_PATH)
TARGET_IMG_DIR = os.path.join(BASE_SAVE_DIR, "Top3_AC_Confusion_Images")

# 4. 參數設定
BINARY_THRESHOLD = 0.45
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
    """計算多類別詳細指標"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df_metrics.reset_index().values, colLabels=df_metrics.reset_index().columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.5)
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm_both(y_true, y_pred, class_names, method_name, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Count
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format='d', ax=ax)
    plt.title(f"Bag CM Count ({method_name})")
    plt.savefig(os.path.join(save_dir, f"MIL_CM_Count_{method_name}.png"), dpi=300); plt.close()

    # Percentage
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(cmap="Oranges", values_format=".2f", ax=ax)
    plt.title(f"Bag CM Percent ({method_name})")
    plt.savefig(os.path.join(save_dir, f"MIL_CM_Percent_{method_name}.png"), dpi=300); plt.close()

def copy_images_for_bags(bag_list, subfolder_name):
    """複製指定 Bag ID 的所有圖片到目標資料夾"""
    target_sub_dir = os.path.join(TARGET_IMG_DIR, subfolder_name)
    if os.path.exists(target_sub_dir): shutil.rmtree(target_sub_dir)
    os.makedirs(target_sub_dir)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    count = 0
    missing = []
    
    print(f"  正在複製至: {subfolder_name} ...")
    for bag_id in tqdm(bag_list):
        bag_id = str(bag_id).strip()
        # 巢狀路徑解析 A63_A4 -> A63/A4
        relative_path = bag_id.replace("_", os.sep)
        src_path = os.path.join(SOURCE_IMG_DIR, relative_path)
        
        # 如果巢狀找不到，試試扁平
        if not (os.path.exists(src_path) and os.path.isdir(src_path)):
            src_path = os.path.join(SOURCE_IMG_DIR, bag_id)
            
        dst_path = os.path.join(target_sub_dir, bag_id)
        
        if os.path.exists(src_path) and os.path.isdir(src_path):
            try:
                if not os.path.exists(dst_path): os.makedirs(dst_path)
                copied_files = 0
                for f in os.listdir(src_path):
                    if f.lower().endswith(valid_extensions):
                        shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, f))
                        copied_files += 1
                if copied_files > 0: count += 1
                else: os.rmdir(dst_path) # 空資料夾移除
            except Exception as e: print(f"    Copy Error {bag_id}: {e}")
        else:
            missing.append(bag_id)
    return count, missing

# ================= 主程式 =================
def main():
    if not os.path.exists(EXCEL_PATH): print(f"錯誤：找不到檔案 {EXCEL_PATH}"); return
    if not os.path.exists(SOURCE_IMG_DIR): print(f"錯誤：找不到圖片目錄 {SOURCE_IMG_DIR}"); return
        
    print(f"1. 讀取與計算 Bag Level (Top-{TOP_K})...")
    df = pd.read_excel(EXCEL_PATH)
    if 'bag_id' not in df.columns: df['bag_id'] = df['filename'].apply(extract_bag_id)
    
    bag_results = []
    grouped = df.groupby('bag_id')

    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # 計算 Top-K 異常分數
        # 假設 prob_A, prob_C 代表異常機率
        p_abnormal_instances = group['prob_A'].values + group['prob_C'].values
        bag_score_topk = get_top_k_mean(p_abnormal_instances, k=TOP_K)
        
        if bag_score_topk > BINARY_THRESHOLD:
            # 判為異常，比較整包 Mean A vs Mean C
            mean_pA = group['prob_A'].mean()
            mean_pC = group['prob_C'].mean()
            final_pred_topk = 0 if mean_pA > mean_pC else 2
        else:
            final_pred_topk = 1 # B
            
        bag_results.append({
            'bag_id': bag_id, 'true_label': true_label, 'pred_topk': final_pred_topk
        })
        
    df_bag = pd.DataFrame(bag_results)
    
    # 2. 生成指標與圖表
    print(f"2. 生成 Top-{TOP_K} 評估報表...")
    class_names = ["A", "B", "C"]
    metrics, cm_df = calculate_per_class_metrics(df_bag['true_label'], df_bag['pred_topk'], class_names)
    
    # 儲存 Excel
    excel_out = os.path.join(BASE_SAVE_DIR, f"MIL_Metrics_Top{TOP_K}.xlsx")
    with pd.ExcelWriter(excel_out, engine='xlsxwriter') as writer:
        cm_df.to_excel(writer, sheet_name=f"Top{TOP_K}", startrow=1)
        writer.sheets[f"Top{TOP_K}"].write_string(0, 0, "Confusion Matrix")
        metrics.to_excel(writer, sheet_name=f"Top{TOP_K}", startrow=7)
        writer.sheets[f"Top{TOP_K}"].write_string(6, 0, "Per-Class Metrics")
    
    # 儲存圖片
    save_metrics_plot(metrics, f"Top-{TOP_K} Metrics (Thr={BINARY_THRESHOLD})", os.path.join(BASE_SAVE_DIR, f"MIL_Metrics_Viz_Top{TOP_K}.png"))
    plot_cm_both(df_bag['true_label'], df_bag['pred_topk'], class_names, f"Top{TOP_K}", BASE_SAVE_DIR)
    
    print(f"   報表已儲存至: {BASE_SAVE_DIR}")

    # 3. 複製混淆影像
    print(f"\n3. 開始提取 Top-{TOP_K} 錯誤影像 (AC Confusion)...")
    
    # 篩選 A -> C
    df_A_to_C = df_bag[(df_bag['true_label'] == 0) & (df_bag['pred_topk'] == 2)]
    if len(df_A_to_C) > 0:
        cnt, miss = copy_images_for_bags(set(df_A_to_C['bag_id']), "Top3_True_A_Pred_C")
        print(f"   [True A -> Pred C] 複製完成: {cnt} 位, 未找到: {len(miss)}")
        
    # 篩選 C -> A
    df_C_to_A = df_bag[(df_bag['true_label'] == 2) & (df_bag['pred_topk'] == 0)]
    if len(df_C_to_A) > 0:
        cnt, miss = copy_images_for_bags(set(df_C_to_A['bag_id']), "Top3_True_C_Pred_A")
        print(f"   [True C -> Pred A] 複製完成: {cnt} 位, 未找到: {len(miss)}")

    print("\n" + "="*50)
    print(f"全部完成！")
    print(f"影像已儲存至: {TARGET_IMG_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()