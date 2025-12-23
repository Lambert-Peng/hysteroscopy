import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay,
    recall_score
)
import os

# ==========================================
# 1. 設定區域
# ==========================================
# 請修改 Excel 路徑 (由 train2_1layer.py 產生的)
excel_path = r"results/train2m_20251211_222442/val_predictions.xlsx"

# 設定顯示名稱 (與 Inference 一致: 0=B, 1=A+C)
CLASS_NAMES = ["B", "A+C"] 

# ==========================================
# 2. 主程式
# ==========================================
def main():
    if not os.path.exists(excel_path):
        print(f"找不到檔案: {excel_path}")
        return

    print(f"正在評估 (邏輯翻轉版): {excel_path}")
    df = pd.read_excel(excel_path)
    save_dir = os.path.dirname(excel_path)

    # --- 關鍵修正：翻轉標籤 ---
    # 原始訓練: 0=A+C (生病), 1=B (健康)
    # 目標邏輯: 0=B (健康),   1=A+C (生病) -> 符合醫學慣例
    
    # 數學上，原本 0 變 1，原本 1 變 0，剛好可以用 (1 - x) 來轉換
    y_true_raw = df['true_label'].values
    y_pred_raw = df['pred_label'].values
    
    y_true = 1 - y_true_raw
    y_pred = 1 - y_pred_raw
    
    # 分數處理：
    # 我們需要 "生病 (A+C)" 的機率作為 Positive Score
    # 原本 prob_class0 就是 A+C 的機率，所以直接用它當作 y_score
    y_score = df['prob_class0'].values 

    # --- 3. 基礎指標 ---
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred) # 預設 pos_label=1 (現在是 A+C)
    sens = recall_score(y_true, y_pred) # Sensitivity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")

    # --- 4. 繪製 Confusion Matrix (數量版 - 藍色) ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Instance Confusion Matrix (Count)")
    plt.savefig(os.path.join(save_dir, "Instance_CM_Count.png"), dpi=300)
    plt.close()

    # --- 5. 繪製 Confusion Matrix (百分比版 - 橘色) ---
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES)
    disp_norm.plot(cmap="Oranges", values_format=".2f")
    plt.title("Instance Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "Instance_CM_Percent.png"), dpi=300)
    plt.close()

    # --- 6. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Instance ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "Instance_ROC.png"), dpi=300)
    plt.close()

    # --- 7. PR Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Instance Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "Instance_PR.png"), dpi=300)
    plt.close()

    print(f"評估圖表已儲存至: {save_dir}")

if __name__ == "__main__":
    main()