import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score, 
    ConfusionMatrixDisplay
)
import os

# ==========================================
# 1. 設定區域
# ==========================================
# 請修改 Excel 路徑 (由 train_AC_specific.py 產生的)
EXCEL_PATH = r"results/train_AC_specific_20251223_170912/val_predictions_AC.xlsx"

# 類別對應 (Excel 中: 0=A, 1=B, 2=C)
# 但 AC 模型只看 A 和 C
IGNORE_CLASS_INDEX = 1  # B

# ==========================================
# 2. 主程式
# ==========================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到檔案 {EXCEL_PATH}")
        return

    print(f"正在讀取 Excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    save_dir = os.path.dirname(EXCEL_PATH)

    # --- 1. 資料過濾 (關鍵步驟) ---
    # 雖然 Excel 裡有 B (true_label=1)，但我們評估 AC 模型時要把它拿掉
    print(f"原始資料筆數: {len(df)}")
    
    # 過濾掉 B
    df_ac = df[df['true_label'] != IGNORE_CLASS_INDEX].copy()
    print(f"過濾後 (只含 A/C) 筆數: {len(df_ac)}")

    if len(df_ac) == 0:
        print("錯誤：過濾後沒有任何 A 或 C 的資料！請檢查 Excel 內容。")
        return

    # --- 2. 標籤與機率處理 ---
    # 原始 Label: A=0, C=2
    # 為了畫二分類圖表，我們將其映射為: A=0, C=1
    
    # y_true: 0 if A, 1 if C
    y_true = df_ac['true_label'].apply(lambda x: 0 if x == 0 else 1).values
    
    # y_score: 取 C 的機率 (prob_C) 作為正類別分數
    # (因為我們定義 C 為 1)
    y_scores = df_ac['prob_C'].values
    
    # y_pred: 根據機率大小決定 (如果 prob_C > prob_A 則判為 C，否則 A)
    # 也可以簡單寫成: 1 if prob_C > 0.5 else 0 (因為 prob_A + prob_C = 1)
    y_pred = (y_scores > 0.5).astype(int)

    # --- 3. 計算指標 ---
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # 計算 AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print("\n" + "="*40)
    print(f"AC 模型評估結果 (A vs C)")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {roc_auc:.4f}")
    
    # --- 4. 繪圖：Confusion Matrix ---
    class_names = ["Suspected (A)", "Cancer (C)"]
    
    # 4.1 數量版 (Count)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix (A vs C) - Counts")
    plt.savefig(os.path.join(save_dir, "CM_AC_Counts.png"), dpi=300)
    plt.close()

    # 4.2 百分比版 (Percentage)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype("float") / row_sums
    
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp_norm.plot(cmap="Oranges", values_format=".2f")
    plt.title("Confusion Matrix (A vs C) - Percentage")
    plt.savefig(os.path.join(save_dir, "CM_AC_Percentage.png"), dpi=300)
    plt.close()

    # --- 5. 繪圖：ROC Curve ---
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (C as Positive)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (A vs C)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "ROC_AC.png"), dpi=300)
    plt.close()

    print(f"\n圖表已儲存至: {save_dir}")
    print(" - CM_AC_Counts.png")
    print(" - CM_AC_Percentage.png")
    print(" - ROC_AC.png")

if __name__ == "__main__":
    main()