import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import os

# ======== 設定 ========
# 修改成你 train3_class 跑出來的 Excel 路徑
excel_path = r"results/best/train3_mean_20251209_235525/val_predictions_3class.xlsx"
save_dir = os.path.dirname(excel_path)

# 定義類別名稱
CLASS_NAMES = ["A", "B", "C"]
n_classes = 3

# ======== 讀取資料 ========
print(f"正在評估: {excel_path}")
df = pd.read_excel(excel_path)

y_true = df['true_label'].values
y_pred = df['pred_label'].values
# 讀取 3 個機率欄位
y_score = df[['prob_A', 'prob_B', 'prob_C']].values 

# ======== 基礎指標 ========
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro') # 多分類使用 Macro F1
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")

# ======== 1. Confusion Matrix ========
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_count.png"), dpi=300)
plt.close()

cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(cmap="Oranges", values_format=".2f")
plt.title("Instance Confusion Matrix")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ======== 準備 One-vs-Rest 資料 ========
# 將標籤轉為 One-hot (例如 A=[1,0,0])
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

# ======== 2. ROC Curve (One-vs-Rest) ========
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_3class.png"), dpi=300)
plt.close()

# ======== 3. PR Curve (One-vs-Rest) ========
plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, color=color, lw=2,
             label=f'Class {CLASS_NAMES[i]} (AP = {avg_precision:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-Rest)')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_3class.png"), dpi=300)
plt.close()

print(f"評估圖表已儲存至: {save_dir}")