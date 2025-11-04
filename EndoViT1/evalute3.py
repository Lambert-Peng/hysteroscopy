import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, matthews_corrcoef,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize

# ======== 設定路徑 ========
excel_path = "results/train3_run_20251104_171501/val_predictions.xlsx"  # 改成實際路徑
save_dir = os.path.dirname(excel_path)
df = pd.read_excel(excel_path)

# ======== 讀取資料 ========
y_true = df["true_label"].to_numpy()
y_pred = df["pred_label"].to_numpy()

# 自動偵測類別
classes = sorted(df["true_label"].unique())
num_classes = len(classes)
print(f"Detected classes: {classes}")

# 取出機率
probs_cols = [f"prob_class{i}" for i in range(num_classes)]
y_score = df[probs_cols].to_numpy()

# ======== 混淆矩陣 ========
cm = confusion_matrix(y_true, y_pred, labels=classes)
ConfusionMatrixDisplay(cm, display_labels=[f"Class {c}" for c in classes]).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_eval.png"), dpi=300)
plt.close()

cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(cm_normalized, display_labels=[f"Class {c}" for c in classes]).plot(cmap="Oranges", values_format=".2f")
plt.title("Confusion Matrix (Percentage)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()
# ======== 各種指標 ========
metrics = {}

# Accuracy
metrics["Accuracy"] = np.mean(y_pred == y_true)

# Precision / Recall / F1
metrics["Precision (Macro)"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
metrics["Recall (Macro)"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
metrics["F1-score (Macro)"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

metrics["Precision (Micro)"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
metrics["Recall (Micro)"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
metrics["F1-score (Micro)"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

# MCC
metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

# Balanced Accuracy
recall_per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
# Specificity per class
specificity_per_class = []
for i, c in enumerate(classes):
    tp = ((y_pred == c) & (y_true == c)).sum()
    fn = ((y_pred != c) & (y_true == c)).sum()
    tn = ((y_pred != c) & (y_true != c)).sum()
    fp = ((y_pred == c) & (y_true != c)).sum()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    specificity_per_class.append(specificity)
metrics["Balanced Accuracy (Macro)"] = np.mean(recall_per_class + np.array(specificity_per_class)) / 2

# Youden Index
youden_per_class = recall_per_class + np.array(specificity_per_class) - 1
metrics["Youden Index (Macro)"] = np.mean(youden_per_class)

# Brier Score per class
brier_per_class = []
y_true_binarized = label_binarize(y_true, classes=classes)
for i in range(num_classes):
    brier_per_class.append(brier_score_loss(y_true_binarized[:, i], y_score[:, i]))
metrics["Brier Score (Mean)"] = np.mean(brier_per_class)

# ROC & PR curves (One-vs-Rest)
plt.figure(figsize=(8,6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_curve_eval.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,6))
for i, c in enumerate(classes):
    p, r, _ = precision_recall_curve(y_true_binarized[:, i], y_score[:, i])
    ap = average_precision_score(y_true_binarized[:, i], y_score[:, i])
    plt.plot(r, p, label=f"Class {c} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_curve_eval.png"), dpi=300)
plt.close()

# ======== 匯出結果到 Excel ========
summary_path = os.path.join(save_dir, "evaluation_summary.xlsx")
pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_excel(summary_path, index=False)

print("\n===== 模型評估結果 =====")
for k, v in metrics.items():
    print(f"{k:30s}: {v:.4f}")
print(f"\n✅ 評估結果已儲存至：{summary_path}")