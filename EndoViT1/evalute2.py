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

# ======== 設定路徑 ========
excel_path = "results/train2_run_20251104_163642/val_predictions.xlsx"
save_dir = os.path.dirname(excel_path)
df = pd.read_excel(excel_path)

# ======== 取出資料 ========
y_true = df["true_label"].to_numpy()
y_pred = df["pred_label"].to_numpy()
y_score = df["prob_class1"].to_numpy()

# ======== 混淆矩陣 ========
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# ======== 各種評估指標 ========
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
npv = tn / (tn + fn) if (tn + fn) != 0 else 0
f1 = f1_score(y_true, y_pred, zero_division=0)
balanced_acc = (recall + specificity) / 2
youden = recall + specificity - 1
mcc = matthews_corrcoef(y_true, y_pred)
roc_auc = auc(*roc_curve(y_true, y_score)[:2][::-1])
ap = average_precision_score(y_true, y_score)
brier = brier_score_loss(y_true, y_score)

# Likelihood Ratios & Diagnostic Odds Ratio
lr_pos = recall / (1 - specificity) if (1 - specificity) != 0 else np.inf
lr_neg = (1 - recall) / specificity if specificity != 0 else np.inf
dor = lr_pos / lr_neg if lr_neg != 0 else np.inf

# ======== 結果整理 ========
metrics = {
    "Accuracy": accuracy,
    "Precision (PPV)": precision,
    "Recall (Sensitivity)": recall,
    "Specificity": specificity,
    "NPV": npv,
    "F1-score": f1,
    "Balanced Accuracy": balanced_acc,
    "Youden Index": youden,
    "MCC": mcc,
    "AUC": roc_auc,
    "Average Precision (mAP@50)": ap,
    "Brier Score": brier,
    "LR+": lr_pos,
    "LR−": lr_neg,
    "Diagnostic Odds Ratio": dor
}

# ======== 印出結果 ========
print("\n===== 模型評估結果 =====")
for k, v in metrics.items():
    print(f"{k:25s}: {v:.4f}")

# ======== 匯出結果到 Excel ========
summary_path = os.path.join(save_dir, "evaluation_summary.xlsx")
pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_excel(summary_path, index=False)
print(f"\n評估結果已儲存到：{summary_path}")

# ======== 圖表繪製 ========
labels = ["A+C", "B"]

# 混淆矩陣
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_eval.png"), dpi=300)
plt.close()

cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(cm_normalized, display_labels=labels).plot(cmap="Oranges", values_format=".2f")
plt.title("Confusion Matrix (Percentage)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc_val = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_curve_eval.png"), dpi=300)
plt.close()

# PR Curve
p, r, _ = precision_recall_curve(y_true, y_score)
plt.plot(r, p, label=f"AP={ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_curve_eval.png"), dpi=300)
plt.close()

print(f"\n所有評估圖表與指標已輸出至：{save_dir}")