import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)
import matplotlib.ticker as mtick

# ======== 設定 ========
excel_path = "results/train3_1layer_run_20251203_014900/val_predictions.xlsx"
save_dir = os.path.dirname(excel_path)
os.makedirs(save_dir, exist_ok=True)
target_specificity = 0.9

# ======== 讀取資料 ========
df = pd.read_excel(excel_path)
if "true_label" not in df.columns:
    raise KeyError("找不到欄位 'true_label'，請確認 Excel 檔案格式。")
prob_cols = [c for c in df.columns if c.startswith("prob_class")]
if len(prob_cols) == 0:
    raise KeyError("找不到任何 prob_class 欄位 (prob_class0, prob_class1, ...)")

# 處理 true_label 非數值情況
orig_labels = df["true_label"].unique()
if df["true_label"].dtype.kind in 'iufc':
    y_true = df["true_label"].to_numpy()
    class_map = {i: i for i in np.unique(y_true)}
else:
    class_map = {v: i for i, v in enumerate(sorted(orig_labels))}
    print("true_label 非數值，已做 mapping：", class_map)
    y_true = df["true_label"].map(class_map).to_numpy()

probs = df[prob_cols].to_numpy()
num_classes = probs.shape[1]
class_names = ["A", "B", "C"]  # 根據實際類別名稱調整

print(f"\n讀取 {len(df)} 筆樣本，共 {num_classes} 類別")

# ======== 整體多分類指標 (pred by argmax prob) ========
pred_labels = np.argmax(probs, axis=1)
acc = accuracy_score(y_true, pred_labels)
macro_prec = precision_score(y_true, pred_labels, average="macro", zero_division=0)
macro_rec = recall_score(y_true, pred_labels, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, pred_labels, average="macro", zero_division=0)

print("\n===== 一般多分類模型評估 =====")
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {macro_prec:.4f}")
print(f"Macro Recall    : {macro_rec:.4f}")
print(f"Macro F1-score  : {macro_f1:.4f}")

summary_metrics = {
    "Accuracy": acc,
    "Macro Precision": macro_prec,
    "Macro Recall": macro_rec,
    "Macro F1-score": macro_f1
}
pd.DataFrame(list(summary_metrics.items()), columns=["Metric", "Value"]).to_excel(
    os.path.join(save_dir, "evaluation_summary.xlsx"), index=False
)

# ======== 混淆矩陣 (Count / Percentage) ========
cm = confusion_matrix(y_true, pred_labels)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A","B","C"]).plot(
    cmap="Blues", values_format="d"
)
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_count.png"), dpi=300)
plt.close()

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["A","B","C"]).plot(
    cmap="Oranges", values_format=".2f"
)
plt.title("Confusion Matrix (Percentage)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ======== per-class ROC & PR (One-vs-Rest) ========
plt.figure(figsize=(8,6))
for i in range(num_classes):
    y_bin = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (per-class, One-vs-Rest)")
plt.legend(fontsize="small", loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_per_class.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,6))
for i in range(num_classes):
    y_bin = (y_true == i).astype(int)
    precision_vals, recall_vals, _ = precision_recall_curve(y_bin, probs[:, i])
    ap = average_precision_score(y_bin, probs[:, i])
    plt.plot(recall_vals, precision_vals, label=f"{class_names[i]} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (per-class, One-vs-Rest)")
plt.legend(fontsize="small", loc="upper right")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_per_class.png"), dpi=300)
plt.close()

# ======== 固定 specificity = 0.9: 對每類別計算 threshold 與指標 ========
results = []
for i in range(num_classes):
    y_bin = (y_true == i).astype(int)
    scores = probs[:, i]
    fpr, tpr, thresholds = roc_curve(y_bin, scores)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    thr = thresholds[idx]
    y_pred_thr = (scores >= thr).astype(int)
    cm_thr = confusion_matrix(y_bin, y_pred_thr)
    if cm_thr.size == 4:
        tn, fp, fn, tp = cm_thr.ravel()
    else:
        tn = fp = fn = tp = 0
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    prec = precision_score(y_bin, y_pred_thr, zero_division=0)
    f1s = f1_score(y_bin, y_pred_thr, zero_division=0)
    acc_c = accuracy_score(y_bin, y_pred_thr)
    ap = average_precision_score(y_bin, scores)
    results.append({
        "class_name": class_names[i],
        "threshold": thr,
        "specificity": spec,
        "sensitivity": sens,
        "precision": prec,
        "f1": f1s,
        "accuracy": acc_c,
        "AP": ap,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    })

# 匯出 Excel
pd.DataFrame(results).to_excel(
    os.path.join(save_dir, f"specificity_eval_{target_specificity}.xlsx"), index=False
)

# ======== 柱狀圖 (Sensitivity, Precision, F1, Accuracy) ========
metrics_names = ["sensitivity", "precision", "f1", "accuracy"]
plt.figure(figsize=(10,6))
bar_width = 0.2
x = np.arange(num_classes)
for idx_m, metric in enumerate(metrics_names):
    values = [r[metric] for r in results]
    bars = plt.bar(x + idx_m*bar_width, values, width=bar_width, label=metric.capitalize())
    if idx_m == 0:
        for j, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{results[j]['threshold']:.2f}", ha='center', va='bottom', fontsize=9)
plt.xticks(x + bar_width*(len(metrics_names)-1)/2, [r["class_name"] for r in results])
plt.ylim(0,1)
plt.ylabel("Score")
plt.title(f"Metrics at Specificity ≈ {target_specificity}")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.savefig(os.path.join(save_dir, f"specificity_metrics_{target_specificity}.png"), dpi=300)
plt.close()

# ======== 結果印出 ========
print(f"\n===== Specificity ≈ {target_specificity} 各類別結果 =====")
for r in results:
    print(f"\nClass {r['class_name']}:")
    print(f"  Threshold   : {r['threshold']:.3f}")
    print(f"  Specificity : {r['specificity']:.3f}")
    print(f"  Sensitivity : {r['sensitivity']:.3f}")
    print(f"  Precision   : {r['precision']:.3f}")
    print(f"  F1-score    : {r['f1']:.3f}")
    print(f"  Accuracy    : {r['accuracy']:.3f}")
    print(f"  AP          : {r['AP']:.3f}")
    print(f"  Confusion Matrix (tn, fp, fn, tp): ({r['tn']}, {r['fp']}, {r['fn']}, {r['tp']})")

print(f"\n所有圖表與結果已輸出至：{save_dir}")
