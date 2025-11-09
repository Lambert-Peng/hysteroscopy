import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, matthews_corrcoef, brier_score_loss
)

# ======== 設定 ========
excel_path = "results/train2_run_20251109_172647/val_predictions.xlsx"
save_dir = os.path.dirname(excel_path)
os.makedirs(save_dir, exist_ok=True)
target_specificity = 0.9
class_names = ["A+C", "B"]  # class 0, class 1

# ======== 讀取資料並檢查 ========
df = pd.read_excel(excel_path)
if "true_label" not in df.columns:
    raise KeyError("找不到欄位 'true_label'，請確認 excel 檔案格式。")
if "pred_label" not in df.columns:
    raise KeyError("找不到欄位 'pred_label'，請確認 excel 檔案格式。")
prob_cols = [c for c in df.columns if c.startswith("prob_class")]
if len(prob_cols) < 2:
    raise KeyError("找不到兩個以上的 prob_class 欄位 (prob_class0, prob_class1)。")

# 允許 true_label 為文字或數字：若是文字，轉成 0..K-1
orig_labels = df["true_label"].unique()
if df["true_label"].dtype.kind in 'iufc':
    y_true = df["true_label"].to_numpy()
else:
    mapping = {v: i for i, v in enumerate(sorted(orig_labels))}
    print("true_label 非數值，已做 mapping：", mapping)
    y_true = df["true_label"].map(mapping).to_numpy()

y_pred = df["pred_label"].to_numpy()
probs = df[prob_cols].to_numpy()
# assume binary: prob_class0, prob_class1
y_score_class0 = probs[:, 0]
y_score_class1 = probs[:, 1]

# ======== 一般二分類整體評估（以 pred_label 為基準） ========
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)
brier = brier_score_loss(y_true, y_score_class1)  # using positive class prob

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
try:
    tn, fp, fn, tp = cm.ravel()
except ValueError:
    # 若不是 2x2（極端情況），補成 0
    tn = fp = fn = tp = 0

specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
npv = tn / (tn + fn) if (tn + fn) != 0 else 0

# AUC & AP（per-class, one-vs-rest）
fpr0, tpr0, thr0 = roc_curve((y_true == 0).astype(int), y_score_class0)
auc0 = auc(fpr0, tpr0)
ap0 = average_precision_score((y_true == 0).astype(int), y_score_class0)

fpr1, tpr1, thr1 = roc_curve((y_true == 1).astype(int), y_score_class1)
auc1 = auc(fpr1, tpr1)
ap1 = average_precision_score((y_true == 1).astype(int), y_score_class1)

metrics = {
    "Accuracy": acc,
    "Precision (PPV)": prec,
    "Recall (Sensitivity)": recall,
    "Specificity": specificity,
    "NPV": npv,
    "F1-score": f1,
    "MCC": mcc,
    "Brier Score": brier,
    "AUC_class0 (A+C)": auc0,
    "AUC_class1 (B)": auc1,
    "AP_class0 (A+C)": ap0,
    "AP_class1 (B)": ap1
}

# ======== 印出詳細指標 ========
print("\n===== 一般二分類評估 (using pred_label) =====")
for k, v in metrics.items():
    print(f"{k:25s}: {v:.4f}")

# 匯出 summary
pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_excel(
    os.path.join(save_dir, "evaluation_summary.xlsx"), index=False
)

# ======== 混淆矩陣圖 (count & percent) ========
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_count.png"), dpi=300)
plt.close()

cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(cmap="Oranges", values_format=".2f")
plt.title("Confusion Matrix (Row Normalized)")
plt.savefig(os.path.join(save_dir, "confusion_matrix_percent.png"), dpi=300)
plt.close()

# ======== ROC: 畫出兩條 (class0, class1)，並標註 AUC ========
plt.figure(figsize=(7,6))
plt.plot(fpr0, tpr0, label=f"{class_names[0]} (AUC={auc0:.3f})")
plt.plot(fpr1, tpr1, label=f"{class_names[1]} (AUC={auc1:.3f})")
plt.plot([0,1],[0,1],"k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (per-class, one-vs-rest)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ROC_per_class.png"), dpi=300)
plt.close()

# ======== PR: 畫出兩條，標註 AP ========
precision0, recall0, _ = precision_recall_curve((y_true == 0).astype(int), y_score_class0)
precision1, recall1, _ = precision_recall_curve((y_true == 1).astype(int), y_score_class1)

plt.figure(figsize=(7,6))
plt.plot(recall0, precision0, label=f"{class_names[0]} (AP={ap0:.3f})")
plt.plot(recall1, precision1, label=f"{class_names[1]} (AP={ap1:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (per-class, one-vs-rest)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "PR_per_class.png"), dpi=300)
plt.close()

# ======== 固定 specificity = 0.9: 對每個 class 做 one-vs-rest 閾值搜尋 ========
def find_threshold_for_specificity(y_binary, scores, target_spec=0.9):
    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_spec))
    thr = thresholds[idx]
    # 計算混淆矩陣與指標
    y_pred_thr = (scores >= thr).astype(int)
    cm_thr = confusion_matrix(y_binary, y_pred_thr)
    if cm_thr.size == 4:
        tn, fp, fn, tp = cm_thr.ravel()
    else:
        tn = fp = fn = tp = 0
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    prec = precision_score(y_binary, y_pred_thr, zero_division=0)
    f1s = f1_score(y_binary, y_pred_thr, zero_division=0)
    acc = accuracy_score(y_binary, y_pred_thr)
    return {
        "threshold": float(thr),
        "specificity": float(spec),
        "sensitivity": float(sens),
        "precision": float(prec),
        "f1": float(f1s),
        "accuracy": float(acc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }, (fpr, tpr)

results_specificity = {}
for cls_idx, (scores, name) in enumerate([(y_score_class0, class_names[0]), (y_score_class1, class_names[1])]):
    y_bin = (y_true == cls_idx).astype(int)
    res, (fpr_c, tpr_c) = find_threshold_for_specificity(y_bin, scores, target_specificity)
    results_specificity[name] = res
    print(f"\n=== Specificity ≈ {target_specificity} for class '{name}' ===")
    for k, v in res.items():
        if k in ("tn","fp","fn","tp"):
            print(f"  {k:4s}: {v}")
        else:
            print(f"  {k:12s}: {v:.4f}")

# 匯出 specificity 結果到 Excel
df_spec = pd.DataFrame([{"class": k, **v} for k, v in results_specificity.items()])
df_spec.to_excel(os.path.join(save_dir, f"specificity_eval_{target_specificity}.xlsx"), index=False)

print(f"\n所有檔案輸出於：{save_dir}")