import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_directml
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


BINARY_MODEL_PATH = "1210/results/best/train2_1layer_mean_20251210_010556/best_model_binary.pt"  # A+C vs B
AC_MODEL_PATH     = "results/best_model_AC.pt"                                              # A vs C (3-class head but ignore B)
VAL_DATA_DIR      = "dataset/val/weights"

TOP_K = 3

# threshold sweep for Stage-1 gate (bag abnormal score vs T)
THR_START = 0.00
THR_END   = 1.00
THR_STEP  = 0.01

SPEC_TARGET = 0.90        # 之前你是用 specificity=0.9 來找 threshold
SELECT_BY = "accuracy"  # options: "macro_f1", "abnormal_recall", "accuracy"

# =============================================================================

CLASS_NAMES = ["A", "B", "C"]
IDX_A, IDX_B, IDX_C = 0, 1, 2

device = torch_directml.device()


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_bag_id(filename: str) -> str:
    parts = str(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def abnormal_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # positive = abnormal (A or C), negative = B
    y_true_bin = (y_true != IDX_B).astype(int)
    y_pred_bin = (y_pred != IDX_B).astype(int)
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    return float(TP / (TP + FN + 1e-8))

def abnormal_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # negative = B
    y_true_bin = (y_true != IDX_B).astype(int)
    y_pred_bin = (y_pred != IDX_B).astype(int)
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    return float(TN / (TN + FP + 1e-8))

def per_class_table(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Return:
      - df_metrics: per-class metrics table (like your screenshot)
      - cm: 3x3 confusion matrix
    Columns:
      Class, Precision, Recall, F1, Spec, NPV, MCC, TP, FP, FN, TN
    """
    labels = [0, 1, 2]
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1v  = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for i, cname in enumerate(CLASS_NAMES):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(cm.sum() - (TP + FN + FP))

        spec = TN / (TN + FP + 1e-8)
        npv  = TN / (TN + FN + 1e-8)

        # Matthews Correlation Coefficient (one-vs-rest for this class)
        denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        mcc = 0.0 if denom <= 0 else (TP * TN - FP * FN) / np.sqrt(denom)

        rows.append({
            "Class": cname,
            "Precision": float(prec[i]),
            "Recall": float(rec[i]),
            "F1": float(f1v[i]),
            "Spec": float(spec),
            "NPV": float(npv),
            "MCC": float(mcc),
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        })

    df = pd.DataFrame(rows, columns=[
        "Class","Precision","Recall","F1","Spec","NPV","MCC","TP","FP","FN","TN"
    ])
    return df, cm

def save_cm_plots(cm: np.ndarray, save_dir: str, title_prefix: str, save_prefix: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    # counts
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"{title_prefix} (Counts)")
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_counts.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # row-normalized
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    cmn = cm.astype(float) / row
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp2.plot(ax=ax, cmap="Oranges", values_format=".2f")
    plt.title(f"{title_prefix} (Percent)")
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_percent.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_metrics_table_figure(df_metrics: pd.DataFrame, save_path: str, title: str):
    """
    Create a figure like your screenshot:
      Big title + table (Class + metrics columns)
    """
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")

    # Title
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)

    # Prepare display values (4 decimals for floats, ints for counts)
    df_disp = df_metrics.copy()
    float_cols = ["Precision","Recall","F1","Spec","NPV","MCC"]
    for c in float_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:.4f}")
    for c in ["TP","FP","FN","TN"]:
        df_disp[c] = df_disp[c].astype(int).astype(str)

    col_labels = list(df_disp.columns)
    cell_text  = df_disp.values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    # Style: thicker borders + readable size
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)

    # thicker grid lines
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1.8)
        if row == 0:
            cell.set_fontsize(15)
            cell.set_text_props(fontweight="bold")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sweep(sweep_df: pd.DataFrame, save_dir: str, title: str, save_name: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep_df["threshold"], sweep_df["macro_f1"], label="macro_f1")
    ax.plot(sweep_df["threshold"], sweep_df["accuracy"], label="accuracy")
    ax.plot(sweep_df["threshold"], sweep_df["abnormal_recall"], label="abnormal_recall")
    ax.plot(sweep_df["threshold"], sweep_df["abnormal_specificity"], label="abnormal_specificity")
    ax.axhline(SPEC_TARGET, linestyle="--", linewidth=1, label=f"spec_target={SPEC_TARGET}")
    ax.set_xlabel("T_abn (bag threshold for A+C vs B)")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches="tight")
    plt.close()

def pick_best_threshold_with_spec(sweep_df: pd.DataFrame):
    cand = sweep_df[sweep_df["abnormal_specificity"] >= SPEC_TARGET].copy()

    if len(cand) == 0:
        tmp = sweep_df.copy()
        tmp["spec_gap"] = np.abs(tmp["abnormal_specificity"] - SPEC_TARGET)
        best = tmp.sort_values(["spec_gap", "threshold"]).iloc[0].to_dict()
        return float(best["threshold"]), best, False

    if SELECT_BY == "macro_f1":
        key = "macro_f1"
    elif SELECT_BY == "abnormal_recall":
        key = "abnormal_recall"
    elif SELECT_BY == "accuracy":
        key = "accuracy"
    else:
        raise ValueError("SELECT_BY must be macro_f1 / abnormal_recall / accuracy")

    best = cand.loc[cand[key].idxmax()].to_dict()
    return float(best["threshold"]), best, True


# =============================================================================
# Dataset / Model
# =============================================================================
class RecursiveFeatureDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        self.labels = [self._get_label(f) for f in self.files]

    def _get_label(self, fpath: Path) -> int:
        filename = fpath.name
        if filename.startswith("A"): return 0
        if filename.startswith("B"): return 1
        if filename.startswith("C"): return 2
        return 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fpath = self.files[idx]
        feature = torch.load(fpath).float()

        # keep same pooling behavior as your scripts
        if feature.dim() == 3:
            feature = feature[0].mean(dim=0)
        elif feature.dim() == 2:
            feature = feature.mean(dim=0)
        elif feature.dim() == 1 and feature.shape[0] > 768:
            feature = feature.view(-1, 768).mean(dim=0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label, fpath.name

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def load_model(path: str, num_classes: int):
    print(f"[INFO] Loading: {path}")
    model = MLPClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =============================================================================
# (Part 1) Inference export: Binary gate + AC(A/C only) => composed tri probs
# =============================================================================
def run_inference_export(save_dir: str) -> str:
    model_bin = load_model(BINARY_MODEL_PATH, num_classes=2)  # output: [Abnormal(A+C), B]
    model_ac  = load_model(AC_MODEL_PATH,     num_classes=3)  # output: [A, B, C] but B is ignored in training

    dataset = RecursiveFeatureDataset(VAL_DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []
    with torch.no_grad():
        for features, label, filenames in loader:
            features = features.to(device)

            # --- (1) Binary model -> p_abn ---
            logits_bin = model_bin(features)
            probs_bin = F.softmax(logits_bin, dim=1).detach().cpu().numpy().astype(np.float32)

            # keep consistent with your inf_M2.py: probs_bin[:,0] is P(Abnormal)
            p_abn = probs_bin[:, 0]
            # p_B = probs_bin[:, 1]  # not used, equals 1-p_abn if model is calibrated

            # --- (2) AC model -> use ONLY A/C logits, ignore B ---
            logits_ac = model_ac(features).detach().cpu().numpy().astype(np.float32)
            logit_A = logits_ac[:, IDX_A]
            logit_C = logits_ac[:, IDX_C]

            exp_A = np.exp(logit_A)
            exp_C = np.exp(logit_C)
            denom = exp_A + exp_C + 1e-8
            p_A_cond = (exp_A / denom).astype(np.float32)  # P(A | A/C)
            p_C_cond = (exp_C / denom).astype(np.float32)  # P(C | A/C)

            # --- (3) Compose tri-class soft probs ---
            prob_A = (p_abn * p_A_cond).astype(np.float32)
            prob_C = (p_abn * p_C_cond).astype(np.float32)
            prob_B = (1.0 - p_abn).astype(np.float32)

            for i in range(len(filenames)):
                tri = np.array([prob_A[i], prob_B[i], prob_C[i]], dtype=np.float32)
                results.append({
                    "filename": filenames[i],
                    "true_label": int(label[i].item()),

                    # soft outputs for MIL
                    "p_abn": float(p_abn[i]),
                    "p_A_cond": float(p_A_cond[i]),
                    "p_C_cond": float(p_C_cond[i]),
                    "prob_A": float(prob_A[i]),
                    "prob_B": float(prob_B[i]),
                    "prob_C": float(prob_C[i]),

                    # optional instance pred (not used in bag-level eval)
                    "pred_tri": int(np.argmax(tri))
                })

    out_xlsx = os.path.join(save_dir, "combined_predictions.xlsx")
    pd.DataFrame(results).to_excel(out_xlsx, index=False)
    print(f"[INFO] Saved: {out_xlsx}")
    return out_xlsx


# =============================================================================
# (Part 2) MIL: stage-1 threshold on pooled p_abn, stage-2 A vs C
# =============================================================================
def mil_predict_one_bag(df_bag: pd.DataFrame, pooling: str, thr: float):
    p_abn   = df_bag["p_abn"].to_numpy(float)
    pA_cond = df_bag["p_A_cond"].to_numpy(float)
    pC_cond = df_bag["p_C_cond"].to_numpy(float)

    n = len(p_abn)
    if n == 0:
        return IDX_B, 0.0

    if pooling == "max":
        idx = int(np.argmax(p_abn))
        I = [idx]
        S = float(np.max(p_abn))
    elif pooling == "mean":
        I = list(range(n))
        S = float(np.mean(p_abn))
    elif pooling == "top3":
        order = np.argsort(-p_abn)
        k = min(TOP_K, n)
        I = order[:k].tolist()
        S = float(np.mean(p_abn[I])) if k > 0 else 0.0
    else:
        raise ValueError("pooling must be max/mean/top3")

    # Stage-1: A+C vs B (bag abnormal score vs threshold)
    if S < thr:
        return IDX_B, S

    # Stage-2: A vs C (weighted by p_abn on selected instances)
    A_score = float(np.sum(p_abn[I] * pA_cond[I]))
    C_score = float(np.sum(p_abn[I] * pC_cond[I]))
    pred = IDX_A if A_score >= C_score else IDX_C
    return pred, S

def evaluate(df: pd.DataFrame, bag_true: pd.Series, pooling: str, thr: float):
    y_true, y_pred = [], []
    for bag_id, df_bag in df.groupby("bag_id"):
        pred, _ = mil_predict_one_bag(df_bag, pooling, thr)
        y_pred.append(pred)
        y_true.append(int(bag_true.loc[bag_id]))

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1v = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
    abn_rec = abnormal_recall(y_true, y_pred)
    abn_spec = abnormal_specificity(y_true, y_pred)
    return y_true, y_pred, acc, macro_f1v, abn_rec, abn_spec


# =============================================================================
# Macro-average ROC overlay (same style as inference3.py)
# =============================================================================
def macro_average_roc_curve_inference3(y_true_3class: np.ndarray, y_score_3class: np.ndarray, n_classes: int = 3):
    y_true_bin = label_binarize(y_true_3class, classes=list(range(n_classes)))

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true_bin[:, i], y_score_3class[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_3class[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, float(macro_auc)

def bag_triclass_scores(df_bag: pd.DataFrame, pooling: str):
    """
    For macro-average ROC, we need bag-level 3-class scores.
    We pool composed probs (prob_A/prob_B/prob_C).
    For top3 selection, we select instances by highest p_abn (most suspicious).
    """
    p_abn = df_bag["p_abn"].to_numpy(float)
    pA = df_bag["prob_A"].to_numpy(float)
    pB = df_bag["prob_B"].to_numpy(float)
    pC = df_bag["prob_C"].to_numpy(float)

    n = len(p_abn)
    if n == 0:
        return 0.0, 0.0, 0.0

    if pooling == "max":
        idx = int(np.argmax(p_abn))
        I = [idx]
    elif pooling == "mean":
        I = list(range(n))
    elif pooling == "top3":
        order = np.argsort(-p_abn)
        k = min(TOP_K, n)
        I = order[:k].tolist()
    else:
        raise ValueError("pooling must be max/mean/top3")

    sA = float(np.mean(pA[I])) if len(I) else 0.0
    sB = float(np.mean(pB[I])) if len(I) else 0.0
    sC = float(np.mean(pC[I])) if len(I) else 0.0
    return sA, sB, sC

def plot_macro_average_roc_overlay(df: pd.DataFrame, save_dir: str):
    bag_true = df.groupby("bag_id")["true_label"].agg(lambda x: int(x.value_counts().idxmax()))

    fig, ax = plt.subplots(figsize=(8, 6))

    for pooling in ["max", "mean", "top3"]:
        y_true = []
        y_score = []
        for bag_id, df_bag in df.groupby("bag_id"):
            y_true.append(int(bag_true.loc[bag_id]))
            sA, sB, sC = bag_triclass_scores(df_bag, pooling)
            y_score.append([sA, sB, sC])

        y_true = np.array(y_true, dtype=int)
        y_score = np.array(y_score, dtype=float)

        if len(np.unique(y_true)) < 2:
            ax.text(0.05, 0.05, f"ROC undefined (only one class) for {pooling}", transform=ax.transAxes)
            continue

        fpr_m, tpr_m, auc_m = macro_average_roc_curve_inference3(y_true, y_score, n_classes=3)
        ax.plot(fpr_m, tpr_m, lw=2, label=f"{pooling} (Macro AUC={auc_m:.3f})")

    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("3-class Macro-Average ROC Overlay (max / mean / top3)")
    ax.grid(True)
    ax.legend(loc="lower right")

    out_path = os.path.join(save_dir, "M2_ROC.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC: {out_path}")


# =============================================================================
# MIL sweep + reports
# =============================================================================
def run_mil_and_reports(infer_xlsx: str, save_dir: str):
    df = pd.read_excel(infer_xlsx)
    required = {"filename", "true_label", "p_abn", "p_A_cond", "p_C_cond", "prob_A", "prob_B", "prob_C"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in inference Excel: {missing}")

    df["bag_id"] = df["filename"].apply(extract_bag_id)

    # bag true label: majority vote (robust to instance noise)
    bag_true = df.groupby("bag_id")["true_label"].agg(lambda x: int(x.value_counts().idxmax()))

    thresholds = np.round(np.arange(THR_START, THR_END + 1e-12, THR_STEP), 6)

    out_xlsx = os.path.join(save_dir, "MIL_sweep_spec.xlsx")
    writer = pd.ExcelWriter(out_xlsx, engine="openpyxl")

    summary_rows = []
    for pooling in ["max", "mean", "top3"]:
        sweep_rows = []
        for thr in thresholds:
            y_true, y_pred, acc, macro_f1v, abn_rec, abn_spec = evaluate(df, bag_true, pooling, float(thr))
            sweep_rows.append({
                "threshold": float(thr),
                "accuracy": acc,
                "macro_f1": macro_f1v,
                "abnormal_recall": abn_rec,
                "abnormal_specificity": abn_spec
            })

        sweep_df = pd.DataFrame(sweep_rows)
        best_thr, _, meets = pick_best_threshold_with_spec(sweep_df)

        sweep_df.to_excel(writer, sheet_name=f"sweep_{pooling}"[:31], index=False)
        plot_sweep(
            sweep_df, save_dir,
            title=f"{pooling} (spec>={SPEC_TARGET}, select_by={SELECT_BY})",
            save_name=f"Sweep_{pooling}.png"
        )

        # evaluate at best threshold
        y_true, y_pred, acc, macro_f1v, abn_rec, abn_spec = evaluate(df, bag_true, pooling, best_thr)
        df_metrics, cm = per_class_table(y_true, y_pred)
        df_metrics.to_excel(writer, sheet_name=f"best_{pooling}"[:31], index=False)

        save_cm_plots(cm, save_dir, f"{pooling} (best T={best_thr:.2f})", f"CM_{pooling}")

        # === NEW: save metrics table figure (like your screenshot) ===
        title = f"{pooling.capitalize()} Metrics"
        fig_path = os.path.join(save_dir, f"MIL_Metrics_Viz_{pooling.capitalize()}.png")
        save_metrics_table_figure(df_metrics, fig_path, title)

        summary_rows.append({
            "pooling": pooling,
            "SPEC_TARGET": SPEC_TARGET,
            "meets_specificity": bool(meets),
            "select_by": SELECT_BY,
            "best_T_abn": float(best_thr),
            "best_accuracy": float(acc),
            "best_macro_f1": float(macro_f1v),
            "best_abnormal_recall": float(abn_rec),
            "best_abnormal_specificity": float(abn_spec),
        })

    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
    writer.close()
    print(f"[INFO] Saved MIL report Excel: {out_xlsx}")

    # only keep macro-average ROC overlay
    plot_macro_average_roc_overlay(df, save_dir)


# =============================================================================
# MAIN
# =============================================================================
def main():
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname("results/best_model_AC.pt"), f"M2_{run_tag}")
    ensure_dir(save_dir)

    print(f"[INFO] SAVE_DIR = {save_dir}")

    print("=" * 80)
    print("[STEP 1] Export combined instance predictions (Binary gate + AC(A/C only) => 3-class probs)")
    infer_xlsx = run_inference_export(save_dir)

    print("=" * 80)
    print("[STEP 2] MIL (max/mean/top3) + threshold sweep (spec>=0.9) + macro-average ROC overlay")
    run_mil_and_reports(infer_xlsx, save_dir)

    print("=" * 80)
    print("[DONE]")
    print("Outputs:")
    print(" - combined_predictions_soft.xlsx")
    print(" - MIL_binaryAC_sweep_spec.xlsx")
    print(" - ROC_macro_average_overlay_max_mean_top3.png")
    print(f"Saved in: {save_dir}")

if __name__ == "__main__":
    main()
