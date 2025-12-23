import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================== 設定 ==================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"results/MIL_3class_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = "results/bestweights/train3_1layer_run_20251203_014900.pt"
VAL_DIR = "dataset/val/weights"   # 底下需有 A, B, C 三類資料夾

IN_DIM = 768
TOP_K = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["A", "B", "C"]       # <-- 三分類
NUM_CLASSES = len(CLASS_NAMES)

# ================== 模型 ==================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

# ================== Bag loader ==================
def load_bag_features(case_path):
    pt_files = sorted([f for f in os.listdir(case_path) if f.endswith(".pt")])
    feats = []
    for f in pt_files:
        t = torch.load(os.path.join(case_path, f), map_location=DEVICE)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        feats.append(t.float())
    if not feats:
        return None
    return torch.cat(feats, dim=0)

# ================== pooling functions ==================
def calc_scores(prob_matrix, k):
    """
    prob_matrix shape: [N_instances, num_classes]
    回傳每種 pooling 的 N x num_classes scores
    """
    s_max  = prob_matrix.max(dim=0).values.cpu().numpy()          # (C,)
    s_mean = prob_matrix.mean(dim=0).cpu().numpy()                # (C,)
    s_topk = prob_matrix.topk(min(len(prob_matrix), k), dim=0).values.mean(dim=0).cpu().numpy()   # (C,)
    return s_max, s_topk, s_mean

# ================== confusion matrix plotting ==================
def plot_cm(y_true, pred, title, save_name):
    labels = CLASS_NAMES
    cm = confusion_matrix(y_true, pred, labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_name))
    plt.close()

# ================== 主程式 ==================
def main():
    model = MLPClassifier(in_dim=IN_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    cases = []
    y_true = []

    pred_max_list  = []
    pred_topk_list = []
    pred_mean_list = []

    smax_all  = []
    stopk_all = []
    smean_all = []

    with torch.no_grad():
        for idx, cls_name in enumerate(CLASS_NAMES):
            cls_path = os.path.join(VAL_DIR, cls_name)
            if not os.path.exists(cls_path): continue

            label = idx    # A=0, B=1, C=2

            for bag in sorted(os.listdir(cls_path)):
                bag_path = os.path.join(cls_path, bag)
                if not os.path.isdir(bag_path): continue

                feats = load_bag_features(bag_path)
                if feats is None: continue

                logits = model(feats)
                probs = torch.softmax(logits, dim=1)   # (N,3)

                # ---------- pooling (三類) ----------
                s_max, s_topk, s_mean = calc_scores(probs, TOP_K)

                # ---------- prediction ----------
                p_max  = int(np.argmax(s_max))
                p_topk = int(np.argmax(s_topk))
                p_mean = int(np.argmax(s_mean))

                cases.append(bag)
                y_true.append(label)

                pred_max_list.append(p_max)
                pred_topk_list.append(p_topk)
                pred_mean_list.append(p_mean)

                smax_all.append(s_max)
                stopk_all.append(s_topk)
                smean_all.append(s_mean)

    # ========== 儲存結果 ==========
    df = pd.DataFrame({
        "case": cases,
        "true": y_true,
        "score_max": smax_all,
        "pred_max": pred_max_list,
        "score_topk": stopk_all,
        "pred_topk": pred_topk_list,
        "score_mean": smean_all,
        "pred_mean": pred_mean_list
    })

    df.to_excel(os.path.join(SAVE_DIR, "MIL_3class_results.xlsx"), index=False)

    # ========== 混淆矩陣 ==========
    plot_cm(df["true"], df["pred_max"],  "Max Pooling",  "cm_max.png")
    plot_cm(df["true"], df["pred_topk"], f"Top-{TOP_K} Pooling", "cm_topk.png")
    plot_cm(df["true"], df["pred_mean"], "Mean Pooling", "cm_mean.png")

    print("三分類 MIL 評估完成！")

if __name__ == "__main__":
    main()