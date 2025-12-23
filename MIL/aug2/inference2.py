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
SAVE_DIR = f"results/MIL_2class_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\results\bestweights\train2_1layer_run_20251203_015602.pt"
VAL_DIR = "dataset/val/weights"
IN_DIM = 768
TOP_K = 2
THRESHOLD = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 模型 ==================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=2):
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
def calc_scores(p, k):
    s_max  = p.max().item()
    s_mean = p.mean().item()
    s_topk = p.topk(min(len(p), k)).values.mean().item()
    return s_max, s_topk, s_mean

# ================== confusion matrix plotting ==================
def plot_cm(y_true, pred, title, save_name):
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    labels = ["Disease(0)", "Healthy(1)"]
    plt.figure(figsize=(5,4))
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
    model = MLPClassifier(in_dim=IN_DIM, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    cases = []
    y_true = []
    p_max_list, p_topk_list, p_mean_list = [], [], []
    s_max_list, s_topk_list, s_mean_list = [], [], []

    with torch.no_grad():
        for cls_name in ["A", "B"]:
            cls_path = os.path.join(VAL_DIR, cls_name)
            if not os.path.exists(cls_path): continue

            label = 0 if cls_name == "A" else 1

            for bag in sorted(os.listdir(cls_path)):
                bag_path = os.path.join(cls_path, bag)
                if not os.path.isdir(bag_path): continue

                feats = load_bag_features(bag_path)
                if feats is None: continue

                logits = model(feats)
                probs = torch.softmax(logits, dim=1)[:, 0]   # prob of Disease (class 0)

                s_max, s_topk, s_mean = calc_scores(probs, TOP_K)
                p_max  = 0 if s_max  > THRESHOLD else 1
                p_topk = 0 if s_topk > THRESHOLD else 1
                p_mean = 0 if s_mean > THRESHOLD else 1

                cases.append(bag)
                y_true.append(label)
                p_max_list.append(p_max)
                p_topk_list.append(p_topk)
                p_mean_list.append(p_mean)
                s_max_list.append(s_max)
                s_topk_list.append(s_topk)
                s_mean_list.append(s_mean)

    df = pd.DataFrame({
        "case": cases,
        "true": y_true,
        "score_max": s_max_list,
        "pred_max": p_max_list,
        "score_topk": s_topk_list,
        "pred_topk": p_topk_list,
        "score_mean": s_mean_list,
        "pred_mean": p_mean_list
    })

    df.to_excel(os.path.join(SAVE_DIR, "MIL_2class_results.xlsx"), index=False)

    plot_cm(df["true"], df["pred_max"],  "Max Pooling", "cm_max.png")
    plot_cm(df["true"], df["pred_topk"], f"Top-{TOP_K} Pooling", "cm_topk.png")
    plot_cm(df["true"], df["pred_mean"], "Mean Pooling", "cm_mean.png")

    print("2-分類 MIL 評估完成！")

if __name__ == "__main__":
    main()