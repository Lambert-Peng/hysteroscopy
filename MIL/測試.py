import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# ================= 設定區 =================

MODEL_CONFIG = {
    # 請修改這裡: 2 或 3
    "type": 2,  
    "path": "results/bestweights/train2_1layer_run_20251201_012915.pt", 
    "val_dir": "dataset/val/weights",        
    "in_dim": 768
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 模型與工具 =================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.linear(x)

def load_bag_features(case_path):
    pt_files = sorted([f for f in os.listdir(case_path) if f.endswith(".pt")])
    if not pt_files: return None
    feats = []
    for f in pt_files:
        t = torch.load(os.path.join(case_path, f), map_location=DEVICE)
        if t.dim() == 1: t = t.unsqueeze(0)
        feats.append(t.float())
    return torch.cat(feats, dim=0)

def get_bag_score(logits, model_type):
    """
    計算 Bag 的平均分數 (Mean Pooling)
    因為 Max/Top-K 已經證明會爆掉，我們專注於 Mean
    """
    probs = torch.softmax(logits, dim=1)
    
    # 取得 "病灶機率"
    if model_type == 2:
        p_disease = probs[:, 0] # Class 0 is Disease
    else:
        p_disease = 1.0 - probs[:, 1] # 1-B is Disease
        
    # === 關鍵: 使用 Mean Pooling ===
    # 這能反映整個 Bag 的平均傾向
    return torch.mean(p_disease).item()

# ================= 主診斷邏輯 =================
def run_diagnosis():
    print(f"--- 開始診斷分佈 (Model Type: {MODEL_CONFIG['type']}) ---")
    model = MLPClassifier(in_dim=MODEL_CONFIG['in_dim'], num_classes=MODEL_CONFIG['type']).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CONFIG['path'], map_location=DEVICE))
    model.eval()

    bag_scores = []
    true_labels = [] # 0=Sick, 1=Healthy
    case_names = []

    root = MODEL_CONFIG['val_dir']
    classes = ['A', 'B', 'C']
    
    with torch.no_grad():
        for cls_name in classes:
            cls_path = os.path.join(root, cls_name)
            if not os.path.exists(cls_path): continue
            
            # 0=病, 1=健
            label = 1 if cls_name == 'B' else 0
            
            for case_name in sorted(os.listdir(cls_path)):
                case_path = os.path.join(cls_path, case_name)
                if not os.path.isdir(case_path): continue
                
                features = load_bag_features(case_path)
                if features is None: continue

                logits = model(features)
                score = get_bag_score(logits, MODEL_CONFIG['type'])
                
                bag_scores.append(score)
                true_labels.append(label)
                case_names.append(case_name)

    # 轉換為 DataFrame
    df = pd.DataFrame({
        "Case": case_names,
        "Label": ["Healthy (B)" if l==1 else "Disease (A/C)" for l in true_labels],
        "Is_Healthy": true_labels, # 1=Healthy
        "Score": bag_scores
    })

    # === 1. 畫出分佈圖 (最重要的一步) ===
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Score", hue="Label", kde=True, bins=20, multiple="layer")
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold 0.5')
    plt.title("Bag Score Distribution (Mean Pooling)")
    plt.xlabel("Disease Probability (Avg)")
    plt.xlim(0, 1)
    plt.savefig("score_distribution.png")
    print("\n[圖表] 分佈圖已儲存為 score_distribution.png (請務必查看！)")

    # === 2. 尋找最佳門檻值 (Optimal Threshold) ===
    # 目標: 分開 A/C 與 B
    # 我們將 True Label 轉為 0=Healthy, 1=Disease 以便計算 ROC
    y_true_binary = [0 if l==1 else 1 for l in true_labels] # 1=Disease
    y_scores = bag_scores
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    
    # Youden's J statistic = TPR - FPR
    # 我們要找 J 最大的點
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    print(f"\n=== 最佳門檻分析 ===")
    print(f"建議最佳門檻 (Best Threshold): {best_thresh:.4f}")
    
    # 用這個門檻重新計算準確度
    df["Pred_Label"] = df["Score"].apply(lambda x: 0 if x > best_thresh else 1) # 0=Disease
    
    # 混淆矩陣
    # Note: df["Is_Healthy"] -> 0=Sick, 1=Healthy
    # df["Pred_Label"] -> 0=Sick, 1=Healthy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df["Is_Healthy"], df["Pred_Label"], labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    # Row 0: True Sick -> [Sick, Healthy]
    # Row 1: True Healthy -> [Sick, Healthy]
    
    print(f"\n使用門檻 {best_thresh:.4f} 的結果:")
    print(f"混淆矩陣:\n{cm}")
    print(f"病灶 (A/C) -> 判為病: {cm[0,0]} (召回數)")
    print(f"病灶 (A/C) -> 判為健: {cm[0,1]} (漏判數)")
    print(f"健康 (B)   -> 判為病: {cm[1,0]} (誤判數)")
    print(f"健康 (B)   -> 判為健: {cm[1,1]} (正確健康)")
    
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\n整體準確率 (Accuracy): {acc:.2%}")

    # 儲存 Excel
    df.to_excel("diagnosis_results_optimized.xlsx", index=False)
    print("詳細分類結果已儲存至 diagnosis_results_optimized.xlsx")

if __name__ == "__main__":
    run_diagnosis()