import os
import shutil
from sklearn.model_selection import train_test_split

# 原始資料路徑
base_dir = r"all_dataset"
img_dir = os.path.join(base_dir, "images")
weight_dir = os.path.join(base_dir, "weights")

# 輸出路徑
out_dir = r"dataset_split"
os.makedirs(out_dir, exist_ok=True)
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, split, "weights"), exist_ok=True)

# 依首字母分類 (A, B, C)
data_dict = {"A": [], "B": [], "C": []}
for f in os.listdir(img_dir):
    if f[0] in data_dict:  # 檢查首字母是否是A/B/C
        data_dict[f[0]].append(f)

# 按 8:2 分割並複製
for label, files in data_dict.items():
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for split, file_list in [("train", train_files), ("val", val_files)]:
        for img_file in file_list:
            name, _ = os.path.splitext(img_file)
            weight_file = name + ".pt"

            # 來源
            img_src = os.path.join(img_dir, img_file)
            weight_src = os.path.join(weight_dir, weight_file)

            # 目的地
            img_dst = os.path.join(out_dir, split, "images", img_file)
            weight_dst = os.path.join(out_dir, split, "weights", weight_file)

            # 複製檔案（若 weight 不存在則略過）
            shutil.copy(img_src, img_dst)
            if os.path.exists(weight_src):
                shutil.copy(weight_src, weight_dst)

print("影像與權重分割完成！")
