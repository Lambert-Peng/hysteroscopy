import os
import shutil
from sklearn.model_selection import train_test_split

# 原始資料路徑
base_dir = r"padded_images"
img_dir = os.path.join(base_dir, "images")

# 輸出路徑
out_dir = r"dataset_split"
os.makedirs(out_dir, exist_ok=True)
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)

# 依首字母分類 (A, B, C)
data_dict = {"A": [], "B": [], "C": []}
for f in os.listdir(img_dir):
    if f[0] in data_dict:  # 檢查首字母是否是A/B/C
        data_dict[f[0]].append(f)

# 按 8:2 分割並複製圖片
for label, files in data_dict.items():
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for split, file_list in [("train", train_files), ("val", val_files)]:
        for img_file in file_list:
            # 來源與目的地
            img_src = os.path.join(img_dir, img_file)
            img_dst = os.path.join(out_dir, split, "images", img_file)
            shutil.copy(img_src, img_dst)

print("影像資料分割完成！")