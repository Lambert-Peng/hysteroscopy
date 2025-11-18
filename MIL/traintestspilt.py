import os
import shutil
from sklearn.model_selection import train_test_split

# ==== 設定路徑 ====
root = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\sorteddata"     # 原始資料夾 (A B C)
new_root = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\dataset"   # 新資料夾 (train/val)

# 建立新資料夾結構
for split in ["train", "val"]:
    for cls in ["A", "B", "C"]:
        os.makedirs(os.path.join(new_root, split, "weights", cls), exist_ok=True)

# ==== 掃描所有 bag 資料夾 ====
data = []      # 資料夾完整路徑
labels = []    # A→0, B→1, C→2

class_map = {"A": 0, "B": 1, "C": 2}

for cls in ["A", "B", "C"]:
    class_dir = os.path.join(root, cls)
    folders = sorted(os.listdir(class_dir))

    for folder in folders:
        folder_path = os.path.join(class_dir, folder)
        if os.path.isdir(folder_path):
            data.append(folder_path)
            labels.append(class_map[cls])

# ==== 8:2 stratified split ====
train_bags, val_bags, train_labels, val_labels = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==== 定義 bag 複製函式 ====
def copy_bag_to(split, bag_path):
    """
    split: 'train' or 'val'
    bag_path: 例如 /root/A/A01
    """
    cls = os.path.basename(os.path.dirname(bag_path))  # A / B / C
    bag_name = os.path.basename(bag_path)              # A01, B10-1, C05...

    # 新的目的資料夾：train/weights/A/A01/
    out_dir = os.path.join(new_root, split, "weights", cls, bag_name)
    os.makedirs(out_dir, exist_ok=True)

    # 複製該 bag 內所有圖片
    for img in os.listdir(bag_path):
        src = os.path.join(bag_path, img)
        if os.path.isfile(src):  # 避免讀到隱藏檔
            dst = os.path.join(out_dir, img)
            shutil.copy(src, dst)

# ==== 開始複製 ====
print("Copying train bags...")
for bag in train_bags:
    copy_bag_to("train", bag)

print("Copying val bags...")
for bag in val_bags:
    copy_bag_to("val", bag)

print("Done!")
