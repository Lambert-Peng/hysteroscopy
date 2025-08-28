import os
import shutil
import pandas as pd
from PIL import Image

# 輸入路徑
csv_file = "output.csv"
dataset_root = "hysteroscopy_dataset"
output_dir = "processed_images"

os.makedirs(output_dir, exist_ok=True)

# 讀取 output.csv
df = pd.read_csv(csv_file)

# 定義裁切區域，根據尺寸判斷
crop_boxes = {
    (600, 400): (163, 69, 437, 324),   # (left, bottom, right, top)
    (1434, 1064): (61, 0, 1328, 1064),
    (720, 480): (268, 82, 595, 387)
}

for idx, row in df.iterrows():
    group = row["Group"]        # e.g., "C_66"
    sample_id = row["Sample ID"]  # e.g., "C11"
    image_name = row["Image"]   # 原始檔名
    size = tuple(map(int, row["Size"].split("x")))  # e.g., "600x400" -> (600,400)

    src_path = os.path.join(dataset_root, group, sample_id, image_name)
    if not os.path.exists(src_path):
        print(f"檔案不存在: {src_path}")
        continue

    # 忽略 (1422,1062) 的圖片
    if size == (1422, 1062):
        print(f"忽略圖片: {src_path}")
        continue

    try:
        with Image.open(src_path) as img:
            if size in crop_boxes:
                box = crop_boxes[size]
                cropped = img.crop(box)
            else:
                cropped = img  # 沒定義的尺寸就不裁切

            # 重新命名 e.g. "C66_C11_01.jpg"
            new_name = f"{group.replace('_','')}_{sample_id}_{image_name}"
            dst_path = os.path.join(output_dir, new_name)

            cropped.save(dst_path)
            print(f"已處理: {dst_path}")

    except Exception as e:
        print(f"處理失敗 {src_path}: {e}")
