import os
import csv
from PIL import Image

# 主資料夾路徑
root_dir = r"C:\台大碩士資料\子宮頸癌data\hysteroscopy_dataset"

# === 用 csv 讀 annotation 檔案 ===
def load_annotation(filepath, group_tag):
    anno_dict = {}
    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) < 3:
                continue
            sample_id, image_name, comment = row[0], row[1], row[2]
            if comment.strip():  # 有註解 → A+ / C+
                anno_dict[(sample_id, image_name)] = f"{group_tag}+"
            else:               # 沒註解 → A / C
                anno_dict[(sample_id, image_name)] = group_tag
    return anno_dict

anno_A = load_annotation(r"C:\台大碩士資料\子宮頸癌data\hysteroscopy_dataset\annotation_A(utf8).csv", "A")
anno_C = load_annotation(r"C:\台大碩士資料\子宮頸癌data\hysteroscopy_dataset\annotation_C(utf8).csv", "C")

output_rows = []

for group in os.listdir(root_dir):
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path):
        continue

    for sample_id in os.listdir(group_path):
        sample_path = os.path.join(group_path, sample_id)
        if not os.path.isdir(sample_path):
            continue

        for fname in os.listdir(sample_path):
            if fname.lower().endswith((".jpg", ".png")):
                fpath = os.path.join(sample_path, fname)

                # 讀取圖片大小
                try:
                    with Image.open(fpath) as img:
                        size = f"{img.width}x{img.height}"
                except:
                    size = ""

                # 決定 Annotation
                if group.startswith("A"):
                    annotation = anno_A.get((sample_id, fname), "A")
                elif group.startswith("C"):
                    annotation = anno_C.get((sample_id, fname), "C")
                elif group.startswith("B"):
                    annotation = "B"
                else:
                    annotation = ""

                output_rows.append([group, sample_id, fname, size, annotation])

# === 輸出 CSV ===
with open("output.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Group", "Sample ID", "Image", "Size", "Annotation"])
    writer.writerows(output_rows)

print("✅ 已經生成 output.csv")
