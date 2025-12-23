import os

val_weight_root = r"C:\台大碩士資料\實驗室\hysteroscopy\EndoViT1\MILnewdata\data\val\weights"   # 修改成你的 val/weights 目錄

for file in os.listdir(val_weight_root):
    if file.endswith("_aug.pt"):          # 找到 augmented 檔案
        file_path = os.path.join(val_weight_root, file)
        print("Deleting:", file_path)
        os.remove(file_path)


'''
# 掃描 A / B / C 三個類別
for cls in ["A", "B", "C"]:
    class_dir = os.path.join(val_weight_root, cls)

    if not os.path.exists(class_dir):
        continue

    # 掃描每個 bag (e.g., A02, A15...)
    for bag in os.listdir(class_dir):
        bag_dir = os.path.join(class_dir, bag)

        if not os.path.isdir(bag_dir):
            continue

        # 掃描此 bag 下所有 weight 檔
        for file in os.listdir(bag_dir):
            if file.endswith("_aug.pt"):    # 找到 augmented 的 weight 檔案
                file_path = os.path.join(bag_dir, file)
                print("Deleting:", file_path)
                os.remove(file_path)'''

print("Done!")
