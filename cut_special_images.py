import os
from PIL import Image

# 原始資料集路徑
dataset_root = "hysteroscopy_dataset/B_351"
output_dir = "special_cropped_images"
os.makedirs(output_dir, exist_ok=True)

# 要處理的 Bxxx 圖片清單
# key = sample_id, value = list of image prefix (01, 02, ...)
targets = {
    "B110": ["01", "02"],
    "B111": ["01", "03"],
    "B113": ["01", "02", "03"],
    "B114": ["01", "02", "03"],
    "B115": ["01", "03"],
    "B116": ["01", "02", "03"],
    "B117": ["01", "02", "03", "04", "05"],
    "B119": ["01", "02", "03"],
    "B121": ["01"],
    "B122": ["02"],
    "B123": ["01", "02"],
    "B125": ["01"],
    "B128": ["01", "02", "03"],
    "B129": ["01", "02", "03"]
}

# 裁切區域 (left, upper, right, lower)
crop_box = (226, 69, 497, 321)

for sample_id, images in targets.items():
    sample_dir = os.path.join(dataset_root, sample_id)

    for prefix in images:
        # 假設檔名是 "01.jpg" 這種格式
        fname = f"{prefix}.jpg"
        src_path = os.path.join(sample_dir, fname)

        if not os.path.exists(src_path):
            print(f"檔案不存在: {src_path}")
            continue

        try:
            with Image.open(src_path) as img:
                cropped = img.crop(crop_box)

                # 輸出檔案命名方式 e.g. B110_01.jpg
                new_name = f"B351_{sample_id}_{prefix}.jpg"
                dst_path = os.path.join(output_dir, new_name)

                cropped.save(dst_path)
                print(f"已處理: {dst_path}")

        except Exception as e:
            print(f"處理失敗 {src_path}: {e}")