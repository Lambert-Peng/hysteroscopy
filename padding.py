import os
from PIL import Image

input_dir = "processed_images"
output_dir = "padded_images"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    src_path = os.path.join(input_dir, fname)
    dst_path = os.path.join(output_dir, fname)

    try:
        with Image.open(src_path) as img:
            w, h = img.size
            # 建立 224x224 的黑底 (你可以改成白底填充)
            new_img = Image.new("RGB", (224, 224), (0, 0, 0))

            # 等比例縮放，最大邊對齊 224
            scale = min(224/w, 224/h)
            new_w, new_h = int(w*scale), int(h*scale)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 貼到中央
            left = (224 - new_w) // 2
            top = (224 - new_h) // 2
            new_img.paste(resized, (left, top))

            new_img.save(dst_path)
            print(f"已Padding: {dst_path}")

    except Exception as e:
        print(f"Padding失敗 {src_path}: {e}")
