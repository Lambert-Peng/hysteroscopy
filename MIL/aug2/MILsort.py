import os
import shutil
import re

# 設定你的圖片來源資料夾
src_dir = r"C:\台大碩士資料\實驗室\hysteroscopy\EndoViT1\1101\data\train\weights"

# 設定要輸出的分類資料夾
dst_dir = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\sorteddata"

# 正規表達式：比對 Axx_AXX_xx..., Bxx_BXX_xx..., Cxx_CXX_xx...
pattern = re.compile(r'^([ABC])[0-9]+_([ABC]\d{1,3}(?:-\d{1,3})?)_')

for filename in os.listdir(src_dir):
    src_path = os.path.join(src_dir, filename)
    if not os.path.isfile(src_path):
        continue

    match = pattern.match(filename)
    if not match:
        print(f"跳過無法解析檔名：{filename}")
        continue

    major = match.group(1)        # A / B / C
    subfolder = match.group(2)    # AXX / BXX / CXX (可含 10-1)

    # 建立目的資料夾
    target_folder = os.path.join(dst_dir, major, subfolder)
    os.makedirs(target_folder, exist_ok=True)

    dst_path = os.path.join(target_folder, filename)

    # 改用複製
    shutil.copy2(src_path, dst_path)

    print(f"複製：{filename} → {major}/{subfolder}/")