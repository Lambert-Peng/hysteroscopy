import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def split_dataset(input_dir, output_root, split_ratio=0.8, seed=42):
    """
    input_dir: 原始混合圖片資料夾
    output_root: 輸出的根目錄 (內含 train/val)
    split_ratio: 訓練集比例 (0.8 代表 8:2)
    """
    random.seed(seed)
    input_path = Path(input_dir)
    
    # 建立輸出資料夾
    train_dir = Path(output_root) / "train"
    val_dir = Path(output_root) / "val"
    
    if os.path.exists(output_root):
        print(f"警告: 輸出目錄 {output_root} 已存在，正在清除舊資料...")
        shutil.rmtree(output_root)
    
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # 1. 讀取所有圖片並依照 Bag ID 分組
    # 結構: { 'A': {'A0': [path1, path2...], 'A1': [...]}, 'B': {...}, 'C': {...} }
    grouped_data = {
        'A': defaultdict(list),
        'B': defaultdict(list),
        'C': defaultdict(list)
    }

    image_paths = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    
    print(f"找到 {len(image_paths)} 張圖片，正在進行分組...")

    for p in image_paths:
        filename = p.name
        # 解析檔名，例如 A63_A0_01.jpg
        parts = filename.split('_')
        
        # 安全檢查
        if len(parts) < 2:
            print(f"跳過格式錯誤的檔案: {filename}")
            continue
            
        # parts[1] 通常是 Bag ID (如 A0, B100)
        bag_id = parts[1]
        
        # 判斷類別 (取 Bag ID 第一個字)
        class_type = bag_id[0].upper()
        
        if class_type in ['A', 'B', 'C']:
            grouped_data[class_type][bag_id].append(p)
        else:
            # 若 Bag ID 不是 A/B/C 開頭，嘗試用 parts[0] (如 C66)
            alt_type = parts[0][0].upper()
            if alt_type in ['A', 'B', 'C']:
                grouped_data[alt_type][bag_id].append(p)

    # 2. 針對每個類別進行 Bag 層級的切分
    total_train = 0
    total_val = 0

    print("\n開始切分 (8:2)...")
    
    for cls in ['A', 'B', 'C']:
        bags = list(grouped_data[cls].keys())
        random.shuffle(bags) # 打亂 Bag 的順序
        
        split_idx = int(len(bags) * split_ratio)
        train_bags = bags[:split_idx]
        val_bags = bags[split_idx:]
        
        print(f"類別 {cls}: 總 Bag 數 {len(bags)} -> Train: {len(train_bags)}, Val: {len(val_bags)}")
        
        # 複製檔案到 Train
        for bag in train_bags:
            for img_path in grouped_data[cls][bag]:
                shutil.copy2(img_path, train_dir / img_path.name)
                total_train += 1
                
        # 複製檔案到 Val
        for bag in val_bags:
            for img_path in grouped_data[cls][bag]:
                shutil.copy2(img_path, val_dir / img_path.name)
                total_val += 1

    print(f"\n切分完成！")
    print(f"Train 總張數: {total_train} (儲存於 {train_dir})")
    print(f"Val 總張數:   {total_val} (儲存於 {val_dir})")

if __name__ == "__main__":
    # 請修改這裡的 input 路徑為你存放「全部原始圖片」的資料夾
    INPUT_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\padded_images" 
    OUTPUT_ROOT = "dataset_split"
    
    split_dataset(INPUT_DIR, OUTPUT_ROOT)