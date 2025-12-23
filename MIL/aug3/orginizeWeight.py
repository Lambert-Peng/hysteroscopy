import torch
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_features(pt_file, output_root="dataset/val/weights"):
    print(f"正在讀取 {pt_file} ...")
    data = torch.load(pt_file)
    
    features = data["features"]   # Tensor [N, 768] (假設 EndoViT 輸出維度)
    filenames = data["filenames"] # List of strings
    
    if len(features) != len(filenames):
        raise ValueError("特徵數量與檔名數量不符！")

    print(f"共有 {len(filenames)} 筆資料待處理。")
    print(f"目標根目錄: {output_root}")

    for i in tqdm(range(len(filenames)), desc="整理檔案"):
        feat = features[i]
        fname = filenames[i]
        
        # 移除副檔名 (.jpg / .png)
        name_stem = os.path.splitext(fname)[0]
        
        # 解析檔名結構
        # 假設檔名格式: A63_A0_01.jpg 或 A63_A0_01_aug1.jpg
        # 使用 '_' 分割
        parts = name_stem.split('_')
        
        if len(parts) < 2:
            print(f"警告: 檔名 {fname} 格式不符合預期 (無法找到 BagID)，跳過。")
            continue
            
        # parts[0] -> 專案代號 (如 A63, B351)
        # parts[1] -> Bag ID / 資料夾名稱 (如 A0, A1, B100)
        bag_id = parts[1]
        
        # 類別通常是 Bag ID 的第一個字母 (A0 -> A, B100 -> B)
        class_name = bag_id[0].upper()
        
        # 如果 Bag ID 第一個字不是 A, B, C，可能需要例外處理
        if class_name not in ['A', 'B', 'C']:
            # 嘗試從 parts[0] 抓取 (例如 C66 -> C)
            class_name = parts[0][0].upper()

        # 建構目標路徑: dataset/train/weights/{Class}/{BagID}/
        target_dir = Path(output_root) / class_name / bag_id
        os.makedirs(target_dir, exist_ok=True)
        
        # 儲存單一 .pt 檔
        # 檔名範例: A63_A0_01.pt 或 A63_A0_01_aug1.pt
        save_name = f"{name_stem}.pt"
        save_path = target_dir / save_name
        
        # 將單一特徵 Tensor 存檔
        torch.save(feat.clone(), save_path)

    print("整理完成！")

if __name__ == "__main__":
    # 設定輸入檔案 (來自上一步驟的輸出)
    input_pt = "val_features.pt"
    
    # 設定輸出目錄
    output_dir = "dataset/val/weights"
    
    if os.path.exists(input_pt):
        organize_features(input_pt, output_dir)
    else:
        print(f"找不到 {input_pt}，請先執行 extract_features.py")