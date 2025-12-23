import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ================= 設定區 (請修改這裡) =================
# 1. Excel 路徑 (請指向您最新跑出來的 Bag Level Excel)
# 例如: inference_bag_topk.py 或 Bag_inference.py 產出的檔案
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\best\combined_eval_thr_0.3\combined_predictions.xlsx"

# 2. 原始圖片的「根目錄」路徑
# 程式會去這裡找 A63/A4/01.jpg 這樣的結構
SOURCE_IMG_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\hysteroscopy_dataset" 

# 3. 輸出目的地
TARGET_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\best\combined_eval_thr_0.3"

# ================= 輔助函式 =================
def extract_bag_id(filename):
    """從檔名提取 Bag ID"""
    parts = str(filename).split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def calculate_bag_predictions(df):
    """
    重現 MIL_inf.py 中的 Max Pooling 邏輯
    """
    bag_results = []
    
    # 確保有 bag_id
    if 'bag_id' not in df.columns:
        df['bag_id'] = df['filename'].apply(extract_bag_id)
        
    grouped = df.groupby('bag_id')
    
    print(f"正在計算 Bag Level 預測 (共 {len(grouped)} 位病人)...")
    
    for bag_id, group in grouped:
        true_label = group['true_label'].iloc[0]
        
        # --- 這裡完全複製 MIL_inf.py 的 Max Pooling 邏輯 ---
        # 1. 檢查是否有任何一張被 M1 (Thr=0.3) 判為異常 (不是 B/1)
        preds = group['M1_pred'].values
        
        if np.any(preds != 1): # 1 is B
            # 判為異常，區分 A 或 C
            # 邏輯：用 "加總機率" 來決定是 A 多還是 C 多
            sum_prob_A = group['prob_A'].sum()
            sum_prob_C = group['prob_C'].sum()
            
            final_pred = 0 if sum_prob_A > sum_prob_C else 2
        else:
            # 全部都是 B，則 Bag 判為 B
            final_pred = 1 
            
        bag_results.append({
            'bag_id': bag_id,
            'true_label': true_label,
            'pred_label': final_pred
        })
        
    return pd.DataFrame(bag_results)

# ================= 主程式 =================
def main():
    # --- 檢查路徑 ---
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到 Excel 檔案 {EXCEL_PATH}")
        return
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"錯誤：找不到原始圖片目錄 {SOURCE_IMG_DIR}")
        return

    # --- 第一步：讀取與計算 ---
    print(f"1. 讀取 Excel: {EXCEL_PATH}")
    df_instance = pd.read_csv(EXCEL_PATH) if EXCEL_PATH.endswith('.csv') else pd.read_excel(EXCEL_PATH)
    
    # 計算 Bag Level 結果
    df_bag = calculate_bag_predictions(df_instance)
    
    # --- 第二步：篩選混淆案例 ---
    # 標籤定義: 0=A, 1=B, 2=C
    
    # Case 1: True A (0) -> Pred C (2)
    df_A_to_C = df_bag[(df_bag['true_label'] == 0) & (df_bag['pred_label'] == 2)].copy()
    
    # Case 2: True C (2) -> Pred A (0)
    df_C_to_A = df_bag[(df_bag['true_label'] == 2) & (df_bag['pred_label'] == 0)].copy()
    
    print("\n" + "="*60)
    print("【AC 混淆錯誤統計 (基於 MIL Max Pooling)】")
    print(f"1. True A -> Pred C: 共 {len(df_A_to_C)} 位")
    print(f"2. True C -> Pred A: 共 {len(df_C_to_A)} 位")
    print("="*60)

    # --- 第三步：定義複製函式 (支援巢狀路徑) ---
    def copy_bags(bag_list, subfolder_name):
        target_sub_dir = os.path.join(TARGET_DIR, subfolder_name)
        
        if os.path.exists(target_sub_dir):
            shutil.rmtree(target_sub_dir)
        os.makedirs(target_sub_dir)
        
        count = 0
        missing = []
        
        # [修改] 設定只要複製的圖片格式
        VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        print(f"\n正在複製: {subfolder_name} ...")
        
        for bag_id in tqdm(bag_list, desc="Copying"):
            bag_id = str(bag_id).strip()
            
            # 解析路徑 A63_A4 -> A63/A4
            relative_path = bag_id.replace("_", os.sep)
            src_path = os.path.join(SOURCE_IMG_DIR, relative_path)
            
            # 備用方案: 扁平路徑
            if not (os.path.exists(src_path) and os.path.isdir(src_path)):
                src_path = os.path.join(SOURCE_IMG_DIR, bag_id)
            
            # 目標路徑
            dst_path = os.path.join(target_sub_dir, bag_id)
            
            # --- [開始修改核心複製邏輯] ---
            if os.path.exists(src_path) and os.path.isdir(src_path):
                try:
                    # 1. 先手動建立目標資料夾
                    if not os.path.exists(dst_path):
                        os.makedirs(dst_path)
                    
                    files_copied_count = 0
                    
                    # 2. 遍歷來源資料夾中的每一個檔案
                    for filename in os.listdir(src_path):
                        # 3. 檢查副檔名 (忽略大小寫)
                        if filename.lower().endswith(VALID_EXTENSIONS):
                            src_file = os.path.join(src_path, filename)
                            dst_file = os.path.join(dst_path, filename)
                            
                            # 4. 複製檔案
                            shutil.copy2(src_file, dst_file)
                            files_copied_count += 1
                    
                    # 確認是否有複製到圖片
                    if files_copied_count > 0:
                        count += 1
                    else:
                        # 如果資料夾裡面沒有圖片，就把剛建好的空資料夾刪掉
                        os.rmdir(dst_path)
                        print(f"  [Skip] {bag_id}: 找到資料夾但裡面沒有圖片檔")
                        
                except Exception as e:
                    print(f"  Error copying {bag_id}: {e}")
            else:
                missing.append(bag_id)
            # --- [修改結束] ---
        
        return count, missing

    # --- 第四步：執行複製 ---
    # 1. 複製 A -> C
    if len(df_A_to_C) > 0:
        ids = set(df_A_to_C['bag_id'].tolist())
        cnt, miss = copy_bags(ids, "True_A_Pred_C")
        # 存清單
        df_A_to_C.to_csv(os.path.join(TARGET_DIR, "list_True_A_Pred_C.csv"), index=False)
        if miss: print(f"  ⚠️ 未找到: {miss}")

    # 2. 複製 C -> A
    if len(df_C_to_A) > 0:
        ids = set(df_C_to_A['bag_id'].tolist())
        cnt, miss = copy_bags(ids, "True_C_Pred_A")
        # 存清單
        df_C_to_A.to_csv(os.path.join(TARGET_DIR, "list_True_C_Pred_A.csv"), index=False)
        if miss: print(f"  ⚠️ 未找到: {miss}")

    # --- 第五步：總結 ---
    print("\n" + "="*60)
    print("處理完成！")
    print(f"輸出位置: {os.path.abspath(TARGET_DIR)}")
    print("="*60)

if __name__ == "__main__":
    main()