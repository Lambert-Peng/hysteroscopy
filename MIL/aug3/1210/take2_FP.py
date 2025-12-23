import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= 設定區 (請修改這裡) =================
# 1. Excel 路徑 (由 inference2.py 產生)
# 請指向 "bag_predictions_all_methods.xlsx"
EXCEL_PATH = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\1210\results\best\train2_1layer_mean_20251210_010556\MIL_Comparison_All\bag_predictions_all_methods.xlsx"

# 2. 原始圖片的「根目錄」路徑 (你的原始資料夾)
# 程式會去這裡找名稱符合 Bag ID 的資料夾 (例如 A63, C66)
SOURCE_IMG_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\hysteroscopy_dataset"  # <--- 請務必修改為你的實際路徑

# 3. 輸出目的地 (程式會把複製過來的圖片放在這裡)
TARGET_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\1210\results\best\train2_1layer_mean_20251210_010556\MIL_Comparison_All"

# 4. 篩選條件
# TARGET_METHOD: 你主要依賴哪種方法? (建議 'prob_topk' 或 'prob_mean')
TARGET_METHOD = "prob_topk" 
# THRESHOLD: 該方法的最佳閾值 (請參考 inference2.py 的輸出，例如 0.4 或 0.5)
THRESHOLD = 0.45

# ================= 主程式 =================
def main():
    # --- 檢查路徑 ---
    if not os.path.exists(EXCEL_PATH):
        print(f"錯誤：找不到 Excel 檔案 {EXCEL_PATH}")
        return
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"錯誤：找不到原始圖片目錄 {SOURCE_IMG_DIR}")
        return

    # --- 第一步：找出漏診名單 ---
    print(f"1. 正在讀取 Excel: {EXCEL_PATH}")
    df = pd.read_csv(EXCEL_PATH) if EXCEL_PATH.endswith('.csv') else pd.read_excel(EXCEL_PATH)
    
    # 篩選漏診 (True=1, Pred_Score < Threshold)
    fn_df = df[
        (df['true_label'] == 1) & 
        (df[TARGET_METHOD] < THRESHOLD)
    ].copy()
    
    # 取得 Bag ID 清單 (例如: "A63_A4", "B12_C3")
    target_bag_ids = set(fn_df['bag_id'].astype(str).str.strip().tolist())
    
    print("\n" + "="*60)
    print(f"【漏診清單 False Negatives】 (Method: {TARGET_METHOD}, Thr: {THRESHOLD})")
    print(f"共發現 {len(target_bag_ids)} 個被誤判為 B 的 A+C 病人")
    print("="*60)
    
    if len(target_bag_ids) == 0:
        print("恭喜！沒有發現漏診案例。")
        return

    # --- 第二步：複製圖片 ---
    print(f"\n2. 開始從 {SOURCE_IMG_DIR} 複製圖片...")
    
    # 重建輸出資料夾
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)
    
    found_count = 0
    missing_ids = []

    for bag_id in tqdm(target_bag_ids, desc="複製進度"):
        # --- 關鍵修改：路徑解析邏輯 ---
        # 假設 Bag ID 是 "A63_A4"，代表原始路徑是 "A63/A4"
        # 我們將 "_" 替換成系統的路徑分隔符號 (os.sep)
        relative_path = bag_id.replace("_", os.sep) 
        
        # 組合出完整的來源路徑: D:/Dataset/.../A63/A4
        src_folder_path = os.path.join(SOURCE_IMG_DIR, relative_path)
        
        # 目標路徑: results/.../A63_A4 (保持扁平方便查看，或者你也想維持巢狀?)
        # 這裡我們用扁平命名 "A63_A4" 作為資料夾名，方便你一眼看 ID
        dst_folder_path = os.path.join(TARGET_DIR, bag_id)
        
        if os.path.exists(src_folder_path) and os.path.isdir(src_folder_path):
            try:
                # 複製整個資料夾 (包含裡面的 01.jpg, 02.jpg...)
                shutil.copytree(src_folder_path, dst_folder_path)
                found_count += 1
            except Exception as e:
                print(f"複製 {bag_id} 時發生錯誤: {e}")
        else:
            # 嘗試另一種可能：也許 Bag ID 只有一層? (例如 "A63")
            # 再試一次直接用 bag_id 當資料夾名
            src_folder_path_flat = os.path.join(SOURCE_IMG_DIR, bag_id)
            if os.path.exists(src_folder_path_flat) and os.path.isdir(src_folder_path_flat):
                shutil.copytree(src_folder_path_flat, dst_folder_path)
                found_count += 1
            else:
                missing_ids.append(bag_id)
                # print(f"找不到路徑: {src_folder_path}") 

    # --- 第三步：輸出報告 ---
    print("\n" + "="*60)
    print("處理完成！")
    print(f"成功複製: {found_count} / {len(target_bag_ids)} 個病人的資料夾")
    print(f"圖片儲存位置: {os.path.abspath(TARGET_DIR)}")
    
    # 輸出 CSV
    csv_output = os.path.join(TARGET_DIR, "false_negative_list.csv")
    fn_df.to_csv(csv_output, index=False)
    print(f"清單已儲存至: {csv_output}")
    print("="*60)

    if len(missing_ids) > 0:
        print(f"\n⚠️ 警告：有 {len(missing_ids)} 個 Bag ID 找不到對應資料夾！")
        print(f"範例: {missing_ids[:5]}")
        print("請檢查：原始路徑中是否真的存在 A63/A4 這樣的結構？")

if __name__ == "__main__":
    main()