import os
import torch

# ====== 設定 ======
image_dir = "all_dataset/images"         # 影像資料夾
pt_path = "features.pt"      # 原始權重 tensor
output_dir = "all_dataset/weights"       # 每張圖的權重輸出資料夾

os.makedirs(output_dir, exist_ok=True)
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
])

features = torch.load(pt_path, map_location='cpu')
assert len(image_files) == features.shape[0]

for idx, img_name in enumerate(image_files):
    single_feature = features[idx].clone()  # ← clone() 創建獨立 tensor
    base_name = os.path.splitext(img_name)[0]
    final_path = os.path.join(output_dir, f"{base_name}.pt")
    tmp_path = final_path + ".tmp"

    if os.path.exists(final_path):
        try:
            _ = torch.load(final_path, map_location='cpu')
            print(f"🟡 已存在且有效，跳過：{final_path}")
            continue
        except Exception:
            print(f"⚠️ 檔案損壞，重新寫入：{final_path}")
            os.remove(final_path)

    torch.save(single_feature, tmp_path)
    os.replace(tmp_path, final_path)
    print(f"✅ 已儲存：{final_path}")

print("🎉 全部完成！")