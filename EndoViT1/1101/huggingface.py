import os
import random
import shutil
import torch
import argparse
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download
from tqdm import tqdm
import time

# ---------- 資料增強與複製 ----------
def prepare_augmented_dataset(input_dir="test_images", output_dir="all_images", input_size=224):
    """
    若 input_dir 名稱包含 'train' -> 對影像做 resize 並在檔名 A/C 時做 augmentation（包含高斯雜訊）
    若 input_dir 名稱包含 'val'   -> 只複製原始檔案到 output_dir（不做 resize / augment）
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    image_paths = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))

    if len(image_paths) == 0:
        raise FileNotFoundError("沒有找到圖片，請確認資料夾是否正確且包含 .png 或 .jpg 檔案！")

    is_train = "train" in input_dir.lower()
    is_val = "val" in input_dir.lower()

    print(f"資料夾: {input_dir}")
    if is_train:
        print("模式: 訓練集 (進行 resize 與資料增強)")
    elif is_val:
        print("模式: 驗證集 (僅複製原始檔案，不做 resize / augment)")
    else:
        # 預設行為：若路徑既不含 train 也不含 val，當作 train 處理（可視需求改）
        print("警告: input 資料夾名稱未包含 'train' 或 'val'，預設為 train 模式（會做增強）")
        is_train = True

    base_transform = T.Resize((input_size, input_size))

    def add_gaussian_noise(img):
        """對影像新增高斯雜訊，輸入/輸出為 PIL Image"""
        std = 0.05  # 可調整雜訊強度
        tensor = TF.to_tensor(img)
        noise = torch.randn_like(tensor) * std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return TF.to_pil_image(noisy_tensor)

    augmentations = [
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.rotate(x, angle=random.choice([15, -15, 30, -30])),
        lambda x: TF.adjust_brightness(x, 1.2),
        lambda x: TF.adjust_contrast(x, 1.3),
        lambda x: TF.adjust_saturation(x, 1.4),
        lambda x: add_gaussian_noise(x),
    ]

    print(f"正在處理 {len(image_paths)} 張圖片...")
    for p in tqdm(image_paths, desc="資料準備", position=0):
        if is_val:
            # 只複製原始檔案（不改檔名、不 resize）
            shutil.copy2(p, Path(output_dir) / p.name)
            continue

        # 以下為 train 的處理（resize 並視情況做 augmentation）
        img = Image.open(p).convert("RGB")

        # 儲存 resize 後版本
        resized = base_transform(img)
        resized.save(Path(output_dir) / p.name)

        # 若檔名開頭是 A 或 C，額外產生 augmentation 影像
        if p.name[0].upper() in ["A", "C"]:
            aug_fn = random.choice(augmentations)
            aug_img = aug_fn(img)
            aug_resized = base_transform(aug_img)
            name, ext = os.path.splitext(p.name)
            aug_resized.save(Path(output_dir) / f"{name}_aug{ext}")

    print("資料準備完成！")
    return is_train  # 回傳是否為 train 模式，以便主程式決定是否繼續後續處理


# ---------- 圖片前處理 ----------
def process_single_image(image_path, input_size=224,
                         dataset_mean=[0.3464, 0.2280, 0.2228],
                         dataset_std=[0.2520, 0.2128, 0.2093]):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    image = Image.open(image_path).convert('RGB')
    processed = transform(image)
    return processed


# ---------- 載入模型 ----------
def load_model_from_huggingface(repo_id, model_filename="pytorch_model.bin"):
    model_path = snapshot_download(repo_id=repo_id, revision="main")
    model_weights_path = Path(model_path) / model_filename

    ckpt = torch.load(model_weights_path, map_location="cpu")
    model_weights = ckpt.get('model', ckpt)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).eval()

    loading = model.load_state_dict(model_weights, strict=False)
    return model, loading


# ---------- 主程式 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="影像特徵擷取工具（train/val 條件式處理）")
    parser.add_argument("--input", type=str, default=r"C:\台大碩士資料\實驗室\hysteroscopy\TrainTestDataset\val", help="輸入資料夾路徑")
    parser.add_argument("--output", type=str, default="all_images", help="輸出資料夾路徑")
    parser.add_argument("--size", type=int, default=224, help="影像輸入大小（train 模式下會被用來 resize）")
    args = parser.parse_args()

    start_time = time.time()

    try:
        # 準備資料集：若是 val 模式，該函式只會複製原始檔案並回傳 is_train=False
        prepare_augmented_dataset(args.input, args.output, input_size=224)


        # 以下為 train 模式：讀取 output 資料夾的影像並做前處理與特徵擷取
        image_folder = Path(args.output)
        image_paths = sorted(image_folder.glob("*.png")) + sorted(image_folder.glob("*.jpg"))

        if len(image_paths) == 0:
            raise FileNotFoundError("Output 資料夾沒有任何圖片，請檢查 prepare_augmented_dataset 是否成功。")

        images_list = []
        for p in tqdm(image_paths, desc="處理圖片", position=0):
            images_list.append(process_single_image(p, input_size=args.size))
        images = torch.stack(images_list)

        # 設定裝置 (自動偵測 GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        # 載入模型
        model, loading_info = load_model_from_huggingface("egeozsoy/EndoViT")
        model = model.to(device, dtype)
        print(f"\n模型載入完成，使用裝置：{device}")

        # 前向傳播
        images = images.to(device, dtype)
        features_list = []
        for i in tqdm(range(images.shape[0]), desc="計算特徵", position=0):
            with torch.no_grad():
                feat = model.forward_features(images[i:i+1])
                features_list.append(feat)
        features = torch.cat(features_list, dim=0)

        # 存檔
        torch.save(features.cpu(), "features.pt")
        print("特徵已存成 features.pt")

    except KeyboardInterrupt:
        print("\n偵測到 Ctrl+C，中斷程式...")

    end_time = time.time()
    print(f"總執行時間: {end_time - start_time:.1f} 秒")
