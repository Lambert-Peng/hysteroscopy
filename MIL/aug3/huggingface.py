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

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def prepare_dataset(input_dir, output_dir, mode="train", input_size=224):
    """
    mode='train': 
       - B類: Resize (1x)
       - A/C類: Resize + 3 Augmentations (4x)
    mode='val':
       - 所有類別: Resize (1x), 不做 Augmentation
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    image_paths = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"在 {input_dir} 找不到圖片！")

    print(f"[{mode.upper()}模式] 處理資料夾: {input_dir}, 圖片數量: {len(image_paths)}")

    base_transform = T.Resize((input_size, input_size))

    def add_gaussian_noise(img):
        std = 0.05
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
        lambda x: add_gaussian_noise(x),
    ]

    for p in tqdm(image_paths, desc="影像前處理"):
        img = Image.open(p).convert("RGB")
        filename = p.name
        name_part, ext = os.path.splitext(filename)
        
        # 1. 基礎處理：所有圖片都要儲存一張原始 Resize 版
        resized_original = base_transform(img)
        resized_original.save(Path(output_dir) / filename)

        # 2. 判斷是否需要擴增
        # 只有在 mode='train' 且 類別為 A 或 C 時才做
        if mode == "train":
            class_prefix = name_part[0].upper() # 簡單判斷檔名首字
            
            # 如果是 A 或 C，額外產生 3 張擴增圖
            if class_prefix in ['A', 'C']:
                for i in range(1, 4):
                    aug_fn = random.choice(augmentations)
                    aug_img = aug_fn(img)
                    aug_resized = base_transform(aug_img)
                    new_name = f"{name_part}_aug{i}{ext}"
                    aug_resized.save(Path(output_dir) / new_name)

    print(f"資料準備完成！存於 {output_dir}")

def process_single_image(image_path, input_size=224):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.3464, 0.2280, 0.2228], std=[0.2520, 0.2128, 0.2093])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image), image_path.name

def load_model(repo_id="egeozsoy/EndoViT"):
    print(f"載入模型: {repo_id}")
    model_path = snapshot_download(repo_id=repo_id, revision="main")
    model_weights_path = Path(model_path) / "pytorch_model.bin"
    ckpt = torch.load(model_weights_path, map_location="cpu")
    model_weights = ckpt.get('model', ckpt)
    
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).eval()
    model.load_state_dict(model_weights, strict=False)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="輸入圖片資料夾")
    parser.add_argument("--mode", type=str, choices=["train", "val"], required=True, help="模式: train (做平衡擴增) 或 val (僅Resize)")
    parser.add_argument("--output_pt", type=str, required=True, help="輸出的特徵檔名 (.pt)")
    args = parser.parse_args()

    # 設定暫存資料夾名稱 (避免 train/val 混用)
    temp_dir = f"temp_processed_{args.mode}"
    
    # 1. 準備資料
    prepare_dataset(args.input, temp_dir, mode=args.mode, input_size=224)

    # 2. 讀取處理後的圖片
    image_folder = Path(temp_dir)
    image_paths = sorted(list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg")))
    
    images_list = []
    filenames_list = []
    
    print("轉為 Tensor 中...")
    for p in image_paths:
        tensor, name = process_single_image(p)
        images_list.append(tensor)
        filenames_list.append(name)

    if not images_list:
        print("無圖片可處理。")
        exit()

    images = torch.stack(images_list)
    
    # 3. 提取特徵
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    model = load_model().to(device)
    
    features_list = []
    batch_size = 32
    images = images.to(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="提取特徵"):
            batch = images[i : i + batch_size]
            feat = model.forward_features(batch)
            features_list.append(feat.cpu())

    all_features = torch.cat(features_list, dim=0)
    
    # 4. 存檔
    torch.save({"features": all_features, "filenames": filenames_list}, args.output_pt)
    print(f"完成！特徵已存為: {args.output_pt}")
    
    # 清理暫存 (可選)
    # shutil.rmtree(temp_dir)