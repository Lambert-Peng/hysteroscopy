import torch
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download
import psutil
import threading
import time
from tqdm import tqdm
import sys

# ---------- 實時單行監控函數 ----------
def monitor_resources(interval=1):
    while monitor_resources.running:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        print(f"\033[F[資源監控] CPU: {cpu_percent:.1f}% | RAM: {ram.used / 1024**2:.1f}MB / {ram.total / 1024**2:.1f}MB")
        time.sleep(interval)
    print()  # 停止時換行

monitor_resources.running = True

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

if __name__ == "__main__":
    start_time = time.time()

    # 先輸出空行給監控使用
    print()  # 空行讓資源監控在這行刷新

    # 啟動實時監控執行緒
    monitor_thread = threading.Thread(target=monitor_resources, kwargs={'interval':1}, daemon=True)
    monitor_thread.start()

    try:
        # 設定測試圖片路徑
        image_folder = Path("test_images")
        image_paths = sorted(image_folder.glob("*.png")) + sorted(image_folder.glob("*.jpg"))

        if len(image_paths) == 0:
            raise FileNotFoundError("⚠️ 沒有找到圖片，請放圖片到 test_images 資料夾下！")

        # 處理圖片（進度條在單獨一行）
        images_list = []
        for p in tqdm(image_paths, desc="處理圖片", position=0):
            images_list.append(process_single_image(p))
        images = torch.stack(images_list)

        # 設定裝置
        device = "cpu"
        dtype = torch.float32

        # 載入模型
        model, loading_info = load_model_from_huggingface("egeozsoy/EndoViT")
        model = model.to(device, dtype)
        print("\n✅ 模型載入完成")

        # 移動圖片到裝置
        images = images.to(device, dtype)

        # 前向傳播（進度條在單獨一行）
        features_list = []
        for i in tqdm(range(images.shape[0]), desc="計算特徵", position=0):
            with torch.no_grad():
                feat = model.forward_features(images[i:i+1])
                features_list.append(feat)
        features = torch.cat(features_list, dim=0)

        # 存檔
        torch.save(features.cpu(), "features.pt")
        print("💾 特徵已存成 features.pt")

    except KeyboardInterrupt:
        print("\n⚠️ 偵測到 Ctrl+C，中斷程式...")

    finally:
        # 停止監控
        monitor_resources.running = False
        monitor_thread.join()

    end_time = time.time()
    print(f"⏱️ 總執行時間: {end_time - start_time:.1f} 秒")
