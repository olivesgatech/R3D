import os
import torch
import clip
import numpy as np
from PIL import Image

# 1. 경로 설정
image_folder = "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd"  # 이미지가 들어있는 최상위 폴더
output_root = "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_features"     # 저장할 최상위 폴더
os.makedirs(output_root, exist_ok=True)

# 2. CLIP 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 3. 이미지 순회 및 feature 저장
with torch.no_grad():
    for root, _, files in os.walk(image_folder):
        jpg_files = [f for f in files if f.lower().endswith(".jpg")]
        if not jpg_files:
            continue

        # output 폴더 경로 설정 (원본 폴더 구조 유지)
        relative_path = os.path.relpath(root, image_folder)
        output_path = os.path.join(output_root, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for fname in jpg_files:
            img_path = os.path.join(root, fname)
            basename = os.path.splitext(fname)[0]
            save_path = os.path.join(output_path, f"{basename}.npy")
            if os.path.exists(save_path):
                continue
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                features = model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy()
                np.save(save_path, features_np)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

print("✅ All features saved with preserved folder structure.")
