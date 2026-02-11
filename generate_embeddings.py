#!/usr/bin/env python3
"""
상품 이미지에서 임베딩 DB 생성 스크립트

사용법:
    python generate_embeddings.py

디렉토리 구조:
    product_images/
    ├── 콜라/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── img3.jpg
    ├── 사이다/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── ...
"""

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Streamlit shim 먼저 import
import sys
sys.path.insert(0, str(Path(__file__).parent))
import backend.st_shim  # noqa: F401

from checkout_core.inference import load_models, extract_dino_embedding, extract_clip_embedding


def generate_embeddings_db(
    images_dir: str = "product_images",
    output_dir: str = "data",
):
    """상품 이미지 폴더에서 임베딩 DB 생성

    Args:
        images_dir: 상품 이미지 폴더 (하위 폴더 = 상품 이름)
        output_dir: 출력 디렉토리 (embeddings.npy, labels.npy)
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"{images_dir} 폴더가 없습니다.")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. 모델 로딩
    print("🔧 모델 로딩 중...")
    bundle = load_models(adapter_dir=str(output_path))
    print(f"   ✓ DINOv3 loaded (LoRA: {bundle['lora_loaded']})")
    print(f"   ✓ CLIP loaded")
    print(f"   ✓ Device: {bundle['device']}")

    dino_dim = bundle["dino_dim"]
    clip_dim = bundle["clip_dim"]

    # 2. 이미지 수집
    print(f"\n📂 {images_dir}/ 스캔 중...")
    image_files = []
    labels = []

    for product_dir in sorted(images_path.iterdir()):
        if not product_dir.is_dir():
            continue

        product_name = product_dir.name
        images = list(product_dir.glob("*.jpg")) + list(product_dir.glob("*.png"))

        if not images:
            print(f"   ⚠️  {product_name}: 이미지 없음")
            continue

        print(f"   ✓ {product_name}: {len(images)}장")

        for img_path in images:
            image_files.append(img_path)
            labels.append(product_name)

    if not image_files:
        raise ValueError("이미지가 없습니다!")

    print(f"\n총 {len(image_files)}장 (상품 {len(set(labels))}개)")

    # 3. 임베딩 생성
    print("\n🚀 임베딩 생성 중...")
    all_embeddings = []

    for img_path, label in tqdm(zip(image_files, labels), total=len(image_files)):
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ⚠️  {img_path} 로드 실패")
            continue

        # DINOv3 임베딩
        dino_emb = extract_dino_embedding(
            img,
            bundle["dino_model"],
            bundle["dino_processor"],
            bundle["device"],
        )

        # CLIP 임베딩
        clip_emb = extract_clip_embedding(
            img,
            bundle["clip_model"],
            bundle["clip_processor"],
            bundle["device"],
        )

        # 연결
        combined = np.concatenate([dino_emb, clip_emb], axis=0)
        all_embeddings.append(combined)

    # 4. 저장
    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
    labels_array = np.array(labels, dtype=object)

    embeddings_file = output_path / "embeddings.npy"
    labels_file = output_path / "labels.npy"

    np.save(embeddings_file, embeddings)
    np.save(labels_file, labels_array)

    print(f"\n✅ 완료!")
    print(f"   📁 {embeddings_file}")
    print(f"      Shape: {embeddings.shape} (N × {dino_dim + clip_dim})")
    print(f"   📁 {labels_file}")
    print(f"      Shape: {labels_array.shape}")
    print(f"\n💡 faiss_index.bin은 서버 시작 시 자동 생성됩니다.")


if __name__ == "__main__":
    generate_embeddings_db()
