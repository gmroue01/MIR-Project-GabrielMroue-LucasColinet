#!/usr/bin/env python3
"""
Generate FAISS indexes for CLIP cross-modal search (Flickr8k).

Produces:
    indexes_faiss/index_images.faiss   — one 512D vector per image  (8 091 vectors)
    indexes_faiss/index_captions.faiss — one 512D vector per caption (40 455 vectors)

Prerequisites:
    - Flickr8k/Images/   must contain the 8 091 Flickr8k images
    - Flickr8k/captions.txt must exist
    - pip install open_clip_torch faiss-cpu torch tqdm pandas Pillow

Usage:
    python scripts/generate_faiss_index.py
"""

import os
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLICKR_DIR = os.path.join(ROOT, "Flickr8k")
IMAGES_DIR = os.path.join(FLICKR_DIR, "Images")
CAPTIONS   = os.path.join(FLICKR_DIR, "captions.txt")
OUT_DIR    = os.path.join(ROOT, "indexes_faiss")
BATCH_SIZE = 64


def load_model(device: str):
    print("Loading CLIP ViT-B/32 (OpenAI weights) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).half().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"  Model ready on {device}.")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, image_paths: list, device: str) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Encoding images"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device).half()
        vecs  = model.encode_image(batch)
        vecs  = vecs / vecs.norm(dim=-1, keepdim=True)
        all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0).astype(np.float32)


@torch.no_grad()
def encode_captions(model, tokenizer, captions: list, device: str) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Encoding captions"):
        batch = captions[i : i + BATCH_SIZE]
        tokens = tokenizer(batch).to(device)
        vecs   = model.encode_text(tokens)
        vecs   = vecs / vecs.norm(dim=-1, keepdim=True)
        all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0).astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main():
    # ── Checks ───────────────────────────────────────────────────────────────
    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERROR] {IMAGES_DIR} not found.")
        sys.exit(1)
    if not os.path.exists(CAPTIONS):
        print(f"[ERROR] {CAPTIONS} not found.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load CSV ─────────────────────────────────────────────────────────────
    df          = pd.read_csv(CAPTIONS)
    unique_imgs = df["image"].unique()
    captions    = df["caption"].tolist()
    print(f"Dataset: {len(unique_imgs)} images, {len(captions)} captions")

    image_paths = [os.path.join(IMAGES_DIR, fn) for fn in unique_imgs]
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"[WARN] {len(missing)} images not found, they will produce zero vectors.")

    # ── Load model ───────────────────────────────────────────────────────────
    model, preprocess, tokenizer = load_model(device)

    # ── Encode images ────────────────────────────────────────────────────────
    print(f"\nEncoding {len(unique_imgs)} images ...")
    t0         = time.time()
    img_vecs   = encode_images(model, preprocess, image_paths, device)
    print(f"  Done in {time.time()-t0:.1f}s — shape {img_vecs.shape}")

    # ── Encode captions ──────────────────────────────────────────────────────
    print(f"\nEncoding {len(captions)} captions ...")
    t0       = time.time()
    cap_vecs = encode_captions(model, tokenizer, captions, device)
    print(f"  Done in {time.time()-t0:.1f}s — shape {cap_vecs.shape}")

    # ── Build & save FAISS indexes ────────────────────────────────────────────
    print("\nBuilding FAISS indexes (IndexFlatIP) ...")
    idx_imgs = build_faiss_index(img_vecs)
    idx_caps = build_faiss_index(cap_vecs)

    out_imgs = os.path.join(OUT_DIR, "index_images.faiss")
    out_caps = os.path.join(OUT_DIR, "index_captions.faiss")

    faiss.write_index(idx_imgs, out_imgs)
    faiss.write_index(idx_caps, out_caps)

    print(f"  index_images.faiss  -> {os.path.getsize(out_imgs)/1024**2:.1f} MB  ({idx_imgs.ntotal} vectors)")
    print(f"  index_captions.faiss -> {os.path.getsize(out_caps)/1024**2:.1f} MB  ({idx_caps.ntotal} vectors)")
    print("\nDone. CLIP search is ready.")


if __name__ == "__main__":
    main()
