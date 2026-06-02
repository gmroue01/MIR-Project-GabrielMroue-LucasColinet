#!/usr/bin/env python3
"""
Pre-computes SIFT keypoints & descriptors for every image in the dataset.
Output : indexes/sift_ransac.npz  (~150-250 MB compressed)

Each image entry in the npz:
  {filename}_kp   (N, 2)   float16  — keypoint (x, y) pixel coordinates
  {filename}_des  (N, 128) uint8    — SIFT descriptors quantized to uint8

Usage
-----
    python scripts/generate_sift_index.py
    python scripts/generate_sift_index.py --dataset /path/to/images --output /path/to/out.npz

Notes
-----
- SIFT descriptors from OpenCV are float32 in [0, 512].
  We store them as uint8 = round(des / 2), so matching with float32 requires ×2
  (already handled in app/descriptors/sift_ransac.py).
- The index is required for SIFT-RANSAC reranking at search time.
  It is NOT committed to git (too large); generate once then upload to
  HuggingFace or keep on the Railway volume.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── Defaults ─────────────────────────────────────────────────────────────────
ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(ROOT, "dataset")
DEFAULT_OUTPUT  = os.path.join(ROOT, "indexes", "sift_ransac.npz")
MAX_KEYPOINTS   = 500


def extract(images_dir: str, max_kp: int) -> dict:
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(images_dir, e)))
    paths.sort()

    if not paths:
        print(f"[ERROR] No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} images — extracting SIFT (nfeatures={max_kp}) ...")

    sift   = cv2.SIFT_create(nfeatures=max_kp)
    data   = {}
    failed = 0

    for path in tqdm(paths, desc="SIFT", ncols=80):
        fn  = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            failed += 1
            data[f"{fn}_kp"]  = np.empty((0, 2),   dtype=np.float16)
            data[f"{fn}_des"] = np.empty((0, 128),  dtype=np.uint8)
            continue

        kp, des = sift.detectAndCompute(img, None)

        if des is None or len(kp) == 0:
            data[f"{fn}_kp"]  = np.empty((0, 2),   dtype=np.float16)
            data[f"{fn}_des"] = np.empty((0, 128),  dtype=np.uint8)
        else:
            kp_arr  = np.array([k.pt for k in kp], dtype=np.float16)
            des_arr = np.clip(des / 2.0, 0, 255).astype(np.uint8)
            data[f"{fn}_kp"]  = kp_arr
            data[f"{fn}_des"] = des_arr

    if failed:
        print(f"[WARN] {failed} images could not be read and were skipped.")

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate SIFT index for reranking.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Directory of images")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT,  help="Output .npz path")
    args = parser.parse_args()

    data = extract(args.dataset, MAX_KEYPOINTS)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving compressed index -> {args.output} ...")
    np.savez_compressed(args.output, **data)

    size_mb = os.path.getsize(args.output) / 1024 ** 2
    n_imgs  = len(data) // 2          # two arrays per image
    print(f"Done. {n_imgs} images | {size_mb:.1f} MB")
    print("Run the app and SIFT-RANSAC reranking will be available automatically.")


if __name__ == "__main__":
    main()
