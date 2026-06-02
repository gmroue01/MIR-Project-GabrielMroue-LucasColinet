#!/usr/bin/env python3
"""
Measure real DL descriptor indexing times by running full inference on the dataset.

For each DL descriptor:
  1. Load the timm backbone (same architecture as the trained model)
  2. Preprocess + forward-pass all dataset images in batches
  3. Apply PCA + whitening + L2-norm (using existing pca_*.npz models)
  4. Time the full pipeline
  5. Update indexes/metrics.json with the measured time

The existing .npz index files are NOT modified — only metrics.json is updated.

Usage:
    python scripts/measure_indexing_time.py
    python scripts/measure_indexing_time.py --skip-slow   # skip ViT and DinoV2
"""
import os, sys, time, json, glob, argparse
import numpy as np
import cv2
import torch
import timm
from PIL import Image
from tqdm import tqdm

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET   = os.path.join(ROOT, "dataset")
INDEX_DIR = os.path.join(ROOT, "indexes")

sys.path.insert(0, ROOT)

# ── Model configs (same as deep_model.py) ────────────────────────────────────

# Models ordered fast → slow so results appear progressively
TIMM_CONFIGS = {
    "mobilenetv2": {"timm_name": "mobilenetv2_100.ra_in1k",         "img_size": 224, "batch_size": 64, "sample": None},
    "resnet50":    {"timm_name": "resnet50.a1_in1k",                 "img_size": 224, "batch_size": 32, "sample": None},
    "vit_base":    {"timm_name": "vit_base_patch16_224.augreg_in1k", "img_size": 224, "batch_size": 16, "sample": 200},
    "dinov2":      {"timm_name": "vit_small_patch14_dinov2.lvd142m", "img_size": 518, "batch_size": 4,  "sample": 100},
}
# sample=N : run inference on N images, extrapolate timing to full dataset

# DL descriptor → timm model name
DESCRIPTOR_TO_MODEL = {
    "dinov2_supcon":     "dinov2",
    "dinov2_zeroshot":   "dinov2",
    "mobilenet_arcface": "mobilenetv2",
    "mobilenet_zeroshot":"mobilenetv2",
    "resnet50_zeroshot": "resnet50",
    "vit_b16_zeroshot":  "vit_base",
}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EPS = 1e-3

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_batch(paths, img_size):
    """Load and preprocess a list of image paths into a (N, 3, H, W) tensor."""
    tensors = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            tensors.append(torch.zeros(3, img_size, img_size))
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors)


def extract_all(model, image_paths, img_size, batch_size, desc=""):
    """Run batched inference and return (N, D) float32 numpy array."""
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"  {desc}", ncols=72):
        batch_paths = image_paths[i:i + batch_size]
        batch = preprocess_batch(batch_paths, img_size)
        with torch.no_grad():
            feats = model(batch).float().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def apply_pca(X, pca_path):
    """Apply existing PCA model to embeddings. Returns reduced (N, k) array.
    If dims don't match (fine-tuned head vs backbone), fits a fresh PCA instead."""
    if not os.path.exists(pca_path):
        return X
    data = np.load(pca_path)
    mean = data["mean"]
    if mean.shape[0] != X.shape[1]:
        # Dimension mismatch: fine-tuned model had a different head.
        # Fit a temporary PCA of the same target size for timing purposes.
        k = int(data["reduced_dim"])
        Xc = X - X.mean(axis=0)
        C = (Xc.T @ Xc) / len(Xc)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        components = eigenvectors[:, idx][:, :k].T.astype(np.float32)
        scale = (1.0 / np.sqrt(eigenvalues[idx][:k] + EPS)).astype(np.float32)
        Xw = (Xc @ components.T) * scale
    else:
        components, whitening_scale = data["components"], data["whitening_scale"]
        Xw = (X - mean) @ components.T * whitening_scale
    norms = np.linalg.norm(Xw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (Xw / norms).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip ViT-B/16 and DinoV2 (very slow on CPU)")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(DATASET, "*.jpg")))
    if not image_paths:
        print(f"ERROR: no images found in {DATASET}")
        sys.exit(1)
    print(f"Dataset: {len(image_paths)} images\n")

    metrics_path = os.path.join(INDEX_DIR, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Group descriptors by model (preserve TIMM_CONFIGS order = fast → slow)
    model_to_descriptors: dict[str, list[str]] = {k: [] for k in TIMM_CONFIGS}
    for desc, model_key in DESCRIPTOR_TO_MODEL.items():
        if args.skip_slow and model_key in ("vit_base", "dinov2"):
            print(f"  SKIP {desc} (--skip-slow)")
            continue
        model_to_descriptors.setdefault(model_key, []).append(desc)

    for model_key, descriptors in model_to_descriptors.items():
        if not descriptors:
            continue
        cfg     = TIMM_CONFIGS[model_key]
        sample  = cfg["sample"]
        n_total = len(image_paths)
        paths   = image_paths[:sample] if sample else image_paths

        print(f"[{model_key}] Loading {cfg['timm_name']}...")
        model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=0)
        model.eval().float()

        label = f"{len(paths)} images (sample)" if sample else f"{len(paths)} images"
        print(f"  Extracting features for {label} "
              f"(batch={cfg['batch_size']}, size={cfg['img_size']}px)...")

        t0 = time.perf_counter()
        embeddings = extract_all(model, paths, cfg["img_size"], cfg["batch_size"], model_key)
        extract_time = time.perf_counter() - t0

        # Extrapolate to full dataset if using a sample
        if sample:
            extract_time = extract_time * (n_total / len(paths))
            print(f"  Extrapolated to {n_total} images: {extract_time:.1f}s")

        # Apply PCA for each descriptor (may differ in target dim)
        for desc in descriptors:
            pca_path = os.path.join(INDEX_DIR, f"pca_{desc}.npz")
            t1 = time.perf_counter()
            apply_pca(embeddings, pca_path)
            pca_time = time.perf_counter() - t1
            # Extrapolate PCA time too if needed
            if sample:
                pca_time = pca_time * (n_total / len(paths))

            total_time = round(extract_time + pca_time, 1)
            print(f"  [{desc}] extract={extract_time:.1f}s + pca={pca_time:.1f}s "
                  f"= {total_time:.1f}s total")

            if desc in metrics:
                metrics[desc]["indexing_time_s"] = total_time

        # Free memory
        del model
        print()

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics.json updated -> {metrics_path}")


if __name__ == "__main__":
    main()
