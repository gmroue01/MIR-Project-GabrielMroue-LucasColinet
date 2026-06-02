#!/usr/bin/env python3
"""
Recompute indexes/metrics.json for all active descriptors.

- Classical descriptors (color_histogram, sift): re-extracted from images.
- Deep-learning descriptors:
    1. Load .pth -> align to filenames.npy -> save full .npz  (timed)
    2. Apply PCA + whitening + L2-norm -> overwrite .npz       (timed together)
  Reported indexing_time_s covers the full pipeline.

Usage:
    python scripts/compute_metrics.py
"""
import os, sys, time, json, glob, re
import numpy as np
import cv2
import torch
from tqdm import tqdm

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(ROOT, "dataset")
INDEX_DIR  = os.path.join(ROOT, "indexes")
MODELS_DIR = os.path.join(ROOT, "modelsV2")

sys.path.insert(0, ROOT)
from app.descriptors import color_histogram, sift

CLASSICAL = {
    "color_histogram": color_histogram.extract,
    "sift":            sift.extract,
}

DL_MODELS = {
    "dinov2_supcon":     "Copie de DinoV2_SupCon_finetuned_best.pth",
    "dinov2_zeroshot":   "Copie de DinoV2_finetuned_best.pth",
    "mobilenet_arcface": "Copie de MobileNetV2_ARCFace_finetuned_best.pth",
    "mobilenet_zeroshot":"Copie de MobileNetV2_zero_shot.pth",
    "resnet50_zeroshot": "Copie de ResNet50_zero_shot.pth",
    "vit_b16_zeroshot":  "Copie de ViT_B_16_zero_shot.pth",
}

# PCA recommended dims (from pca_recommendations_report.md)
PCA_DIMS = {
    "dinov2_supcon":     13,
    "dinov2_zeroshot":   13,
    "mobilenet_arcface": 49,
    "vit_b16_zeroshot":  230,
    "mobilenet_zeroshot":257,
    "resnet50_zeroshot": 256,
}

EPS = 1e-3


def normalize_fn(name: str) -> str:
    return re.sub(r" \(", "(", name)


# ── Classical ─────────────────────────────────────────────────────────────────

def index_classical(name, extract_fn, image_paths):
    features, times, failed = [], [], 0
    for path in tqdm(image_paths, desc=f"  {name}", ncols=72):
        img = cv2.imread(path)
        if img is None:
            features.append(np.zeros(1, dtype=np.float32))
            failed += 1
            continue
        t0 = time.perf_counter()
        f  = extract_fn(img).astype(np.float32)
        times.append(time.perf_counter() - t0)
        features.append(f)

    dim    = max(f.shape[0] for f in features)
    matrix = np.zeros((len(features), dim), dtype=np.float32)
    for i, f in enumerate(features):
        matrix[i, :f.shape[0]] = f

    np.savez_compressed(os.path.join(INDEX_DIR, f"{name}.npz"), features=matrix)
    return matrix, times, failed


# ── PCA + whitening ───────────────────────────────────────────────────────────

def fit_pca_whitening(X: np.ndarray, k: int):
    mean = X.mean(axis=0)
    Xc   = X - mean
    C    = (Xc.T @ Xc) / len(Xc)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    components       = eigenvectors[:, :k].T.astype(np.float32)
    whitening_scale  = (1.0 / np.sqrt(eigenvalues[:k] + EPS)).astype(np.float32)
    return mean.astype(np.float32), components, whitening_scale


def apply_pca(X, mean, components, whitening_scale):
    Xc     = X - mean
    Xp     = Xc @ components.T
    Xw     = Xp * whitening_scale
    norms  = np.linalg.norm(Xw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (Xw / norms).astype(np.float32)


# ── DL indexation ─────────────────────────────────────────────────────────────

def index_dl(name, pth_filename, ref_filenames):
    """Load .pth, align, apply PCA. Returns (reduced_matrix, elapsed_s, original_dim)."""
    pth_path = os.path.join(MODELS_DIR, pth_filename)
    if not os.path.exists(pth_path):
        print(f"  SKIP — {pth_path} not found")
        return None, 0.0, 0

    t0 = time.perf_counter()

    # 1. Load
    data          = torch.load(pth_path, map_location="cpu", weights_only=False)
    raw_emb       = data["embeddings"]
    raw_filenames = [os.path.basename(str(f)) for f in data["filenames"]]
    emb_np = raw_emb.float().numpy() if isinstance(raw_emb, torch.Tensor) \
             else np.array(raw_emb, dtype=np.float32)
    original_dim = emb_np.shape[1]

    # 2. Align to ref_filenames order
    fn_to_idx = {normalize_fn(fn): i for i, fn in enumerate(raw_filenames)}
    n         = len(ref_filenames)
    matrix    = np.zeros((n, original_dim), dtype=np.float32)
    missing   = 0
    for i, fn in enumerate(ref_filenames):
        key = normalize_fn(fn)
        if key in fn_to_idx:
            matrix[i] = emb_np[fn_to_idx[key]]
        else:
            missing += 1
    if missing:
        print(f"  WARNING: {missing} filenames missing")

    # 3. PCA + whitening + L2
    k = PCA_DIMS.get(name, original_dim)
    mean, components, whitening_scale = fit_pca_whitening(matrix, k)
    reduced = apply_pca(matrix, mean, components, whitening_scale)

    # 4. Save reduced index
    np.savez_compressed(os.path.join(INDEX_DIR, f"{name}.npz"), features=reduced)

    # 5. Save PCA model
    np.savez_compressed(
        os.path.join(INDEX_DIR, f"pca_{name}.npz"),
        mean=mean,
        components=components,
        whitening_scale=whitening_scale,
        original_dim=np.array(original_dim),
        reduced_dim=np.array(k),
    )

    elapsed = time.perf_counter() - t0
    return reduced, elapsed, original_dim


# ── Search timing ─────────────────────────────────────────────────────────────

def measure_search_time(name, n_queries=100):
    path = os.path.join(INDEX_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return 0.0
    matrix = np.load(path)["features"].astype(np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    n    = min(n_queries, len(matrix))
    idxs = np.random.choice(len(matrix), n, replace=False)
    ts = []
    for i in idxs:
        q  = matrix[i]
        t0 = time.perf_counter()
        np.dot(matrix, q)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    image_paths   = sorted(glob.glob(os.path.join(DATASET, "*.jpg")))
    ref_filenames = [os.path.basename(p) for p in image_paths]
    print(f"Dataset: {len(image_paths)} images\n")

    metrics = {}

    # Classical
    for name, fn in CLASSICAL.items():
        print(f"[{name}] Indexing...")
        t_start = time.perf_counter()
        matrix, times, failed = index_classical(name, fn, image_paths)
        indexing_time = time.perf_counter() - t_start

        size_mb    = os.path.getsize(os.path.join(INDEX_DIR, f"{name}.npz")) / 1024**2
        avg_search = measure_search_time(name)

        metrics[name] = {
            "indexing_time_s":    round(indexing_time, 3),
            "descriptor_size_mb": round(size_mb, 4),
            "avg_search_time_s":  round(avg_search, 6),
            "num_images":         len(image_paths),
            "descriptor_dim":     int(matrix.shape[1]),
            "failed":             failed,
            "source":             "computed",
        }
        print(f"  Done -> {indexing_time:.1f}s | {size_mb:.2f} MB | "
              f"dim={matrix.shape[1]} | search={avg_search*1000:.2f}ms/query\n")

    # DL
    for name, pth_file in DL_MODELS.items():
        print(f"[{name}] Loading .pth + PCA...")
        reduced, indexing_time, orig_dim = index_dl(name, pth_file, ref_filenames)
        if reduced is None:
            continue

        size_mb    = os.path.getsize(os.path.join(INDEX_DIR, f"{name}.npz")) / 1024**2
        avg_search = measure_search_time(name)
        k          = int(reduced.shape[1])

        metrics[name] = {
            "indexing_time_s":    round(indexing_time, 3),
            "descriptor_size_mb": round(size_mb, 4),
            "avg_search_time_s":  round(avg_search, 6),
            "num_images":         int(reduced.shape[0]),
            "descriptor_dim":     k,
            "original_dim":       orig_dim,
            "failed":             0,
            "source":             "precomputed",
        }
        print(f"  Done -> {indexing_time:.1f}s | {size_mb:.2f} MB | "
              f"dim={orig_dim}->{k} | search={avg_search*1000:.2f}ms/query\n")

    out_path = os.path.join(INDEX_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics.json saved -> {out_path}")
    print(f"Descriptors: {list(metrics.keys())}")


if __name__ == "__main__":
    main()
