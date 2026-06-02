#!/usr/bin/env python3
"""
Apply PCA + whitening + L2-norm to all DL descriptor indexes.

Recommended dimensions from pca_recommendations_report.md (95% variance threshold):
  dinov2_supcon    256 -> 13
  dinov2_zeroshot  256 -> 13
  mobilenet_arcface 256 -> 49
  vit_b16_zeroshot  768 -> 230
  mobilenet_zeroshot 1280 -> 257
  resnet50_zeroshot  2048 -> 256

For each descriptor:
  1. Load existing .npz embeddings
  2. Fit PCA with whitening (regularised, eps=1e-3) on all vectors
  3. Transform + L2-normalise
  4. Overwrite the .npz with reduced embeddings
  5. Save the PCA model (mean / components / whitening scale) to pca_{name}.npz

Usage:
    python scripts/apply_pca.py
"""
import os, sys
import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(ROOT, "indexes")

# Recommended dims from the report
RECOMMENDED_DIMS = {
    "dinov2_supcon":     13,
    "dinov2_zeroshot":   13,
    "mobilenet_arcface": 49,
    "vit_b16_zeroshot":  230,
    "mobilenet_zeroshot":257,
    "resnet50_zeroshot": 256,
}

EPS = 1e-3  # whitening regularisation


def fit_pca_whitening(X: np.ndarray, k: int):
    """
    Fit PCA with whitening on X (n x d).
    Returns (mean, components k x d, whitening_scale k).
    """
    mean = X.mean(axis=0)
    Xc   = X - mean

    # Covariance matrix (d x d) — more efficient than full SVD on (n x d)
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / n

    # Eigendecomposition (eigh for symmetric matrix, returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep top-k
    components       = eigenvectors[:, :k].T          # (k, d)
    whitening_scale  = 1.0 / np.sqrt(eigenvalues[:k] + EPS)  # (k,)

    return mean.astype(np.float32), components.astype(np.float32), whitening_scale.astype(np.float32)


def apply_transform(X: np.ndarray, mean, components, whitening_scale) -> np.ndarray:
    """Project, whiten and L2-normalise."""
    Xc      = X - mean
    X_proj  = Xc @ components.T          # (n, k)
    X_white = X_proj * whitening_scale   # (n, k)
    norms   = np.linalg.norm(X_white, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X_white / norms).astype(np.float32)


def apply_pca_to_descriptor(name: str, k: int):
    npz_path = os.path.join(INDEX_DIR, f"{name}.npz")
    if not os.path.exists(npz_path):
        print(f"  [{name}] SKIP — {npz_path} not found")
        return

    X = np.load(npz_path)["features"].astype(np.float32)
    n, d = X.shape
    print(f"  [{name}] {d}D -> {k}D  ({n} vectors)")

    mean, components, whitening_scale = fit_pca_whitening(X, k)
    X_reduced = apply_transform(X, mean, components, whitening_scale)

    # Overwrite descriptor index with reduced embeddings
    np.savez_compressed(npz_path, features=X_reduced)

    # Save PCA model for future query-time transforms
    pca_model_path = os.path.join(INDEX_DIR, f"pca_{name}.npz")
    np.savez_compressed(
        pca_model_path,
        mean=mean,
        components=components,
        whitening_scale=whitening_scale,
        original_dim=np.array(d),
        reduced_dim=np.array(k),
    )

    size_after = os.path.getsize(npz_path) / 1024**2
    print(f"    -> saved {size_after:.2f} MB  (pca model: {pca_model_path})")


def main():
    print(f"Index dir: {INDEX_DIR}\n")
    for name, k in RECOMMENDED_DIMS.items():
        apply_pca_to_descriptor(name, k)
    print("\nDone.")


if __name__ == "__main__":
    main()
