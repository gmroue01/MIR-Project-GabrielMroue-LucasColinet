"""
Similarity/distance measures in two forms:
- pairwise(a, b): scalar distance between two 1-D vectors
- batch(query, matrix): vectorized 1-D array of distances from query to every row in matrix
"""
import numpy as np
from scipy.spatial.distance import cosine


# ---------- Pairwise (scalar) ----------

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine(a, b))


def chi_square(a: np.ndarray, b: np.ndarray) -> float:
    num = (a - b) ** 2
    den = a + b
    mask = den > 0
    return float(0.5 * np.sum(num[mask] / den[mask]))


def jensen_shannon(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(a - a.min(), 0, None)
    b = np.clip(b - b.min(), 0, None)
    sa, sb = a.sum(), b.sum()
    p = a / sa if sa > 0 else np.ones_like(a) / len(a)
    q = b / sb if sb > 0 else np.ones_like(b) / len(b)
    m = 0.5 * (p + q)
    eps = 1e-12
    js = 0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + \
         0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
    return float(np.clip(js, 0.0, 1.0))


# ---------- Batch (vectorized: query vs full matrix) ----------

def euclidean_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # For L2-normalized vectors: ||a-b||² = 2 - 2·(a·b)  →  avoids large diff matrix
    dots = matrix.dot(query)
    sq = np.clip(2.0 - 2.0 * dots, 0.0, None)
    return np.sqrt(sq)


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = q_norm * m_norms
    denom = np.where(denom == 0, 1e-12, denom)
    sims = matrix.dot(query) / denom
    return 1.0 - np.clip(sims, -1.0, 1.0)


def chi_square_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    num = (matrix - query) ** 2
    den = matrix + query
    safe = np.where(den > 0, num / den, 0.0)
    return 0.5 * safe.sum(axis=1)


def jensen_shannon_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Shift to non-negative
    q = np.clip(query - query.min(), 0, None)
    M = np.clip(matrix - matrix.min(axis=1, keepdims=True), 0, None)

    sq = q.sum()
    p = q / sq if sq > 0 else np.ones_like(q) / len(q)

    sm = M.sum(axis=1, keepdims=True)
    sm = np.where(sm == 0, 1.0, sm)
    Q = M / sm  # (N, D)

    mix = 0.5 * (p + Q)  # (N, D)
    eps = 1e-12

    kl_p = np.sum(p * np.log((p + eps) / (mix + eps)), axis=1)
    kl_q = np.sum(Q * np.log((Q + eps) / (mix + eps)), axis=1)
    return np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)


# ---------- Dispatch ----------

MEASURES = {
    "euclidean": euclidean,
    "cosine":    cosine_distance,
}

BATCH_MEASURES = {
    "euclidean": euclidean_batch,
    "cosine":    cosine_batch,
}
