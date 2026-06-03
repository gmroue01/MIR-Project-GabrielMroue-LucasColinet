"""
SIFT + RANSAC reranking.

Given a query image and a pool of candidates (already retrieved by fast
embedding search), reorders them by geometrically-verified SIFT matches.

Pre-requisite: generate the index once with
    python scripts/generate_sift_index.py
which produces  indexes/sift_ransac.npz.
"""

import os
from typing import List, Optional

import cv2
import numpy as np

INDEX_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "indexes")
_NPZ_PATH      = os.path.join(INDEX_DIR, "sift_ransac.npz")
_PCA_PATH      = os.path.join(INDEX_DIR, "pca_sift_ransac.npz")

_sift_data: Optional[dict] = None
_pca_mode: Optional[bool] = None   # True when index contains PCA-reduced float16 descriptors
_bf = cv2.BFMatcher(cv2.NORM_L2)


def _load() -> Optional[dict]:
    global _sift_data, _pca_mode
    if _sift_data is None and os.path.exists(_NPZ_PATH):
        raw = np.load(_NPZ_PATH)
        _sift_data = {k: raw[k] for k in raw.files}
        # PCA-reduced index stores float16 descriptors; original stores uint8
        sample_key = next((k for k in raw.files if k.endswith("_des")), None)
        _pca_mode = (sample_key is not None and raw[sample_key].dtype == np.float16)
    return _sift_data


def is_available() -> bool:
    if _sift_data is not None:
        return True
    return os.path.exists(_NPZ_PATH)


def _inlier_count(q_kp, q_des, c_kp, c_des,
                  ratio: float = 0.75, ransac_thresh: float = 5.0) -> int:
    """RANSAC inliers between query and one candidate. Returns 0 on failure."""
    if q_des is None or c_des is None:
        return 0

    # PCA-reduced index: float16, already whitened+L2-normalised — cast only.
    # Legacy index: uint8 (÷2 at index time) → restore float32.
    if _pca_mode:
        qf = q_des.astype(np.float32)
        cf = c_des.astype(np.float32)
    else:
        qf = q_des.astype(np.float32) * 2.0
        cf = c_des.astype(np.float32) * 2.0

    if len(qf) < 4 or len(cf) < 4:
        return 0

    try:
        raw = _bf.knnMatch(qf, cf, k=2)
    except cv2.error:
        return 0

    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]

    if len(good) < 4:
        return len(good)

    src = np.float32([q_kp[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([c_kp[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        return int(mask.sum()) if mask is not None else 0
    except cv2.error:
        return 0


def rerank(query_filename: str, candidates: List[dict]) -> List[dict]:
    """
    Rerank candidates by SIFT-RANSAC inlier score vs the query.

    Parameters
    ----------
    query_filename : str  e.g. "0_0_BMW_X3_1.jpg"
    candidates     : list of result dicts with at least 'filename' and 'rank'

    Returns
    -------
    Same dicts reordered, with updated 'rank' and added 'sift_score' field.
    Falls back to original order if the SIFT index is unavailable.
    """
    data = _load()
    if data is None:
        return candidates

    q_kp  = data.get(f"{query_filename}_kp")
    q_des = data.get(f"{query_filename}_des")
    if q_kp is None or q_des is None:
        return candidates

    scores = []
    for r in candidates:
        fn    = r["filename"]
        c_kp  = data.get(f"{fn}_kp")
        c_des = data.get(f"{fn}_des")
        scores.append(_inlier_count(q_kp, q_des, c_kp, c_des))

    order    = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    reranked = []
    for new_rank, old_idx in enumerate(order, start=1):
        r = dict(candidates[old_idx])
        r["rank"]       = new_rank
        r["sift_score"] = int(scores[old_idx])
        reranked.append(r)

    return reranked
