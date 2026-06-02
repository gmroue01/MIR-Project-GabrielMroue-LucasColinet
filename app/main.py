import logging
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional

from app.auth import (
    auth_enabled, verify_password, create_token, decode_token,
    check_rate_limit, record_failure,
)

from app.searcher import search, search_with_reranking, get_indexing_metrics, get_all_images, _load_meta
from app.evaluation.metrics import compute_all
from app.similarity.measures import MEASURES
from app.config import DATASET_DIR, STATIC_DIR
from app.clip_searcher import _searcher as clip_searcher

log = logging.getLogger(__name__)

app = FastAPI(title="MIR - Image Search Engine")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS env var in production (comma-separated list of origins).
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Body size limit (1 MB) ────────────────────────────────────────────────────
_MAX_BODY = 1 * 1024 * 1024

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > _MAX_BODY:
        return JSONResponse({"detail": "Request body too large (max 1 MB)"}, status_code=413)
    return await call_next(request)

# ── Authentication ────────────────────────────────────────────────────────────
_OPEN_PATHS = {"/api/health", "/api/auth/login"}
# Static mounts (/images/, /flickr8k/) are also protected when auth is enabled.
_AUTH_PREFIXES = ("/api/", "/images/", "/flickr8k/")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if not auth_enabled():
        return await call_next(request)
    if path in _OPEN_PATHS:
        return await call_next(request)
    if not any(path.startswith(p) for p in _AUTH_PREFIXES):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or not decode_token(auth[7:]):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)

# Serve dataset images (cars)
if os.path.isdir(DATASET_DIR):
    app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")

# Serve Flickr8K images
_flickr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Flickr8K", "Images")
if os.path.isdir(_flickr_dir):
    app.mount("/flickr8k", StaticFiles(directory=_flickr_dir), name="flickr8k")


# ---------- Schemas ----------

class SearchRequest(BaseModel):
    query_index: int
    descriptors: List[str]
    measure: str
    top_k: int = 50
    use_reranking: bool = False
    pool_percent: int = Field(25, ge=1, le=100)


# ---------- Routes ----------

@app.get("/api/images")
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    class_filter: Optional[str] = None,
):
    images = get_all_images()
    if class_filter:
        images = [img for img in images if class_filter.lower() in img["class"].lower()]
    total = len(images)
    start = (page - 1) * page_size
    end = start + page_size
    return {"total": total, "page": page, "page_size": page_size, "images": images[start:end]}


@app.get("/api/classes")
def list_classes():
    _, classes = _load_meta()
    unique = sorted(set(classes.tolist()))
    counts = {c: int(np.sum(classes == c)) for c in unique}
    return {"classes": unique, "counts": counts}


@app.get("/api/descriptors")
def list_descriptors():
    return {"descriptors": [
        "color_histogram",
        "mobilenet_arcface", "mobilenet_zeroshot",
        "resnet50_zeroshot",
        "vit_b16_zeroshot",
        "dinov2_supcon", "dinov2_zeroshot",
        "sift",
    ]}


@app.get("/api/measures")
def list_measures():
    return {"measures": list(MEASURES.keys())}


@app.post("/api/search")
def search_endpoint(req: SearchRequest):
    valid_descriptors = {
        "color_histogram",
        "mobilenet_arcface", "mobilenet_zeroshot",
        "resnet50_zeroshot",
        "vit_b16_zeroshot",
        "dinov2_supcon", "dinov2_zeroshot",
        "sift",
    }
    for d in req.descriptors:
        if d not in valid_descriptors:
            raise HTTPException(400, f"Unknown descriptor: {d}")
    if req.measure not in MEASURES:
        raise HTTPException(400, f"Unknown measure: {req.measure}")
    if req.top_k not in (20, 50):
        raise HTTPException(400, "top_k must be 20 or 50.")

    filenames, classes = _load_meta()
    if req.query_index < 0 or req.query_index >= len(filenames):
        raise HTTPException(400, "Invalid query_index.")

    query_class = str(classes[req.query_index])
    total_relevant = int(np.sum(classes == query_class)) - 1

    from app.descriptors.sift_ransac import is_available as sift_available
    if req.use_reranking and not sift_available():
        raise HTTPException(503, "Index SIFT non disponible. Lancez : python scripts/generate_sift_index.py")

    t0 = time.perf_counter()
    if req.use_reranking:
        results_top50 = search_with_reranking(
            req.query_index, req.descriptors, req.measure,
            top_k=50, pool_percent=req.pool_percent,
        )
    else:
        results_top50 = search(req.query_index, req.descriptors, req.measure, top_k=50)
    search_time = time.perf_counter() - t0

    results_top20 = results_top50[:20]

    metrics_20 = compute_all(results_top20, query_class, total_relevant, k_values=[20])
    metrics_50 = compute_all(results_top50, query_class, total_relevant, k_values=[50])

    return {
        "query_filename": str(filenames[req.query_index]),
        "query_class": query_class,
        "results_top20": results_top20,
        "results_top50": results_top50,
        "metrics": {
            "top20": {
                "precision": metrics_20.get("precision@20", 0),
                "recall": metrics_20.get("recall@20", 0),
                "average_precision": metrics_20["average_precision"],
                "r_precision": metrics_20["r_precision"],
            },
            "top50": {
                "precision": metrics_50.get("precision@50", 0),
                "recall": metrics_50.get("recall@50", 0),
                "average_precision": metrics_50["average_precision"],
                "r_precision": metrics_50["r_precision"],
            },
            "total_relevant_in_db": total_relevant,
        },
        "search_time_s": round(search_time, 4),
        "reranked": req.use_reranking,
    }


class MapRequest(BaseModel):
    descriptors: List[str]
    measure: str
    top_k: int = 50
    max_queries: int = 100  # cap to keep response time reasonable


@app.post("/api/map")
def compute_map(req: MapRequest):
    """
    Compute Mean Average Precision over a sample of queries (one per class).
    Returns per-class AP and overall MAP.
    """
    valid_descriptors = {
        "color_histogram",
        "mobilenet_arcface", "mobilenet_zeroshot",
        "resnet50_zeroshot",
        "vit_b16_zeroshot",
        "dinov2_supcon", "dinov2_zeroshot",
        "sift",
    }
    for d in req.descriptors:
        if d not in valid_descriptors:
            raise HTTPException(400, f"Unknown descriptor: {d}")
    if req.measure not in MEASURES:
        raise HTTPException(400, f"Unknown measure: {req.measure}")
    if req.top_k not in (20, 50, 100):
        raise HTTPException(400, "top_k must be 20, 50 or 100.")

    filenames, classes = _load_meta()
    unique_classes = sorted(set(classes.tolist()))

    # Pick one representative query per class (first image)
    class_to_idx = {}
    for i, c in enumerate(classes.tolist()):
        if c not in class_to_idx:
            class_to_idx[c] = i

    query_indices = [class_to_idx[c] for c in unique_classes]
    if req.max_queries and len(query_indices) > req.max_queries:
        query_indices = query_indices[:req.max_queries]
        unique_classes = unique_classes[:req.max_queries]

    t0 = time.perf_counter()
    ap_per_class = {}

    for qidx, qclass in zip(query_indices, unique_classes):
        total_relevant = int(np.sum(classes == qclass)) - 1
        results = search(qidx, req.descriptors, req.measure, top_k=req.top_k)
        m = compute_all(results, qclass, total_relevant, k_values=[req.top_k])
        ap_per_class[qclass] = round(m["average_precision"], 4)

    map_score = float(np.mean(list(ap_per_class.values())))
    elapsed = time.perf_counter() - t0

    return {
        "map": round(map_score, 4),
        "num_queries": len(query_indices),
        "ap_per_class": ap_per_class,
        "top_k": req.top_k,
        "elapsed_s": round(elapsed, 2),
    }


@app.get("/api/indexing-metrics")
def indexing_metrics():
    m = get_indexing_metrics()
    if not m:
        raise HTTPException(503, "Indexes not built yet. Run the indexer first.")
    return m


@app.get("/api/health")
def health():
    return {"status": "ok"}


class LoginRequest(BaseModel):
    password: str

@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(429, "Too many failed attempts. Please wait 5 minutes.")
    if not verify_password(body.password):
        record_failure(ip)
        raise HTTPException(401, "Mot de passe incorrect")
    return {"token": create_token()}



# ── CLIP / Flickr8K endpoints ──────────────────────────────────────────────

class ClipTextRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = 10

class ClipImageRequest(BaseModel):
    image_idx: int
    top_k: int = 10

class ClipEvalRequest(BaseModel):
    image_indices: List[int] = []
    text_queries:  List[str] = []
    top_k:         int = 10


@app.get("/api/clip/images")
def clip_images(
    page:      int = Query(1,  ge=1),
    page_size: int = Query(30, ge=1, le=200),
):
    try:
        return clip_searcher.get_images_page(page, page_size)
    except Exception as e:
        log.exception("clip_images failed")
        raise HTTPException(503, "Service temporarily unavailable")


@app.post("/api/clip/text-to-image")
def clip_text_to_image(req: ClipTextRequest):
    if req.top_k not in (5, 10, 20):
        raise HTTPException(400, "top_k must be 5, 10 or 20.")
    try:
        results = clip_searcher.text_to_image(req.query, req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        log.exception("clip_text_to_image failed")
        raise HTTPException(503, "Service temporarily unavailable")


@app.post("/api/clip/image-to-text")
def clip_image_to_text(req: ClipImageRequest):
    if req.top_k not in (5, 10, 20):
        raise HTTPException(400, "top_k must be 5, 10 or 20.")
    try:
        results = clip_searcher.image_to_text(req.image_idx, req.top_k)
        return {"image_idx": req.image_idx, "results": results}
    except Exception as e:
        log.exception("clip_image_to_text failed")
        raise HTTPException(503, "Service temporarily unavailable")


@app.post("/api/clip/evaluate")
def clip_evaluate(req: ClipEvalRequest):
    if not req.image_indices and not req.text_queries:
        raise HTTPException(400, "Provide at least one image index or text query.")
    if req.top_k not in (5, 10, 20):
        raise HTTPException(400, "top_k must be 5, 10 or 20.")
    try:
        return clip_searcher.evaluate(req.image_indices, req.text_queries, req.top_k)
    except Exception as e:
        log.exception("clip_evaluate failed")
        raise HTTPException(503, "Service temporarily unavailable")


# Explicit assets route (must be before the catch-all)
@app.get("/assets/{path:path}")
def serve_asset(path: str):
    assets_dir = os.path.realpath(os.path.join(STATIC_DIR, "assets"))
    asset      = os.path.realpath(os.path.join(assets_dir, path))
    # Prevent path traversal outside the assets directory
    if not asset.startswith(assets_dir + os.sep):
        raise HTTPException(404, "Not found")
    if os.path.isfile(asset):
        return FileResponse(asset)
    raise HTTPException(404, "Asset not found")


# SPA catch-all
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}
