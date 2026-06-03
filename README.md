# MIR — Content-Based Image Retrieval Engine

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack **Content-Based Image Retrieval (CBIR)** system that lets you find visually similar car images using a range of classical and deep learning descriptors — with SIFT-RANSAC geometric reranking, PCA dimensionality reduction, and CLIP-based multimodal search.

> Academic project — M1 Computer Vision, 2025–2026

---

## Features

- **8 descriptors** — from Color Histogram and SIFT to fine-tuned DINOv2, MobileNet ArcFace, ResNet50, and ViT-B/16
- **2 similarity measures** — Euclidean, Cosine
- **PCA + whitening** — dimensionality reduction applied to all deep learning indexes (up to 157× compression)
- **SIFT-RANSAC reranking** — geometric verification to boost precision on top results
- **Evaluation suite** — per-query Precision, Recall, Average Precision, R-Precision, and MAP@20/50/100 over 46 classes
- **CLIP multimodal search** — text-to-image and image-to-text retrieval on the Flickr8K dataset (8,091 images)
- **Benchmark page** — compare all descriptors side by side on indexing time, index size, and search latency

---

## Dataset

The main search corpus is a curated car dataset:

| Property | Value |
|---|---|
| Images | 5,000 |
| Classes | 46 car models |
| Brands | BMW · Ford · Hyundai · Opel · Volkswagen |
| Images per brand | 1,000 |

<details>
<summary>Class breakdown</summary>

| Brand | Models | Images |
|---|---|---|
| BMW | X3, X5, X6, i3, i8, Serie1, Serie3, Serie3Berline, Serie5 | 1 000 |
| Ford | EcoSport, Fiesta, Galaxy, GT, Kuga, Puma, S-Max, Transit | 1 000 |
| Hyundai | i10, i20, i30, i40, Ioniq5, Kona, Nexo, Santa Fe, Tucson, Veloster | 1 000 |
| Opel | Astra, Corsa, CrosslandX, Grandland, Insignia, Mokka, Vivaro, Zafira + 2 more | 1 000 |
| Volkswagen | Golf, GolfVariant, Passat, Polo, Sharan, T-Cross, T-Roc, Tiguan, Touareg, Up | 1 000 |

</details>

---

## Descriptors

| Descriptor | Type | Dim (stored) | Original dim | Search latency |
|---|---|---|---|---|
| Color Histogram | Classical | 24 | 24 | 0.03 ms |
| SIFT (BoVW) | Classical | 128 | 128 | 0.10 ms |
| DINOv2 SupCon | Deep (fine-tuned) | **13** | 256 | 0.03 ms |
| DINOv2 ZeroShot | Deep | **13** | 256 | 0.03 ms |
| MobileNet ArcFace | Deep (fine-tuned) | **49** | 256 | 0.06 ms |
| MobileNet ZeroShot | Deep | **257** | 1 280 | 0.11 ms |
| ResNet50 ZeroShot | Deep | **256** | 2 048 | 0.10 ms |
| ViT-B/16 ZeroShot | Deep | **230** | 768 | 0.11 ms |

All deep learning indexes are compressed with **PCA + whitening + L2-normalization** (95% variance threshold).

---

## Project Structure

```
MIR_Project/
│
├── app/                        # FastAPI backend
│   ├── main.py                 # API routes
│   ├── searcher.py             # Core search engine (cosine / brute-force)
│   ├── indexer.py              # Index loading and caching
│   ├── pca_reducer.py          # PCA transform for query vectors
│   ├── clip_searcher.py        # CLIP multimodal search engine
│   ├── auth.py                 # JWT authentication
│   ├── config.py               # Paths configuration
│   ├── descriptors/            # Feature extraction (color, SIFT, DL models)
│   ├── similarity/             # Distance measures
│   ├── evaluation/             # Precision, Recall, AP, MAP
│   └── training/               # Training utilities (loss functions, trainer)
│
├── frontend/                   # React 18 + Vite frontend
│   └── src/
│       ├── components/         # Search, Benchmark, CLIP, Auth pages
│       └── api.js              # Axios API client
│
├── indexes/                    # Pre-computed descriptor indexes
│   ├── *.npz                   # Feature matrices (one per descriptor)
│   ├── pca_*.npz               # PCA models (mean, components, whitening scale)
│   ├── filenames.npy           # Ordered image filenames
│   ├── classes.npy             # Class labels
│   └── metrics.json            # Benchmark metrics (timing, size, dims)
│
├── indexes_faiss/              # FAISS indexes for CLIP search
│   ├── index_images.faiss
│   └── index_captions.faiss
│
├── Flickr8k/                   # Flickr8K dataset (CLIP)
│   ├── captions.txt
│   └── Images/
│
├── notebook/                   # Training notebooks (Google Colab)
│   ├── MIR_DinoV2_training.ipynb
│   ├── MIR_MobileNetV2_ArcFace_training.ipynb
│   ├── MIR_ResNet50_training.ipynb
│   ├── MIR_Vitbase16_training.ipynb
│   └── MIR_SIFT_reranking.ipynb
│
├── scripts/                    # Utility scripts
│   ├── compute_metrics.py      # Full re-indexing pipeline
│   ├── apply_pca.py            # PCA reduction on DL indexes
│   ├── apply_pca_sift.py       # PCA(32D) reduction on SIFT-RANSAC index
│   ├── generate_sift_index.py  # Build SIFT-RANSAC index (runs apply_pca_sift automatically)
│   └── generate_faiss_index.py # Build FAISS indexes for CLIP search
│
├── modelsV2/                   # Fine-tuned .pth weights
├── dataset/                    # Car images — 5 000 .jpg
│
├── requirements.txt
├── run.py                      # App entry point
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- The `dataset/` folder (5,000 car images)

### 1. Clone and install

```bash
git clone https://github.com/gmroue01/MIR_Project.git
cd MIR_Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Place the dataset

```
MIR_Project/
└── dataset/
    ├── Acura_Integra_Type_R_2001_0001.jpg
    └── ...
```

Pre-computed indexes (`indexes/`) are already included — **no re-indexing needed**.

### 3. Run

```bash
python run.py
# → http://localhost:8000
```

### Authentication (optional)

By default the app runs without authentication. To enable it:

```bash
# Windows (PowerShell)
$env:APP_PASSWORD = "yourpassword"
$env:JWT_SECRET   = "a-long-random-string"

# macOS / Linux
export APP_PASSWORD="yourpassword"
export JWT_SECRET="a-long-random-string"
```

### Docker

```bash
docker-compose up --build
# → http://localhost:8000
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/compute_metrics.py` | Full re-indexing from scratch (classical + DL + PCA) and update `metrics.json` |
| `scripts/apply_pca.py` | Apply PCA reduction to existing DL `.npz` indexes without reloading `.pth` weights |
| `scripts/apply_pca_sift.py` | Apply PCA(32D) to the SIFT-RANSAC index — 27% size reduction, +32% RANSAC inliers |
| `scripts/generate_sift_index.py` | Build the `sift_ransac.npz` index and automatically apply PCA(32D) (~146 MB, required for geometric reranking) |
| `scripts/generate_faiss_index.py` | Build the FAISS indexes for CLIP cross-modal search (`index_images.faiss`, `index_captions.faiss`) |

---

## Training Notebooks

All models were trained on **Google Colab (GPU)**. The pre-trained weights are stored in `modelsV2/`.

| Notebook | Architecture | Training strategy |
|---|---|---|
| `MIR_DinoV2_training.ipynb` | DINOv2 ViT-S/14 | SupCon loss + ZeroShot |
| `MIR_MobileNetV2_ArcFace_training.ipynb` | MobileNetV2 | ArcFace loss |
| `MIR_ResNet50_training.ipynb` | ResNet-50 | Supervised fine-tuning |
| `MIR_Vitbase16_training.ipynb` | ViT-B/16 | Supervised fine-tuning |
| `MIR_SIFT_reranking.ipynb` | SIFT + RANSAC | Geometric verification prototyping |

---

## Tech Stack

**Backend** — FastAPI · Uvicorn · NumPy · OpenCV · PyTorch · Timm · FAISS · open_clip · scikit-learn

**Frontend** — React 18 · Vite · Axios

**Deployment** — Docker · Railway

---

## Documentation

- [`USER_GUIDE.md`](USER_GUIDE.md) — step-by-step user manual (in French)
- [`DOCKER.md`](DOCKER.md) — containerization and deployment notes

---

## License

MIT — see [`LICENSE`](LICENSE)
