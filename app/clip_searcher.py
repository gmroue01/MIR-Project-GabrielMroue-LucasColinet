"""
CLIP cross-modal search engine.
Uses OpenAI ViT-B/32 weights (via open_clip) to match the embeddings
generated in the training notebook with clip.load("ViT-B/32").

Index layout:
  index_images.faiss   — 8 091 vectors, one per image, ordered like df['image'].unique()
  index_captions.faiss — 40 455 vectors, one per caption row, ordered like df.iloc[i]
"""
import os
import numpy as np
import pandas as pd
import torch
import faiss
import open_clip
from typing import List, Dict, Any

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLICKR_DIR = os.path.join(BASE_DIR, "Flickr8k")
FAISS_DIR  = os.path.join(BASE_DIR, "indexes_faiss")


class CLIPSearcher:
    def __init__(self):
        self._data_ready  = False   # FAISS + CSV loaded (fast)
        self._model_ready = False   # CLIP model loaded (slow, only for text queries)
        self.index_imgs   = None
        self.index_caps   = None
        self.unique_imgs  = None
        self.df           = None
        self.model        = None
        self.tokenizer    = None
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Lazy loading (two levels) ─────────────────────────────────────────

    def _load_data(self):
        """Load FAISS indexes + CSV. Fast (~1s). Required by all endpoints."""
        if self._data_ready:
            return
        self.index_imgs  = faiss.read_index(os.path.join(FAISS_DIR, "index_images.faiss"))
        self.index_caps  = faiss.read_index(os.path.join(FAISS_DIR, "index_captions.faiss"))
        self.df          = pd.read_csv(os.path.join(FLICKR_DIR, "captions.txt"))
        self.unique_imgs = self.df["image"].unique()
        self._data_ready = True

    def _load_model(self):
        """Load CLIP ViT-B/32. Slow on first call (~10s, downloads weights once)."""
        if self._model_ready:
            return
        self._load_data()
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        # fp16 pour réduire l'empreinte mémoire (~175 MB vs ~350 MB)
        self.model        = self.model.to(self.device).half().eval()
        self.tokenizer    = open_clip.get_tokenizer("ViT-B-32")
        self._model_ready = True

    # ── Internal helpers ──────────────────────────────────────────────────

    def _encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            vec    = self.model.encode_text(tokens)
            vec    = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().astype(np.float32)

    def _img_vec(self, image_idx: int) -> np.ndarray:
        """Reconstruct stored image embedding from FAISS index."""
        return self.index_imgs.reconstruct(image_idx).reshape(1, -1)

    @staticmethod
    def _ap_precision_recall(hits: List[int], n_relevant: int, top_k: int):
        if n_relevant == 0:
            return 0.0, 0.0, 0.0
        p_at_k = sum(hits) / top_k
        r_at_k = sum(hits) / n_relevant
        ap, running = 0.0, 0
        for i, h in enumerate(hits, start=1):
            if h:
                running += 1
                ap += running / i
        ap /= n_relevant
        return float(ap), float(p_at_k), float(r_at_k)

    # ── Public API ────────────────────────────────────────────────────────

    def get_images_page(self, page: int, page_size: int) -> Dict:
        self._load_data()
        total = len(self.unique_imgs)
        start = (page - 1) * page_size
        end   = min(start + page_size, total)
        return {
            "total":     total,
            "page":      page,
            "page_size": page_size,
            "images":    [
                {"index": i, "filename": str(self.unique_imgs[i])}
                for i in range(start, end)
            ],
        }

    def text_to_image(self, query: str, top_k: int = 10) -> List[Dict]:
        self._load_model()
        vec             = self._encode_text(query)
        scores, indices = self.index_imgs.search(vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            results.append({
                "rank":     rank,
                "filename": str(self.unique_imgs[idx]),
                "score":    float(score),
                "index":    int(idx),
            })
        return results

    def image_to_text(self, image_idx: int, top_k: int = 10) -> List[Dict]:
        self._load_data()
        vec             = self._img_vec(image_idx)
        scores, indices = self.index_caps.search(vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = self.df.iloc[int(idx)]
            results.append({
                "rank":     rank,
                "caption":  str(row["caption"]),
                "filename": str(row["image"]),
                "score":    float(score),
                "index":    int(idx),
            })
        return results

    def evaluate(
        self,
        image_indices: List[int],
        text_queries:  List[str],
        top_k:         int = 10,
    ) -> Dict:
        self._load_data()
        out = {}

        # ── Text → Image ──────────────────────────────────────────────────
        if text_queries:
            # Map each known caption text to its source image filename
            cap_to_img = dict(
                zip(self.df["caption"].str.strip(), self.df["image"])
            )
            t2i = []
            for text in text_queries:
                results   = self.text_to_image(text, top_k)
                gt_img    = cap_to_img.get(text.strip())
                hits      = [1 if r["filename"] == gt_img else 0 for r in results]
                ap, p, r  = self._ap_precision_recall(hits, 1 if gt_img else 0, top_k)
                t2i.append({"text": text, "precision_at_k": p, "recall_at_k": r, "ap": ap})
            out["text_to_image"] = {
                "map":     float(np.mean([q["ap"] for q in t2i])),
                "queries": t2i,
            }

        # ── Image → Text ──────────────────────────────────────────────────
        if image_indices:
            i2t = []
            for img_idx in image_indices:
                img_name   = str(self.unique_imgs[img_idx])
                results    = self.image_to_text(img_idx, top_k)
                gt_caps    = set(
                    self.df[self.df["image"] == img_name]["caption"].str.strip().tolist()
                )
                hits       = [1 if r["caption"].strip() in gt_caps else 0 for r in results]
                ap, p, r   = self._ap_precision_recall(hits, len(gt_caps), top_k)
                i2t.append({
                    "filename":      img_name,
                    "index":         img_idx,
                    "precision_at_k": p,
                    "recall_at_k":   r,
                    "ap":            ap,
                })
            out["image_to_text"] = {
                "map":     float(np.mean([q["ap"] for q in i2t])),
                "queries": i2t,
            }

        return out


# Singleton — shared across all requests
_searcher = CLIPSearcher()
