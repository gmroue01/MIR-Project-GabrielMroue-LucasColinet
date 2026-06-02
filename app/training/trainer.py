"""
Training loop for metric learning.
Supports two loss modes:
  - "proxy_anchor" (default): Proxy-Anchor loss with class-balanced batches
  - "triplet": Online hard triplet mining (original)
"""
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.training.triplet_loss import batch_hard_triplet_loss
from app.training.proxy_anchor_loss import ProxyAnchorLoss
from app.training.models import MetricModel


def evaluate(
    model: MetricModel,
    loader: DataLoader,
    device: torch.device,
    loss_fn=None,
) -> dict:
    """Compute validation loss and Recall@1 (nearest-neighbour accuracy)."""
    model.eval()
    all_embs, all_labels = [], []

    with torch.no_grad():
        val_loss, val_steps = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embs = model(imgs)
            if loss_fn is not None:
                loss = loss_fn(embs, labels)
            else:
                loss, _ = batch_hard_triplet_loss(embs, labels)
            val_loss += loss.item()
            val_steps += 1
            all_embs.append(embs.cpu())
            all_labels.append(labels.cpu())

    all_embs   = torch.cat(all_embs)
    all_labels = torch.cat(all_labels)

    # Recall@1: nearest neighbour (excluding self) by cosine similarity
    sim = all_embs @ all_embs.T
    sim.fill_diagonal_(-1e9)
    nn_labels = all_labels[sim.argmax(dim=1)]
    recall_1  = (nn_labels == all_labels).float().mean().item()

    return {
        "val_loss": val_loss / max(val_steps, 1),
        "recall@1": recall_1,
    }


def train(
    model_name: str,
    dataset_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_min_factor: float = 0.01,   # eta_min = lr * lr_min_factor (cosine schedule floor)
    img_size: int | None = None,   # None → use model default from MODEL_CONFIGS
    margin: float = 0.3,
    device_str: str = "cuda",
    loss_type: str = "proxy_anchor",
    # Balanced sampler params (proxy_anchor only)
    n_classes_per_batch: int = 16,
    n_samples_per_class: int = 4,
    # Proxy-Anchor loss hyperparams
    proxy_alpha: float = 32.0,
    proxy_delta: float = 0.1,
    proxy_weight_decay: float = 0.0,
    # Early stopping
    patience: int = 5,
    # Memory optimisation
    mixed_precision: bool = False,   # float16 autocast + GradScaler (CUDA only)
    accumulation_steps: int = 1,     # >1 → accumulate gradients over N micro-batches
):
    from app.training.dataset import build_splits, CarDataset, ClassBalancedSampler, make_transforms
    from app.training.models import MODEL_CONFIGS

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    actual_size = img_size or MODEL_CONFIGS[model_name]["img_size"]
    print(f"\n{'='*60}")
    print(f"Training {model_name} | loss={loss_type} | img_size={actual_size} | device={device}")
    print(f"{'='*60}")

    # Dataset — share class_to_idx so val labels are consistent with train proxies
    train_paths, val_paths = build_splits(dataset_dir, val_ratio=0.2)
    t_tf = make_transforms(actual_size, augment=True)
    v_tf = make_transforms(actual_size, augment=False)

    train_ds = CarDataset(train_paths, transform=t_tf)
    val_ds   = CarDataset(val_paths,   transform=v_tf, class_to_idx=train_ds.class_to_idx)

    if loss_type == "proxy_anchor":
        sampler = ClassBalancedSampler(train_ds, n_classes_per_batch, n_samples_per_class)
        effective_batch = n_classes_per_batch * n_samples_per_class
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)
        print(f"Balanced batches: {n_classes_per_batch} classes × {n_samples_per_class} samples = {effective_batch}/batch")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Classes: {train_ds.num_classes}")

    # Model
    model = MetricModel(model_name, img_size=actual_size).to(device)
    print(f"Trainable params: {model.count_trainable():,}")

    # Loss
    loss_fn = None
    if loss_type == "proxy_anchor":
        from app.training.models import EMBED_DIM
        loss_fn = ProxyAnchorLoss(
            train_ds.num_classes, EMBED_DIM,
            alpha=proxy_alpha, delta=proxy_delta,
            proxy_weight_decay=proxy_weight_decay,
        ).to(device)
        proxy_lr = lr * 100  # proxies converge faster than the backbone
        optimizer = AdamW([
            {"params": model.trainable_params(), "lr": lr,       "weight_decay": weight_decay},
            {"params": loss_fn.parameters(),     "lr": proxy_lr, "weight_decay": loss_fn.proxy_weight_decay},
        ])
    else:
        optimizer = AdamW(model.trainable_params(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * lr_min_factor)

    use_amp = mixed_precision and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision  : ON (float16)")
    if accumulation_steps > 1:
        print(f"Grad accumulation: {accumulation_steps} micro-batches "
              f"(effective batch ×{accumulation_steps})")

    history         = []
    best_recall     = 0.0
    patience_counter = 0
    best_path       = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        if loss_fn is not None:
            loss_fn.train()

        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                embs = model(imgs)
                if loss_type == "proxy_anchor":
                    loss = loss_fn(embs, labels)
                else:
                    loss, _ = batch_hard_triplet_loss(embs, labels, margin=margin)

            epoch_loss += loss.item()
            # Divide before backward so accumulated gradient equals the true mean
            scaler.scale(loss / accumulation_steps).backward()

            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip_params = list(model.trainable_params())
                if loss_fn is not None:
                    clip_params += list(loss_fn.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        n_batches  = len(train_loader)
        train_loss = epoch_loss / n_batches

        eval_loss_fn = loss_fn if loss_type == "proxy_anchor" else None
        val_metrics  = evaluate(model, val_loader, device, loss_fn=eval_loss_fn)
        elapsed = time.time() - t0

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_metrics["val_loss"], 4),
            "recall@1":   round(val_metrics["recall@1"], 4),
            "lr":         round(scheduler.get_last_lr()[0], 6),
            "time_s":     round(elapsed, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_metrics['val_loss']:.4f} | "
            f"R@1: {val_metrics['recall@1']:.3f} | "
            f"{elapsed:.0f}s"
        )

        if val_metrics["recall@1"] > best_recall:
            best_recall      = val_metrics["recall@1"]
            patience_counter = 0
            ckpt = {
                "epoch":            epoch,
                "model_name":       model_name,
                "loss_type":        loss_type,
                "model_state_dict": model.state_dict(),
                "val_metrics":      val_metrics,
                "history":          history,
                "hyperparams": {
                    "lr": lr, "weight_decay": weight_decay,
                    "lr_min_factor": lr_min_factor, "patience": patience,
                    "img_size": actual_size,
                    "mixed_precision": use_amp,
                    "accumulation_steps": accumulation_steps,
                },
            }
            if loss_fn is not None:
                ckpt["proxy_state_dict"] = loss_fn.state_dict()
            torch.save(ckpt, best_path)
            print(f"  → Saved best model (R@1={best_recall:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping: no R@1 improvement for {patience} epochs.")
                break

    hist_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Recall@1: {best_recall:.3f} | Saved to {best_path}")
    return history, best_recall
