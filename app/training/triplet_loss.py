"""
Online Hard Triplet Mining with margin-based triplet loss.

For each anchor in the batch:
- Hardest positive: same class, maximum distance
- Hardest negative: different class, minimum distance (but > 0)

Reference: "In Defense of the Triplet Loss" (Hermans et al., 2017)
"""
import torch
import torch.nn.functional as F


def pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute pairwise L2 distances for a batch of L2-normalized embeddings."""
    dot = embeddings @ embeddings.T
    # Clamp for numerical stability before sqrt
    sq_dist = torch.clamp(2.0 - 2.0 * dot, min=0.0)
    if squared:
        return sq_dist
    return torch.sqrt(sq_dist + 1e-12)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Online hard triplet mining loss.
    embeddings: (N, D) L2-normalized
    labels:     (N,)   int class indices
    Returns (loss, metrics_dict)
    """
    dist_mat = pairwise_distances(embeddings, squared=squared)  # (N, N)

    labels = labels.unsqueeze(1)  # (N, 1)
    same_class = labels == labels.T   # (N, N) bool
    diff_class = ~same_class

    # Mask out diagonal (anchor == anchor)
    eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
    same_class = same_class & ~eye

    # Hardest positive: max distance among same-class pairs
    # Set non-positive pairs to 0 before max
    pos_dist = dist_mat * same_class.float()
    hardest_pos, _ = pos_dist.max(dim=1)  # (N,)

    # Hardest negative: min distance among different-class pairs
    # Set same-class pairs to large value before min
    neg_dist = dist_mat + same_class.float() * 1e9
    hardest_neg, _ = neg_dist.min(dim=1)  # (N,)

    triplet_loss = F.relu(hardest_pos - hardest_neg + margin)

    # Only count anchors that have at least one valid positive
    valid_mask = same_class.any(dim=1)
    loss = triplet_loss[valid_mask].mean()

    # Fraction of violated triplets (loss > 0)
    n_violated = (triplet_loss[valid_mask] > 0).float().mean().item()

    metrics = {
        "loss": loss.item(),
        "avg_pos_dist": hardest_pos[valid_mask].mean().item(),
        "avg_neg_dist": hardest_neg[valid_mask].mean().item(),
        "fraction_violated": n_violated,
    }
    return loss, metrics
