"""
Proxy-Anchor Loss for metric learning.
Reference: "Proxy Anchor Loss for Deep Metric Learning" (Kim et al., CVPR 2020)

One learnable proxy per class. For each batch:
  - Positive term: pull embeddings toward their class proxy
  - Negative term: push embeddings away from all other proxies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        alpha: float = 32.0,
        delta: float = 0.1,
        proxy_weight_decay: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.delta = delta
        # Proxies are L2-normalised in forward() so weight decay fights that normalisation;
        # keep at 0 unless you want to experiment with regularised proxies.
        self.proxy_weight_decay = proxy_weight_decay
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        P = F.normalize(self.proxies, p=2, dim=1)  # (C, D)
        cos_sim = embeddings @ P.T                  # (N, C)

        labels = labels.long()
        pos_mask = torch.zeros(len(labels), self.num_classes, dtype=torch.bool, device=embeddings.device)
        pos_mask.scatter_(1, labels.unsqueeze(1), True)
        neg_mask = ~pos_mask

        pos_exp = torch.exp(-self.alpha * (cos_sim - self.delta))
        neg_exp = torch.exp( self.alpha * (cos_sim + self.delta))

        pos_term = (pos_mask.float() * pos_exp).sum(dim=0)  # (C,)
        neg_term = (neg_mask.float() * neg_exp).sum(dim=0)  # (C,)

        # Only average over proxies that have at least one positive in the batch
        with_pos = pos_mask.any(dim=0)
        pos_loss = torch.log(1 + pos_term)[with_pos].mean()
        neg_loss = torch.log(1 + neg_term).mean()

        return pos_loss + neg_loss
