"""
Metric learning wrappers around timm backbones with partial fine-tuning.

Each model:
  - Loads pretrained timm backbone (num_classes=0 → feature extractor)
  - Freezes all layers except the last N blocks / layers
  - Adds a projection head → 512-dim L2-normalized embedding

Freeze strategies per architecture:
  MobileNetV2  → unfreeze last 3 inverted residual blocks + head
  ResNet50     → unfreeze layer3 + layer4
  ViT_base     → unfreeze last 4 transformer blocks + norm
  DinoV2       → unfreeze last 4 transformer blocks + norm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

EMBED_DIM = 512

MODEL_CONFIGS = {
    "mobilenetv2": {
        "timm_name": "mobilenetv2_100.ra_in1k",
        "img_size": 224,
        "feat_dim": 1280,
    },
    "resnet50": {
        "timm_name": "resnet50.a1_in1k",
        "img_size": 224,
        "feat_dim": 2048,
    },
    "vit_base": {
        "timm_name": "vit_base_patch16_224.augreg_in1k",
        "img_size": 224,
        "feat_dim": 768,
    },
    "dinov2": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "img_size": 518,
        "feat_dim": 384,
    },
}


def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def _apply_freeze_strategy(backbone: nn.Module, model_name: str):
    _freeze_all(backbone)

    if model_name == "mobilenetv2":
        # features is a Sequential of ~18 blocks; unfreeze last 3 + conv_head
        blocks = list(backbone.blocks)
        for block in blocks[-3:]:
            _unfreeze(block)
        _unfreeze(backbone.conv_head)

    elif model_name == "resnet50":
        _unfreeze(backbone.layer3)
        _unfreeze(backbone.layer4)

    elif model_name in ("vit_base", "dinov2"):
        blocks = list(backbone.blocks)
        for block in blocks[-4:]:
            _unfreeze(block)
        _unfreeze(backbone.norm)

    return backbone


class MetricModel(nn.Module):
    def __init__(self, model_name: str, img_size: int | None = None):
        super().__init__()
        cfg = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.img_size   = img_size or cfg["img_size"]

        # ViT-based models require img_size at creation time so timm can
        # interpolate the position embeddings to the new grid.
        is_vit = model_name in ("vit_base", "dinov2")
        backbone = timm.create_model(
            cfg["timm_name"], pretrained=True, num_classes=0,
            **({"img_size": self.img_size} if is_vit else {}),
        )
        self.backbone = _apply_freeze_strategy(backbone, model_name)

        feat_dim = cfg["feat_dim"]
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).float()
        emb = self.projection(feats)
        return F.normalize(emb, p=2, dim=1)  # L2-normalize → unit sphere

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_params())


def load_for_inference(model_name: str, weights_path: str) -> MetricModel:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    img_size = state.get("hyperparams", {}).get("img_size")
    model = MetricModel(model_name, img_size=img_size)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
