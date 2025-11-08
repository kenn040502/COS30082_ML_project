# extractor_dinov2.py
# -------------------------------------------
# DINOv2 ViT-B/14 reg4 feature extractor (plant-pretrained checkpoint)
# - Loads timm backbone with no classifier head (num_classes=0)
# - Cleans checkpoint keys (module./model., drops head/classifier/fc)
# - Deterministic setup for reproducibility
# - Provides get_transform(input_size=518) for ImageNet normalization
# - Returns (model, feat_dim) from load_dinov2_feature_extractor(...)
#
# Expected usage from build_features.py:
#   model, feat_dim = load_dinov2_feature_extractor(weights_path, device)
#   tf = get_transform(518)
#   feats = model(images_tensor)  # (B, feat_dim)
# -------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import random

import inspect
import argparse

import timm
from torchvision import transforms


# -----------------------------
# Determinism (harmless for extraction)
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------
# Transforms
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transform(input_size: int = 518) -> transforms.Compose:
    """
    ImageNet normalization with resize to the DINOv2 reg4 resolution (default 518).
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------
# Key cleaning for checkpoints
# -----------------------------
def _clean_state_dict(state: dict) -> dict:
    """
    Strip common prefixes and drop classifier/head tensors so we can load
    into a headless (num_classes=0) backbone cleanly.
    """
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        # Drop any classifier layers / heads
        if "head" in k or "classifier" in k or k.startswith("fc."):
            continue
        clean[k] = v
    return clean

def _smart_load_checkpoint(p, map_location="cpu"):
    """
    Loads checkpoints across PyTorch versions:
    - Try vanilla load (<=2.5 or simple 2.6 ckpts)
    - Try weights_only=True with allowlisted globals (2.6+ safe mode)
    - Fall back to weights_only=False (only if the file is trusted)
    """
    # 1) vanilla
    try:
        return torch.load(p, map_location=map_location)
    except Exception:
        pass

    # 2) allowlist argparse.Namespace then try weights_only=True
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])  # type: ignore[attr-defined]
    except Exception:
        pass

    supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    if supports_weights_only:
        try:
            return torch.load(p, map_location=map_location, weights_only=True)
        except Exception:
            # 3) final fallback – trusted source only
            return torch.load(p, map_location=map_location, weights_only=False)
    else:
        # older torch without weights_only kw already failed at (1)
        raise


# -----------------------------
# Wrapper (optional; timm with num_classes=0 already returns features)
# -----------------------------
class FeatureBackbone(nn.Module):
    """
    Thin wrapper to ensure forward() returns a 2D (B, D) feature tensor.
    """
    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.backbone = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For timm models created with num_classes=0, forward() returns global pooled features.
        feats = self.backbone(x)
        # Ensure flat (B, D)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return feats


# -----------------------------
# Loader
# -----------------------------
def load_dinov2_feature_extractor(
    weights_path: Path | str,
    device: Optional[str] = None,
    backbone: str = "vit_base_patch14_reg4_dinov2",
    input_size: int = 518,
) -> Tuple[nn.Module, int]:
    """
    Load a DINOv2 ViT-B/14 reg4 backbone as a feature extractor (no classifier head).
    Returns (model_on_device, feat_dim).
    """
    set_seed(42)

    weights_path = Path(weights_path)
    assert weights_path.exists(), f"Checkpoint not found: {weights_path}"

    # Pick device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DINOv2] device = {device}")
    print(f"[DINOv2] backbone = {backbone}")
    print(f"[DINOv2] input_size = {input_size}")
    print(f"[DINOv2] loading weights: {weights_path.name}")

    # Create a headless backbone (timm returns pooled features if num_classes=0)
    model = timm.create_model(backbone, pretrained=False, num_classes=0)
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None:
        # Fallback; ViT-B/14 usually 768
        feat_dim = 768
    print(f"[DINOv2] feature_dim = {feat_dim}")

    # Robust load across PyTorch versions / formats
    ckpt = _smart_load_checkpoint(weights_path, map_location="cpu")

    # Unwrap common layouts
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "module" in ckpt:
        state = ckpt["module"]
    elif isinstance(ckpt, dict):
        state = ckpt  # already a flat state dict
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Strip prefixes and drop classifier/head tensors
    state = _clean_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[DINOv2] loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        # Most missing keys should be classifier-related when we drop the head.
        # Print first few for visibility.
        print("[DINOv2] missing (first 10):", missing[:10])
    if len(unexpected) > 0:
        print("[DINOv2] unexpected (first 10):", unexpected[:10])

    # Wrap & move to device
    feat_model = FeatureBackbone(model).to(device)
    feat_model.eval()

    return feat_model, feat_dim


# -----------------------------
# Self-test (optional)
# -----------------------------
if __name__ == "__main__":
    # Example quick test (adjust the path)
    ckpt_path = Path("../weights/model_best.pth.tar")
    if not ckpt_path.exists():
        print(f"[SelfTest] Skipped: {ckpt_path} not found.")
        raise SystemExit(0)

    model, dim = load_dinov2_feature_extractor(ckpt_path)
    tf = get_transform(518)

    # Dummy batch (B=2) to verify forward works
    x = torch.randn(2, 3, 518, 518, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(x)
    print("[SelfTest] out shape:", tuple(out.shape))  # expect (2, dim)
