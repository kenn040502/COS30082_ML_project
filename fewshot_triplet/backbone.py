# fewshot_triplet/backbone.py
from __future__ import annotations
import timm, torch
import torch.nn.functional as F

def load_backbone(model_name: str, device: torch.device):
    print(f"🧠 Loading backbone via timm: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # features only
    model.eval().to(device)

    # --- Force transforms to the model's native image size (e.g., 518 for DINOv2 reg4) ---
    # Try to read from the patch embed (most reliable), else default_cfg.
    target_hw = None
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        s = model.patch_embed.img_size
        if isinstance(s, (tuple, list)): target_hw = (int(s[0]), int(s[1]))
        else:                            target_hw = (int(s), int(s))
    if target_hw is None:
        # fallback to default_cfg
        inp = model.default_cfg.get("input_size", (3, 224, 224))
        target_hw = (int(inp[1]), int(inp[2]))

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    cfg = resolve_data_config(model=model)
    # Build transform explicitly with the target size
    preprocess = create_transform(
        input_size=(3, target_hw[0], target_hw[1]),
        is_training=False,
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=cfg.get("mean", (0.5, 0.5, 0.5)),
        std=cfg.get("std", (0.5, 0.5, 0.5)),
        crop_pct=1.0,            # center-crop with 1.0 keeps full image after resize
    )
    print(f"🖼️ Preprocess input size set to: {target_hw[0]}×{target_hw[1]}")
    return model, preprocess

@torch.no_grad()
def encode_images(model, device, images: torch.Tensor):
    feats = model(images.to(device))
    if feats.dim() > 2:
        feats = feats.mean(dim=tuple(range(2, feats.dim())))
    return F.normalize(feats, dim=-1)

def estimate_out_dim(model, device) -> int:
    # Use the model-native size for the probe too (so shapes line up)
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        s = model.patch_embed.img_size
        if isinstance(s, (tuple, list)): H, W = int(s[0]), int(s[1])
        else:                            H, W = int(s), int(s)
    else:
        inp = model.default_cfg.get("input_size", (3, 224, 224))
        H, W = int(inp[1]), int(inp[2])

    x = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        z = model(x)
        if z.dim() > 2:
            z = z.mean(dim=tuple(range(2, z.dim())))
    return int(z.shape[-1])
