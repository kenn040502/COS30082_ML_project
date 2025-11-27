from __future__ import annotations
import torch
import timm

def load_dino(model_name: str, device: torch.device):
    """
    Load a DINOv2 (or any timm vision backbone) and return:
      - model: feature extractor (no classifier head)
      - preprocess: torchvision-style transform
    """
    print(f"🌿 Loading backbone via timm: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)

    # Build transforms from timm's config
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    cfg = resolve_data_config({}, model=model)
    preprocess = create_transform(**cfg)
    print(f"🖼️  Preprocess input size set to: {cfg.get('input_size', ('?', '?'))[1]}×{cfg.get('input_size', ('?', '?'))[2]}")
    return model, preprocess
