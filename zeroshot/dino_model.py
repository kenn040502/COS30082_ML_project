# zeroshot/dino_model.py
from __future__ import annotations
import timm, torch
import torch.nn.functional as F

def load_dino(model_name: str, device: torch.device):
    """
    Load DINOv2 from timm and an eval-time preprocessing transform.
    """
    print(f"🌿 Loading DINOv2 via timm: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # returns features
    model.eval().to(device)

    # Build transforms from timm's default_cfg
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    cfg = resolve_data_config({}, model=model)
    preprocess = create_transform(**cfg, is_training=False)
    return model, preprocess

@torch.no_grad()
def encode_images(model, device, images: torch.Tensor):
    """
    images: [B,3,H,W] float in [0,1] normalized by preprocess
    returns L2-normalized embeddings [B,D]
    """
    feats = model(images.to(device))
    if feats.dim() > 2:
        feats = feats.mean(dim=tuple(range(2, feats.dim())))
    return F.normalize(feats, dim=-1)

@torch.no_grad()
def build_class_prototypes(
    model,
    preprocess,
    device,
    class_to_paths: dict[str, list[str]],
    class_ids_in_order: list[str],
    max_per_class: int,
    img_batch: int,
):
    """
    For each class id in class_ids_in_order, average up to N embeddings
    from class_to_paths[cid].
    Returns tensor [C,D] L2-normalized.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    protos = []
    for cid in class_ids_in_order:
        paths = class_to_paths.get(cid, [])
        if not paths:
            protos.append(None)
            continue
        paths = paths[:max(1, max_per_class)]

        batch = []
        embs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(preprocess(img))
            except Exception as e:
                continue
            if len(batch) >= img_batch:
                x = torch.stack(batch, 0).to(device)
                z = encode_images(model, device, x)
                embs.append(z.cpu())
                batch.clear()
        if batch:
            x = torch.stack(batch, 0).to(device)
            z = encode_images(model, device, x)
            embs.append(z.cpu())
            batch.clear()

        if not embs:
            protos.append(None)
            continue
        z = torch.cat(embs, 0).mean(0, keepdim=True)  # [1,D]
        z = F.normalize(z, dim=-1)
        protos.append(z)

    # replace missing with zeros then stack
    D = protos[0].shape[-1] if any(p is not None for p in protos) else 0
    out = []
    for p in protos:
        if p is None:
            out.append(torch.zeros(1, D))
        else:
            out.append(p)
    return torch.cat(out, dim=0)  # [C,D]
