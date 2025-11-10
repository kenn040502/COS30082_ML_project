# zeroshot/evaluate.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np

@torch.no_grad()
def _encode_any(model, x: torch.Tensor) -> torch.Tensor:
    # Works for timm (DINOv2) and CLIP-like backbones
    if hasattr(model, "encode_image"):
        z = model.encode_image(x)
    else:
        z = model(x)
        if z.dim() > 2:
            z = z.mean(dim=tuple(range(2, z.dim())))
    return F.normalize(z, dim=-1)

def _load_image(path, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img)

@torch.no_grad()
def evaluate_batched(
    model,
    preprocess,
    device: torch.device,
    image_paths: List[str],
    cls2idx: Dict[str, int],
    text_cls_feats: torch.Tensor,    # [C, D] normalized prototypes
    img_batch: int = 8,
    use_logit_scale: bool = False,
    want_confusion: bool = True,
    return_details: bool = True,
):
    """
    Returns:
      top1_percent, top5_percent, total_images, confusion(CxC or None),
      details = list of (path, true_idx, pred_idx, top5_idxs(list[int]), top5_scores(list[float]))
    Expect image_paths like ".../train/<domain>/<class_id>/file.jpg".
    """
    C = text_cls_feats.shape[0]
    details = []
    if want_confusion:
        conf = np.zeros((C, C), dtype=np.int32)
    else:
        conf = None

    def infer_true_idx(p: str) -> int:
        # .../train/<domain>/<class_id>/filename
        parts = p.replace("\\", "/").split("/")
        try:
            dom = parts[-3]
            cid = parts[-2]
        except Exception:
            return -1
        return cls2idx.get(cid, -1)

    total = 0
    correct1 = 0
    correct5 = 0

    batch = []
    batch_meta = []  # (path, true_idx)
    for path in image_paths:
        try:
            x = _load_image(path, preprocess)
        except Exception:
            continue
        t = infer_true_idx(path)
        batch.append(x)
        batch_meta.append((path, t))
        if len(batch) >= img_batch:
            _flush()
            batch, batch_meta = [], []
    if batch:
        _flush()

    def finalize():
        top1 = 100.0 * correct1 / max(1, total)
        top5 = 100.0 * correct5 / max(1, total)
        return top1, top5, total, conf, details

    def _encode(xb: torch.Tensor) -> torch.Tensor:
        z = _encode_any(model, xb)           # [B,D]
        return z

    def _flush():
        nonlocal correct1, correct5, total, details, conf
        xb = torch.stack(batch, 0).to(device)     # [B,3,H,W]
        z = _encode(xb)                            # [B,D]
        # similarity to prototypes
        sims = z @ text_cls_feats.T                # [B,C]
        if use_logit_scale and hasattr(model, "logit_scale"):
            sims = model.logit_scale.exp() * sims

        vals, idxs = sims.topk(k=min(5, sims.shape[1]), dim=-1)  # top-k scores and indices
        preds = idxs[:, 0]

        # update stats / details
        for i in range(xb.shape[0]):
            path, t = batch_meta[i]
            p = int(preds[i].item())
            topk = idxs[i].tolist()
            scores = vals[i].tolist()
            total += 1
            if t >= 0:
                if p == t:
                    correct1 += 1
                if t in topk:
                    correct5 += 1
                if conf is not None and t < conf.shape[0]:
                    conf[t, p] += 1
            if return_details:
                details.append((path, t, p, topk, scores))

    # call finalize at the end
    return finalize()
