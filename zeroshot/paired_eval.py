# zeroshot/paired_eval.py
from __future__ import annotations
import os, csv, torch
import torch.nn.functional as F
from typing import List, Tuple
from PIL import Image

@torch.no_grad()
def _encode_any(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        z = model.encode_image(x)
    else:
        z = model(x)
        if z.dim() > 2:
            z = z.mean(dim=tuple(range(2, z.dim())))
    return F.normalize(z, dim=-1)

@torch.no_grad()
def _embed_many(model, preprocess, device, paths: List[str], batch=8):
    embs = []
    buf = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            buf.append(preprocess(img))
        except Exception:
            # skip unreadable
            continue
        if len(buf) >= batch:
            x = torch.stack(buf, 0).to(device)
            z = _encode_any(model, x).cpu()
            embs.append(z)
            buf = []
    if buf:
        x = torch.stack(buf, 0).to(device)
        z = _encode_any(model, x).cpu()
        embs.append(z)
    if not embs:
        return torch.empty(0, 1)
    return F.normalize(torch.cat(embs, 0), dim=-1)

@torch.no_grad()
def paired_retrieval_eval(
    model,
    preprocess,
    device: torch.device,
    data_root: str,
    pairs: List[Tuple[str,str,str]],
    img_batch: int = 8,
):
    # Build unique herb gallery, and query photo list
    herb_paths = []
    herb_index = {}
    photos = []
    gt_herb_idx = []
    for p_rel, h_rel, _ in pairs:
        hp = os.path.join(data_root, h_rel)
        pp = os.path.join(data_root, p_rel)
        if hp not in herb_index:
            herb_index[hp] = len(herb_paths)
            herb_paths.append(hp)
        photos.append(pp)
        gt_herb_idx.append(herb_index[hp])

    herb_emb = _embed_many(model, preprocess, device, herb_paths, batch=img_batch)  # [Nh,D]
    if herb_emb.numel() == 0:
        raise RuntimeError("Failed to embed herbarium gallery.")
    Nh = herb_emb.shape[0]

    # Embed photos in batches and compute ranks
    ranks = []
    buf = []
    meta = []  # index within photos
    for i, pp in enumerate(photos):
        try:
            img = Image.open(pp).convert("RGB")
            buf.append(preprocess(img))
            meta.append(i)
        except Exception:
            ranks.append(Nh + 1)
            continue
        if len(buf) >= img_batch:
            _flush_photo_batch(model, device, buf, meta, herb_emb, gt_herb_idx, ranks)
            buf, meta = [], []
    if buf:
        _flush_photo_batch(model, device, buf, meta, herb_emb, gt_herb_idx, ranks)

    import numpy as np
    R = np.array(ranks, dtype=np.int32)
    valid = R[R > 0]
    if valid.size == 0:
        raise RuntimeError("No valid photo queries embedded.")

    r1 = float(np.mean(valid == 1)) * 100.0
    r5 = float(np.mean(valid <= 5)) * 100.0
    mean_rank = float(np.mean(valid))

    return {
        "num_pairs": int(len(pairs)),
        "num_valid_queries": int(valid.size),
        "R@1 (%)": round(r1, 4),
        "R@5 (%)": round(r5, 4),
        "mean_rank": round(mean_rank, 3),
    }

@torch.no_grad()
def _flush_photo_batch(model, device, buf, meta, herb_emb, gt_herb_idx, ranks_out):
    x = torch.stack(buf, 0).to(device)
    z = _encode_any(model, x)                    # [B,D]
    z = F.normalize(z, dim=-1)
    S = z @ herb_emb.T                           # [B,Nh]
    order = torch.argsort(S, dim=-1, descending=True)
    for bi, idx in enumerate(meta):
        gt = gt_herb_idx[idx]
        pos = (order[bi] == gt).nonzero(as_tuple=False)
        if pos.numel() == 0:
            ranks_out.append(herb_emb.shape[0] + 1)
        else:
            ranks_out.append(int(pos.item()) + 1)

@torch.no_grad()
def paired_per_query_ranks_csv(
    model,
    preprocess,
    device: torch.device,
    data_root: str,
    pairs: List[Tuple[str,str,str]],
    img_batch: int,
    out_csv: str,
):
    # prepare same gallery/query as above
    herb_paths = []
    herb_index = {}
    rows = []  # (photo_abs, herb_abs, class_id, rank)
    photos = []
    gt_herb_idx = []
    for p_rel, h_rel, cid in pairs:
        hp = os.path.join(data_root, h_rel)
        pp = os.path.join(data_root, p_rel)
        if hp not in herb_index:
            herb_index[hp] = len(herb_paths)
            herb_paths.append(hp)
        photos.append(pp)
        gt_herb_idx.append(herb_index[hp])
        rows.append([pp, hp, cid, None])  # fill rank later

    herb_emb = _embed_many(model, preprocess, device, herb_paths, batch=img_batch)
    if herb_emb.numel() == 0:
        raise RuntimeError("Failed to embed herbarium gallery for per-query CSV.")
    Nh = herb_emb.shape[0]

    # Compute ranks with same ordering
    ranks = []
    buf = []
    meta = []
    for i, pp in enumerate(photos):
        try:
            img = Image.open(pp).convert("RGB")
            buf.append(preprocess(img))
            meta.append(i)
        except Exception:
            ranks.append(Nh + 1)
            continue
        if len(buf) >= img_batch:
            _flush_photo_batch(model, device, buf, meta, herb_emb, gt_herb_idx, ranks)
            buf, meta = [], []
    if buf:
        _flush_photo_batch(model, device, buf, meta, herb_emb, gt_herb_idx, ranks)

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_path_abs", "herb_path_abs", "class_id", "rank_of_paired_herb"])
        for i, r in enumerate(ranks):
            rows[i][3] = r
            w.writerow(rows[i])
