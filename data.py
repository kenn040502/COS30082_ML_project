from __future__ import annotations
import os
from typing import List, Tuple, Dict

def _list_class_dirs(path: str) -> List[str]:
    """List class-id subfolders under a given path."""
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )

def build_class_mapping_from_folders(data_root: str) -> tuple[list[str], dict[str, int]]:
    """
    Look under:
      train/herbarium/<class_id>/
      train/photo/<class_id>/
    Build:
      - class_ids: sorted list of all class IDs (union of both domains)
      - cls2idx: mapping class_id -> integer index
    """
    herb = _list_class_dirs(os.path.join(data_root, "train", "herbarium"))
    photo = _list_class_dirs(os.path.join(data_root, "train", "photo"))
    union = sorted(set(herb) | set(photo))
    cls2idx: Dict[str, int] = {cid: i for i, cid in enumerate(union)}
    return union, cls2idx

def collect_images(data_root: str, domain: str) -> list[tuple[str, str]]:
    """
    Collect all images for a domain (herbarium or photo) under:
      train/<domain>/<class_id>/*.jpg|*.jpeg|*.png

    Returns list of (image_path, class_id).
    """
    base = os.path.join(data_root, "train", domain)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Not found: {base}")
    samples: list[tuple[str, str]] = []
    for cid in sorted(os.listdir(base)):
        cdir = os.path.join(base, cid)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cdir, fn), cid))
    if not samples:
        print(f"[data] ⚠️ No images found under {base}")
    return samples
