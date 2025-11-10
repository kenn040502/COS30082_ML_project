# zeroshot/data.py
from __future__ import annotations
import os, csv
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict

def _list_class_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def build_class_mapping_from_folders(data_root: str) -> tuple[list[str], dict[str, int]]:
    herb = _list_class_dirs(os.path.join(data_root, "train", "herbarium"))
    photo = _list_class_dirs(os.path.join(data_root, "train", "photo"))
    union = sorted(set(herb) | set(photo))
    cls2idx = {cid: i for i, cid in enumerate(union)}
    print(f"📁 Found {len(union)} classes (union of herbarium & photo).")
    return union, cls2idx

def load_id_to_name_csv(data_root: str) -> dict[str, str]:
    path = os.path.join(data_root, "list", "class_names.csv")
    mapping: dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row.get("class_id", "")).strip()
                name = str(row.get("name", "")).strip()
                if cid and name:
                    mapping[cid] = name
        if mapping:
            print(f"🧭 Loaded {len(mapping)} class names from {path}")
    return mapping

def collect_images(data_root: str, domain: str) -> list[tuple[str, str]]:
    """Return flat list of (image_path, class_id) for given domain."""
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
        print(f"⚠️  No images found under {base}")
    return samples

def collect_class_images_dict(data_root: str, domain: str) -> dict[str, list[str]]:
    """Return dict: class_id -> list of image paths (for prototype building)."""
    base = os.path.join(data_root, "train", domain)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Not found: {base}")
    out: DefaultDict[str, list[str]] = defaultdict(list)
    for cid in sorted(os.listdir(base)):
        cdir = os.path.join(base, cid)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out[cid].append(os.path.join(cdir, fn))
    return dict(out)
