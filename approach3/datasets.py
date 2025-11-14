from __future__ import annotations
import os, random
from typing import List, Tuple, Dict
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset

from data import build_class_mapping_from_folders, collect_images
from paired_data import load_pairs_csv

# ------------- Dataset classes -------------

class PlantDataset(Dataset):
    """
    For training / validation:
      samples: list of (path, class_idx, domain_label, is_paired_class)
        domain_label: 0 = herbarium, 1 = photo
        is_paired_class: 1 if class has herbarium-photo pairs, else 0
    """
    def __init__(self, samples: List[Tuple[str, int, int, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid, dom, paired = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, cid, dom, paired, path

class TestDataset(Dataset):
    """
    For test:
      samples: list of (path, class_idx, is_paired_class)
    """
    def __init__(self, samples: List[Tuple[str, int, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid, is_paired = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, cid, is_paired, path

# ------------- Collate -------------

@dataclass
class TrainBatch:
    x: torch.Tensor
    y: torch.Tensor
    d: torch.Tensor
    paired: torch.Tensor
    paths: List[str]

def collate_train(batch):
    xs, ys, ds, ps, paths = zip(*batch)
    return TrainBatch(
        x=torch.stack(xs, 0),
        y=torch.tensor(ys, dtype=torch.long),
        d=torch.tensor(ds, dtype=torch.long),
        paired=torch.tensor(ps, dtype=torch.float32),
        paths=list(paths),
    )

# ------------- Sample builders -------------

def build_class_mapping(data_root: str) -> tuple[list[str], dict[str, int]]:
    """Wrapper around data.build_class_mapping_from_folders."""
    return build_class_mapping_from_folders(data_root)

def build_paired_set(data_root: str) -> set[str]:
    """
    Load class IDs that have herbarium-photo pairs from list/pairs.csv
    """
    paired_cids: set[str] = set()
    pairs = load_pairs_csv(data_root)  # (photo_path, herb_path, class_id)
    for _, _, cid in pairs:
        paired_cids.add(cid)
    return paired_cids

def make_train_val_samples(
    data_root: str,
    cls2idx: Dict[str, int],
    paired_cids: set[str],
    val_ratio: float = 0.2,
):
    """
    Returns:
      train_samples: list of (path, class_idx, domain_label, is_paired_class)
      val_samples:   list of (path, class_idx, domain_label, is_paired_class)
    where val_samples are only from photo domain (field).
    """
    herb = collect_images(data_root, "herbarium")
    photo = collect_images(data_root, "photo")
    if not herb or not photo:
        raise RuntimeError("No training images found in train/herbarium or train/photo")

    def to_samples(items, domain_label: int):
        s = []
        for path, cid in items:
            if cid not in cls2idx:
                continue
            y = cls2idx[cid]
            is_paired = 1 if cid in paired_cids else 0
            s.append((path, y, domain_label, is_paired))
        return s

    herb_samples = to_samples(herb, 0)
    photo_samples = to_samples(photo, 1)
    print(f"[data] herbarium={len(herb_samples)}, photo={len(photo_samples)}")

    # train/val split only on photo samples
    random.shuffle(photo_samples)
    split = int((1.0 - val_ratio) * len(photo_samples))
    photo_train = photo_samples[:split]
    photo_val = photo_samples[split:]

    train_samples = herb_samples + photo_train
    val_samples = photo_val
    return train_samples, val_samples

def make_test_samples(
    data_root: str,
    cls2idx: Dict[str, int],
    paired_cids: set[str],
) -> List[Tuple[str, int, int]]:
    """
    Load test samples.

    Supports 3 formats:

    1) test/*.jpg + list/groundtruth.txt  <-- YOUR DATASET
       groundtruth.txt lines: "test/xxxx.jpg class_id"

    2) test/photo/<class_id>/*.jpg

    3) test/<class_id>/*.jpg

    Returns list of (path, class_idx, is_paired).
    """
    base_test_root = os.path.join(data_root, "test")

    # ---- Case 1: flat test/ + list/groundtruth.txt ----
    gt_file1 = os.path.join(data_root, "list", "groundtruth.txt")
    gt_file2 = os.path.join(data_root, "groundtruth.txt")

    gt_path = None
    if os.path.isfile(gt_file1):
        gt_path = gt_file1
    elif os.path.isfile(gt_file2):
        gt_path = gt_file2

    if gt_path is not None:
        print(f"[data] Using flat test folder + groundtruth.txt: {gt_path}")

        samples: List[Tuple[str, int, int]] = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expect: "test/xxxx.jpg class_id"
                parts = line.split()
                if len(parts) != 2:
                    continue
                path_rel, class_id = parts
                cid = class_id.strip()
                if cid not in cls2idx:
                    # class in groundtruth but not in train mapping
                    continue

                full_path = os.path.join(data_root, path_rel)
                y = cls2idx[cid]
                is_paired = 1 if cid in paired_cids else 0
                samples.append((full_path, y, is_paired))

        print(f"[data] Loaded {len(samples)} test samples from groundtruth.txt")
        return samples

    # ---- Case 2: test/photo/<class_id>/*.jpg ----
    base_test_photo = os.path.join(base_test_root, "photo")
    if os.path.isdir(base_test_photo):
        base = base_test_photo
        print(f"[data] Using test/photo for evaluation: {base}")
    # ---- Case 3: test/<class_id>/*.jpg ----
    elif os.path.isdir(base_test_root):
        base = base_test_root
        print(f"[data] Using test directory for evaluation (per-class folders): {base}")
    else:
        raise FileNotFoundError(f"No test directory found under {data_root}")

    samples: List[Tuple[str, int, int]] = []
    for cid in sorted(os.listdir(base)):
        cdir = os.path.join(base, cid)
        if not os.path.isdir(cdir):
            continue
        if cid not in cls2idx:
            continue
        y = cls2idx[cid]
        is_paired = 1 if cid in paired_cids else 0
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cdir, fn), y, is_paired))

    print(f"[data] Loaded {len(samples)} test samples")
    return samples

