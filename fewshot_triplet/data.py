from __future__ import annotations
import os, random
from typing import List, Tuple, Dict
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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

def collect_images(data_root: str, domain: str) -> list[tuple[str, str]]:
    base = os.path.join(data_root, "train", domain)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Not found: {base}")
    samples = []
    for cid in sorted(os.listdir(base)):
        cdir = os.path.join(base, cid)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                samples.append((os.path.join(cdir, fn), cid))
    if not samples:
        print(f"⚠️  No images found under {base}")
    return samples

class LabeledDataset(Dataset):
    def __init__(self, samples: list[tuple[str,str]], cls2idx: dict[str,int], preprocess):
        self.samples = []
        for p, cid in samples:
            if cid in cls2idx and os.path.isfile(p):
                self.samples.append((p, cls2idx[cid]))
        self.preprocess = preprocess

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert('RGB')
        x = self.preprocess(img)
        return x, y, p

def split_fewshot(data_root: str, shots_per_class: int, train_domain: str, eval_domain: str):
    class_ids, cls2idx = build_class_mapping_from_folders(data_root)
    train_all = collect_images(data_root, train_domain)
    eval_all  = collect_images(data_root, eval_domain)

    per_class = {cid: [] for cid in class_ids}
    for p, cid in train_all:
        if cid in per_class:
            per_class[cid].append(p)
    shots = []
    for cid in class_ids:
        imgs = per_class.get(cid, [])
        random.shuffle(imgs)
        for p in imgs[:max(0, shots_per_class)]:
            shots.append((p, cid))

    return shots, eval_all, cls2idx, class_ids

def make_loaders(shots, eval_samples, cls2idx, preprocess, batch_train=16, batch_eval=64, num_workers=2):
    ds_train = LabeledDataset(shots, cls2idx, preprocess)
    ds_eval  = LabeledDataset(eval_samples, cls2idx, preprocess)
    dl_train = DataLoader(ds_train, batch_size=batch_train, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_eval  = DataLoader(ds_eval,  batch_size=batch_eval,  shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds_train, ds_eval, dl_train, dl_eval
