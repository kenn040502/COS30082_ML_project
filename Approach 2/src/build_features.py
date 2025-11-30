import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from extractor_dinov2 import load_dinov2_feature_extractor, get_transform


# -------------------------------------------------
# 1. Helpers to read list files and (optionally) ground truth
# -------------------------------------------------

def read_list_file(list_path: Path) -> List[Tuple[Path, int, str]]:
    """
    Reads train.txt or test.txt.

    Each line is like:
        train/herbarium/105951/283042.jpg 105951
    or:
        test/1000.jpg -1

    Returns a list of tuples:
        (absolute_image_path, numeric_label_from_list_or_-1, basename)
    basename is e.g. "1000.jpg", used later to line up with groundtruth.
    """
    root = list_path.parent.parent  # go up from ".../list/train.txt" -> dataset root
    out = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 1:
                continue

            rel_path = parts[0]  # e.g. test/1000.jpg
            img_name = Path(rel_path).name  # e.g. 1000.jpg

            # default label=-1 if missing
            lab = -1
            if len(parts) >= 2:
                try:
                    lab = int(parts[1])
                except ValueError:
                    lab = -1

            abs_path = root / rel_path
            if not abs_path.exists():
                print(f"[WARN] missing file listed: {abs_path}")
                continue

            out.append((abs_path, lab, img_name))
    if not out:
        raise RuntimeError(f"No valid items found in {list_path}")
    return out


def read_groundtruth(gt_path: Path) -> Dict[str, int]:
    """
    Reads groundtruth.txt.

    Format from your dataset:
        test/1000.jpg 130657
        test/136.jpg 13276
        ...

    Returns dict:
        { "1000.jpg": 130657, "136.jpg": 13276, ... }
    """
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue

            rel_path = parts[0]      # e.g. test/1000.jpg
            label_id = parts[1]      # e.g. 130657

            img_name = Path(rel_path).name  # "1000.jpg"
            try:
                gt_map[img_name] = int(label_id)
            except ValueError:
                # skip weird lines
                continue
    return gt_map


def apply_groundtruth_if_available(
    items: List[Tuple[Path, int, str]],
    gt_map: Dict[str, int] | None
) -> np.ndarray:
    """
    Build the final label array y for saving to .npz.

    - If gt_map is None (train mode), just use the labels from the list file.
    - If gt_map is provided (test mode), override the labels using the map
      so each image gets its true ground truth ID.

    Returns np.array of shape (N,), dtype=int
    """
    final_labels = []
    missing = []

    for (_, list_label, img_name) in items:
        if gt_map is None:
            # training or any list that already has valid labels
            final_labels.append(list_label)
        else:
            # test mode with provided groundtruth.txt
            if img_name in gt_map:
                final_labels.append(gt_map[img_name])
            else:
                # no groundtruth? mark as -1
                final_labels.append(-1)
                missing.append(img_name)

    y = np.array(final_labels, dtype=int)

    if gt_map is not None:
        if missing:
            print(f"[WARN] {len(missing)} test images missing ground truth. First few: {missing[:10]}")
        else:
            print("[INFO] All test images got ground truth labels from groundtruth.txt")

    return y


# -------------------------------------------------
# 2. Torch Dataset for extraction
# -------------------------------------------------

class ImageFeatureDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int, str]], transform):
        """
        items is [(abs_path, label, img_name), ...]
        """
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label, img_name = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label, img_name


# -------------------------------------------------
# 3. Main feature builder
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-list", required=True,
                        help="Path to train.txt or test.txt")
    parser.add_argument("--weights", required=True,
                        help="Path to DINOv2 checkpoint (model_best.pth.tar)")
    parser.add_argument("--out", required=True,
                        help="Output npz path, e.g. features/train.npz or features/test.npz")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for extraction")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (auto if not set)")
    parser.add_argument("--groundtruth", type=str, default=None,
                        help="(Optional) Path to groundtruth.txt. "
                             "If provided, we will inject correct test labels instead of -1.")
    args = parser.parse_args()

    list_path = Path(args.data_list)
    gt_path = Path(args.groundtruth) if args.groundtruth else None

    # step A: read the list file (train.txt or test.txt)
    items = read_list_file(list_path)

    # step B: if groundtruth is given, load it and override the labels
    gt_map = read_groundtruth(gt_path) if gt_path is not None else None
    labels_np = apply_groundtruth_if_available(items, gt_map)

    # step C: load DINOv2 feature extractor
    model, feat_dim = load_dinov2_feature_extractor(
        Path(args.weights),
        device=args.device,
    )
    tf = get_transform(518)

    # step D: build dataloader
    ds = ImageFeatureDataset(items, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    all_feats = []
    # NOTE: we don't actually need labels during forward, but nice to keep
    print(f"[BUILD] items to process: {len(ds)}")

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dl, desc=f"Extract {list_path.name}", ncols=80)
        for batch_imgs, _, _ in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            feats = model(batch_imgs)  # (B, feat_dim), already pooled
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats)

    X = np.concatenate(all_feats, axis=0)  # shape (N, feat_dim)

    # safety check: align X rows with labels rows
    assert X.shape[0] == labels_np.shape[0], \
        f"Feature count {X.shape[0]} != label count {labels_np.shape[0]}"

    # step E: save final npz
    out_path = Path(args.out)
    np.savez_compressed(out_path, X=X, y=labels_np)
    print(f"[BUILD] saved: {out_path}  |  X={X.shape}, y={labels_np.shape}")


if __name__ == "__main__":
    main()
