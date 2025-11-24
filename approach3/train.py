from __future__ import annotations
"""
Approach 3: Domain-adaptive metric learning with DINOv2 backbone.

Usage (example, PowerShell):
    $ROOT = "C:\\Users\\User\\Documents\\GitHub\\COS30082_ML_project\\AML_project_herbarium_dataset"
    python -m approach3_train --data-root "$ROOT" --outdir "runs_approach3"

This script:
  - Builds train set from train/herbarium and train/photo
  - Uses DINOv2 backbone from timm (via dino_model.load_dino)
  - Adds projection head + classifier + domain discriminator
  - Trains with:
      * classification loss (cross-entropy, class-weighted)
      * triplet loss on embedding space
      * domain-adversarial loss (DANN-style) for herbarium vs photo
  - Logs metrics per epoch and saves:
      * best checkpoint: outdir/best_model.pt
      * training_curves.png (loss & accuracy)
      * lr_curve.png (learning rate)
      * train_log.json
"""

import os, argparse, json, random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    # fallback: no-progress iterator
    def tqdm(x, **kwargs):
        return x


def mixup_data(x, y, alpha=0.2, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha <= 0:
        return x, y, 1.0, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, lam, y_b


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, device=None):
    if alpha <= 0:
        return x, y, 1.0, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exact area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, lam, y_b

from dino_model import load_dino
from data import build_class_mapping_from_folders, collect_images
from paired_data import load_pairs_csv


# -------------------------
# Dataset
# -------------------------

class PlantDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, int, int]], transform):
        """
        samples: list of (path, class_idx, domain_label, is_paired_class)
            domain_label: 0 = herbarium, 1 = photo
            is_paired_class: 1 if this class appears in pairs.csv, else 0
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid, dom, paired = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, cid, dom, paired, path


# -------------------------
# Gradient Reversal Layer
# -------------------------

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


# -------------------------
# Model heads
# -------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, emb_dim),
        )

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=-1)


class SpeciesClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class DomainDiscriminator(nn.Module):
    def __init__(self, emb_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, lambd: float = 1.0):
        z_rev = grad_reverse(z, lambd)
        return self.net(z_rev)


# -------------------------
# Utils
# -------------------------

@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    d: torch.Tensor
    paired: torch.Tensor
    paths: List[str]


def collate_fn(batch):
    xs, ys, ds, ps, paths = zip(*batch)
    return Batch(
        x=torch.stack(xs, 0),
        y=torch.tensor(ys, dtype=torch.long),
        d=torch.tensor(ds, dtype=torch.long),
        paired=torch.tensor(ps, dtype=torch.float32),
        paths=list(paths),
    )


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    import numpy as np
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_triplets(z: torch.Tensor, y: torch.Tensor, margin: float = 0.3):
    """
    Simple triplet mining:
      for each anchor, pick a random positive (if exists) and a random negative.
    """
    device = z.device
    N = z.size(0)
    z = torch.nn.functional.normalize(z, dim=-1)

    # group indices by class
    indices_by_class = {}
    for i in range(N):
        c = int(y[i].item())
        indices_by_class.setdefault(c, []).append(i)

    anchors, positives, negatives = [], [], []
    all_indices = list(range(N))

    for c, idxs in indices_by_class.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            # anchor i, pick a different positive j
            j = random.choice([k for k in idxs if k != i])
            # pick negative from other classes
            neg_pool = [k for k in all_indices if y[k].item() != c]
            if not neg_pool:
                continue
            k = random.choice(neg_pool)
            anchors.append(i)
            positives.append(j)
            negatives.append(k)

    if not anchors:
        return torch.tensor(0.0, device=device)

    a = torch.tensor(anchors, device=device, dtype=torch.long)
    p = torch.tensor(positives, device=device, dtype=torch.long)
    n = torch.tensor(negatives, device=device, dtype=torch.long)

    za, zp, zn = z[a], z[p], z[n]
    d_ap = 1.0 - (za * zp).sum(dim=-1)  # cosine distance
    d_an = 1.0 - (za * zn).sum(dim=-1)
    loss = torch.nn.functional.relu(d_ap - d_an + margin).mean()
    return loss


# -------------------------
# Training loop
# -------------------------

def train(args):
    device = torch.device(args.device)

    # Build class mapping
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)
    num_classes = len(class_ids)
    print(f"Found {num_classes} classes.")

    # Load pairs info to know which classes are "paired"
    paired_cids = set()
    try:
        pairs = load_pairs_csv(args.data_root)
        for _, _, cid in pairs:
            paired_cids.add(cid)
        print(f"Loaded {len(paired_cids)} paired classes from list/pairs.csv")
    except Exception as e:
        print(f"⚠️ Could not load pairs.csv: {e}")

    # Build train samples
    herb = collect_images(args.data_root, "herbarium")
    photo = collect_images(args.data_root, "photo")
    if not herb or not photo:
        raise RuntimeError("No training images found under train/herbarium or train/photo")

    def to_samples(items, domain_label: int):
        samples = []
        for path, cid in items:
            if cid not in cls2idx:
                continue
            y = cls2idx[cid]
            is_paired = 1 if cid in paired_cids else 0
            samples.append((path, y, domain_label, is_paired))
        return samples

    herb_samples = to_samples(herb, 0)
    photo_samples = to_samples(photo, 1)
    all_samples = herb_samples + photo_samples
    print(f"Train samples: herbarium={len(herb_samples)}, photo={len(photo_samples)}")

    # Compute class weights for imbalance
    all_labels = [s[1] for s in all_samples]
    class_weights = compute_class_weights(all_labels, num_classes).to(device)
    print("Class weights (first 10):", class_weights[:10].tolist())

    # Train/val split on photo domain for monitoring
    random.shuffle(photo_samples)
    split = int(0.8 * len(photo_samples))
    photo_train = photo_samples[:split]
    photo_val = photo_samples[split:]
    train_samples = herb_samples + photo_train
    val_samples = photo_val  # only field images

    # Load backbone & preprocess
    backbone, preprocess = load_dino(args.model, device)
    in_dim = backbone.num_features if hasattr(backbone, "num_features") else 768

    # Performance: enable cuDNN benchmark when using GPU for fixed-size inputs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Optional train-time augmentations (applied before the timm preprocess)
    try:
        from torchvision import transforms as T
        if args.augment:
            # Configurable augmentation strength (light or strong)
            size = 518
            if getattr(args, 'aug_strength', 'strong') == 'light':
                aug_list = [
                    T.RandomResizedCrop((size, size), scale=(0.9, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                ]
            else:
                aug_list = [
                    T.RandomResizedCrop((size, size), scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                ]
                try:
                    from torchvision.transforms import RandAugment
                    aug_list.append(RandAugment(num_ops=2, magnitude=9))
                except Exception:
                    pass

            # RandomErasing must be applied after preprocess because it expects a tensor
            post_pre_list = []
            try:
                post_pre_list.append(T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
            except Exception:
                # RandomErasing may not be available in older torchvision versions
                pass

            aug = T.Compose(aug_list)
            # apply aug (PIL) -> preprocess (to tensor & normalize) -> post_pre_list (tensor ops)
            train_transform = T.Compose([aug, preprocess] + post_pre_list)
        else:
            train_transform = preprocess
    except Exception:
        # If torchvision isn't available or composition fails, fall back to preprocess
        train_transform = preprocess

    # Datasets & loaders
    ds_train = PlantDataset(train_samples, train_transform)
    ds_val = PlantDataset(val_samples, preprocess)

    pin_memory = True if device.type == "cuda" else False
    persistent = True if args.num_workers > 0 else False

    # improve throughput with prefetch_factor when workers > 0
    dl_train_kwargs = dict(batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=pin_memory,
                           persistent_workers=persistent, collate_fn=collate_fn)
    dl_val_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=pin_memory,
                         persistent_workers=persistent, collate_fn=collate_fn)

    if args.num_workers > 0:
        dl_train_kwargs["prefetch_factor"] = 2
        dl_val_kwargs["prefetch_factor"] = 2

    dl_train = DataLoader(ds_train, **dl_train_kwargs)
    dl_val = DataLoader(ds_val, **dl_val_kwargs)

    # Heads
    emb_dim = args.emb_dim
    proj = ProjectionHead(in_dim, emb_dim).to(device)
    clf = SpeciesClassifier(emb_dim, num_classes).to(device)
    dom = DomainDiscriminator(emb_dim).to(device)

    params = list(proj.parameters()) + list(clf.parameters()) + list(dom.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_cls": [],
        "train_triplet": [],
        "train_da": [],
        "val_acc": [],
        "lr": [],
    }

    os.makedirs(args.outdir, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt = None

    # mixed precision scaler (if requested and CUDA available)
    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for ep in range(1, args.epochs + 1):
        proj.train(); clf.train(); dom.train()
        total_loss = total_cls = total_trip = total_da = 0.0
        n_batches = 0

        epoch_iter = tqdm(dl_train, desc=f"Train Epoch {ep}/{args.epochs}")
        for batch in epoch_iter:
            # move tensors (use non_blocking when pin_memory True)
            non_blocking = pin_memory
            x = batch.x.to(device, non_blocking=non_blocking)
            y = batch.y.to(device, non_blocking=False)
            d_label = batch.d.to(device, non_blocking=False)
            paired = batch.paired.to(device, non_blocking=False)

            # apply MixUp / CutMix on inputs if requested
            mixed = False
            # If both are enabled choose randomly per-batch to apply MixUp or CutMix
            if getattr(args, 'cutmix', False) and getattr(args, 'mixup', False):
                if random.random() < 0.5:
                    x, y_a, lam, y_b = cutmix_data(x, y, alpha=args.cutmix_alpha, device=device)
                else:
                    x, y_a, lam, y_b = mixup_data(x, y, alpha=args.mixup_alpha, device=device)
                mixed = True
            elif getattr(args, 'cutmix', False):
                x, y_a, lam, y_b = cutmix_data(x, y, alpha=args.cutmix_alpha, device=device)
                mixed = True
            elif getattr(args, 'mixup', False):
                x, y_a, lam, y_b = mixup_data(x, y, alpha=args.mixup_alpha, device=device)
                mixed = True

            # backbone features (frozen) - compute under no_grad
            with torch.no_grad():
                if use_amp:
                    # autocast okay inside no_grad for faster FP16 ops
                    with torch.cuda.amp.autocast(enabled=True):
                        h = backbone(x)
                else:
                    h = backbone(x)
                if h.dim() > 2:
                    h = h.mean(dim=tuple(range(2, h.dim())))

            # forward (heads) with optional autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = proj(h)
                logits = clf(z)
                if mixed:
                    # for mixup/cutmix use blended classification loss and skip triplet/domain losses
                    loss_cls = lam * F.cross_entropy(logits, y_a, weight=class_weights) + \
                               (1.0 - lam) * F.cross_entropy(logits, y_b, weight=class_weights)
                    loss_trip = torch.tensor(0.0, device=device)
                    loss_da = torch.tensor(0.0, device=device)
                else:
                    loss_cls = F.cross_entropy(logits, y, weight=class_weights)
                    loss_trip = make_triplets(z, y, margin=args.margin)

                    # domain loss: only for classes that are in paired set (paired==1)
                    mask = (paired > 0.5)
                    if mask.any():
                        z_paired = z[mask]
                        d_paired = d_label[mask]
                        dom_logits = dom(z_paired, lambd=args.da_lambda)
                        loss_da = F.cross_entropy(dom_logits, d_paired)
                    else:
                        loss_da = torch.tensor(0.0, device=device)

                loss = args.w_cls * loss_cls + args.w_triplet * loss_trip + args.w_da * loss_da

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_cls += float(loss_cls.detach().cpu().item())
            total_trip += float(loss_trip.detach().cpu().item())
            total_da += float(loss_da.detach().cpu().item())
            n_batches += 1
            # update tqdm with losses
            epoch_iter.set_postfix(loss=f"{total_loss/max(1,n_batches):.4f}", val_acc="?")

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # Validation accuracy (photo domain)
        proj.eval(); clf.eval()
        correct = total = 0
        val_iter = tqdm(dl_val, desc=f"Val Epoch {ep}/{args.epochs}")
        with torch.no_grad():
            for batch in val_iter:
                x = batch.x.to(device, non_blocking=pin_memory)
                y = batch.y.to(device)
                h = backbone(x)
                if h.dim() > 2:
                    h = h.mean(dim=tuple(range(2, h.dim())))
                z = proj(h)
                logits = clf(z)
                pred = logits.argmax(dim=-1)
                total += int(y.size(0))
                correct += int((pred == y).sum().item())
        val_acc = 100.0 * correct / max(1, total)

        mean_loss = total_loss / max(1, n_batches)
        mean_cls = total_cls / max(1, n_batches)
        mean_trip = total_trip / max(1, n_batches)
        mean_da = total_da / max(1, n_batches)

        print(f"Epoch {ep:02d}/{args.epochs}: "
              f"loss={mean_loss:.4f} (cls={mean_cls:.4f}, trip={mean_trip:.4f}, da={mean_da:.4f}) "
              f"| val_acc={val_acc:.2f}% | lr={lr_now:.2e}")

        history["epoch"].append(ep)
        history["train_loss"].append(mean_loss)
        history["train_cls"].append(mean_cls)
        history["train_triplet"].append(mean_trip)
        history["train_da"].append(mean_da)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(args.outdir, "best_model.pt")
            torch.save({
                "backbone_name": args.model,
                "proj_state": proj.state_dict(),
                "clf_state": clf.state_dict(),
                "dom_state": dom.state_dict(),
                "class_ids": class_ids,
                "cls2idx": cls2idx,
                "emb_dim": emb_dim,
            }, best_ckpt)
            print(f"  🔥 New best model saved to {best_ckpt}")

    # Save training log
    log_path = os.path.join(args.outdir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Plot curves
    epochs = history["epoch"]
    # Loss / Val acc
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["train_cls"], label="Cls loss")
    plt.plot(epochs, history["train_triplet"], label="Triplet loss")
    plt.plot(epochs, history["train_da"], label="Domain loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    loss_png = os.path.join(args.outdir, "training_losses.png")
    plt.savefig(loss_png, dpi=150); plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["val_acc"], label="Val (photo) Top-1")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.tight_layout()
    acc_png = os.path.join(args.outdir, "val_accuracy.png")
    plt.savefig(acc_png, dpi=150); plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["lr"], label="LR")
    plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.legend(); plt.tight_layout()
    lr_png = os.path.join(args.outdir, "lr_curve.png")
    plt.savefig(lr_png, dpi=150); plt.close()

    print(f"Saved PNGs:\n  {loss_png}\n  {acc_png}\n  {lr_png}")
    if best_ckpt:
        print(f"Best validation accuracy: {best_val_acc:.2f}% (model at {best_ckpt})")
    else:
        print("No best checkpoint saved? Something went wrong.")

def parse_args():
    p = argparse.ArgumentParser("Approach 3 - Domain-adaptive metric learning with DINOv2")
    p.add_argument("--data-root", required=True, help="Root of AML herbarium dataset")
    p.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m",
                   help="timm model name for DINOv2 backbone")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--emb-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    p.add_argument("--use-onecycle", action="store_true", dest="use_onecycle",
                   help="Use OneCycleLR scheduler (per-iteration)")
    p.add_argument("--finetune-backbone", action="store_true", dest="finetune_backbone",
                   help="Unfreeze entire backbone and fine-tune with smaller LR")
    p.add_argument("--backbone-lr", type=float, default=1e-4,
                   help="Learning rate for backbone when finetuning")
    p.add_argument("--augment", action="store_true", help="Enable train-time data augmentation")
    p.add_argument("--aug-strength", choices=["light","strong"], default="strong",
                   help="Augmentation strength when --augment is set")
    p.add_argument("--use-sampler", action="store_true", dest="use_sampler",
                   help="Use WeightedRandomSampler to handle class imbalance")
    p.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    p.add_argument("--w-cls", type=float, default=1.0, dest="w_cls")
    p.add_argument("--w-triplet", type=float, default=0.5, dest="w_triplet")
    p.add_argument("--w-da", type=float, default=0.1, dest="w_da")
    p.add_argument("--da-lambda", type=float, default=1.0, dest="da_lambda",
                   help="Gradient reversal strength")
    p.add_argument("--outdir", default="runs_approach3")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mixup", action="store_true", help="Enable MixUp augmentation")
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="Alpha for MixUp Beta distribution")
    p.add_argument("--cutmix", action="store_true", help="Enable CutMix augmentation")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="Alpha for CutMix Beta distribution")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)
