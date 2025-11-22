from __future__ import annotations
import os, json, random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # <<< added
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler

from dino_model import load_dino

from .models import ProjectionHead, SpeciesClassifier, DomainDiscriminator
from .datasets import (
    PlantDataset,
    collate_train,
    build_class_mapping,
    build_paired_set,
    make_train_val_samples,
)
from .losses import compute_class_weights, triplet_loss_random, batch_hard_triplet_loss

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_main(args):
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # ----- Class mapping & paired set -----
    class_ids, cls2idx = build_class_mapping(args.data_root)
    num_classes = len(class_ids)
    print(f"[train] Found {num_classes} classes.")

    paired_cids = set()
    try:
        paired_cids = build_paired_set(args.data_root)
        print(f"[train] Loaded {len(paired_cids)} paired classes from list/pairs.csv")
    except Exception as e:
        print(f"[train] ⚠️ Could not load pairs.csv: {e}")

    # ----- Samples -----
    train_samples, val_samples = make_train_val_samples(args.data_root, cls2idx, paired_cids, val_ratio=0.2)
    print(f"[train] Train samples: {len(train_samples)}, Val (photo) samples: {len(val_samples)}")

    all_labels = [s[1] for s in train_samples]
    class_weights = compute_class_weights(all_labels, num_classes).to(device)
    print("[train] Class weights (first 10):", class_weights[:10].tolist())

    # ----- Backbone & preprocess -----
    backbone, preprocess = load_dino(args.model, device)
    in_dim = backbone.num_features if hasattr(backbone, "num_features") else 768

    # ----- Datasets & loaders -----
    # Build transforms: optionally add augmentation before the backbone preprocess
    if getattr(args, "augment", False):
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomGrayscale(p=0.05),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                preprocess,  # DINO preprocess (resize / to tensor / normalize)
            ]
        )
    else:
        train_transform = preprocess

    val_transform = preprocess

    ds_train = PlantDataset(train_samples, train_transform)
    ds_val = PlantDataset(val_samples, val_transform)

    # Optionally use WeightedRandomSampler to counter class imbalance
    sampler = None
    if getattr(args, "use_sampler", False):
        # compute class counts from train_samples
        counts = {}
        for _, y, _, _ in train_samples:
            counts[y] = counts.get(y, 0) + 1
        sample_weights = [1.0 / counts[s[1]] for s in train_samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_train,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_train,
    )

    # ----- Heads -----
    emb_dim = args.emb_dim
    proj = ProjectionHead(in_dim, emb_dim).to(device)
    clf = SpeciesClassifier(emb_dim, num_classes).to(device)
    dom = DomainDiscriminator(emb_dim).to(device)

    # prepare parameter groups; optionally include backbone for finetuning
    head_params = list(proj.parameters()) + list(clf.parameters()) + list(dom.parameters())
    if getattr(args, "finetune_backbone", False):
        # unfreeze backbone
        for p in backbone.parameters():
            p.requires_grad = True
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        params = [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": getattr(args, "backbone_lr", args.lr * 0.1)},
        ]
    else:
        params = head_params

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # scheduler: OneCycle per-iteration or fallback Cosine per-epoch
    use_onecycle = getattr(args, "use_onecycle", False)
    if use_onecycle:
        steps_per_epoch = max(1, len(dl_train))
        scheduler = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
        step_scheduler_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        step_scheduler_per_batch = False

    history = {
        "epoch": [],
        "train_loss": [],
        "train_cls": [],
        "train_triplet": [],
        "train_da": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_ckpt = None

    scaler = GradScaler() if (getattr(args, "amp", False) and device.type == "cuda") else None

    for ep in range(1, args.epochs + 1):
        proj.train(); clf.train(); dom.train()
        total_loss = total_cls = total_trip = total_da = 0.0
        n_batches = 0

        # ===== TRAIN LOOP with tqdm =====
        for batch in tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}", ncols=100):
            x = batch.x.to(device)
            y = batch.y.to(device)
            d_label = batch.d.to(device)
            paired_flag = batch.paired.to(device)

            # backbone forward (frozen or trainable depending on args)
            if getattr(args, "finetune_backbone", False):
                h = backbone(x)
            else:
                with torch.no_grad():
                    h = backbone(x)

            if h.dim() > 2:
                h = h.mean(dim=tuple(range(2, h.dim())))

            # use AMP when available
            if scaler is not None:
                with autocast():
                    z = proj(h)
                    logits = clf(z)
                    # classification
                    loss_cls = F.cross_entropy(logits, y, weight=class_weights)
                    # batch-hard triplet
                    loss_trip = batch_hard_triplet_loss(z, y, margin=args.margin)
                    # domain adversarial (only on paired classes)
                    mask = (paired_flag > 0.5)
                    if mask.any():
                        z_p = z[mask]
                        d_p = d_label[mask]
                        dom_logits = dom(z_p, lambd=args.da_lambda)
                        loss_da = F.cross_entropy(dom_logits, d_p)
                    else:
                        loss_da = torch.tensor(0.0, device=device)
                    loss = args.w_cls * loss_cls + args.w_triplet * loss_trip + args.w_da * loss_da

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # unscale before clipping
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                z = proj(h)
                logits = clf(z)
                # classification
                loss_cls = F.cross_entropy(logits, y, weight=class_weights)
                # batch-hard triplet
                loss_trip = batch_hard_triplet_loss(z, y, margin=args.margin)

                # domain adversarial (only on paired classes)
                mask = (paired_flag > 0.5)
                if mask.any():
                    z_p = z[mask]
                    d_p = d_label[mask]
                    dom_logits = dom(z_p, lambd=args.da_lambda)
                    loss_da = F.cross_entropy(dom_logits, d_p)
                else:
                    loss_da = torch.tensor(0.0, device=device)

                loss = args.w_cls * loss_cls + args.w_triplet * loss_trip + args.w_da * loss_da

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

            # step scheduler per batch if OneCycle
            if step_scheduler_per_batch:
                scheduler.step()

            # domain adversarial (only on paired classes)
            mask = (paired_flag > 0.5)
            if mask.any():
                z_p = z[mask]
                d_p = d_label[mask]
                dom_logits = dom(z_p, lambd=args.da_lambda)
                loss_da = F.cross_entropy(dom_logits, d_p)
            else:
                loss_da = torch.tensor(0.0, device=device)

            loss = args.w_cls * loss_cls + args.w_triplet * loss_trip + args.w_da * loss_da

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_trip += loss_trip.item()
            total_da += loss_da.item()
            n_batches += 1

        # step scheduler per epoch if not using OneCycle
        if not step_scheduler_per_batch:
            scheduler.step()
        # get learning rate (take first param group)
        lr_now = None
        try:
            lr_now = opt.param_groups[0]["lr"]
        except Exception:
            lr_now = scheduler.get_last_lr()[0]

        # ===== VALIDATION LOOP with tqdm =====
        proj.eval(); clf.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc="Validating", ncols=100):
                x = batch.x.to(device)
                y = batch.y.to(device)
                h = backbone(x)
                if h.dim() > 2:
                    h = h.mean(dim=tuple(range(2, h.dim())))
                z = proj(h)
                logits = clf(z)
                pred = logits.argmax(dim=-1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        val_acc = 100.0 * correct / max(1, total)

        mean_loss = total_loss / max(1, n_batches)
        mean_cls = total_cls / max(1, n_batches)
        mean_trip = total_trip / max(1, n_batches)
        mean_da = total_da / max(1, n_batches)

        print(f"[train] Epoch {ep:02d}/{args.epochs}: "
              f"loss={mean_loss:.4f} (cls={mean_cls:.4f}, trip={mean_trip:.4f}, da={mean_da:.4f}) "
              f"| val_acc={val_acc:.2f}% | lr={lr_now:.2e}")

        history["epoch"].append(ep)
        history["train_loss"].append(mean_loss)
        history["train_cls"].append(mean_cls)
        history["train_triplet"].append(mean_trip)
        history["train_da"].append(mean_da)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        # save best
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
            print(f"[train] 🔥 New best model saved to {best_ckpt}")

    # ===== Save log =====
    log_path = os.path.join(args.outdir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Log saved to {log_path}")

    # ===== Plots =====
    epochs = history["epoch"]

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

    print("[train] Saved plots:")
    print(" ", loss_png)
    print(" ", acc_png)
    print(" ", lr_png)
    if best_ckpt:
        print(f"[train] Best val acc: {best_val_acc:.2f}% (model at {best_ckpt})")
    else:
        print("[train] ⚠️ No best checkpoint?")
