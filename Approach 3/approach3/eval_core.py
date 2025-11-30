from __future__ import annotations
import os, json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dino_model import load_dino

from .models import ProjectionHead, SpeciesClassifier
from .datasets import TestDataset, build_paired_set, make_test_samples

def eval_main(args):
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # ----- Load checkpoint -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone_name = ckpt.get("backbone_name", args.model)
    class_ids = ckpt["class_ids"]
    cls2idx = ckpt["cls2idx"]
    emb_dim = ckpt["emb_dim"]
    num_classes = len(class_ids)

    print(f"[eval] Loaded checkpoint: {args.ckpt}")
    print(f"[eval] Backbone: {backbone_name}, classes={num_classes}, emb_dim={emb_dim}")

    # ----- Paired set -----
    paired_cids = set()
    try:
        paired_cids = build_paired_set(args.data_root)
        print(f"[eval] Loaded {len(paired_cids)} paired classes from list/pairs.csv")
    except Exception as e:
        print(f"[eval] ⚠️ Could not load pairs.csv: {e}")

    # ----- Backbone & heads -----
    backbone, preprocess = load_dino(backbone_name, device)
    in_dim = backbone.num_features if hasattr(backbone, "num_features") else 768

    proj = ProjectionHead(in_dim, emb_dim).to(device)
    clf = SpeciesClassifier(emb_dim, num_classes).to(device)
    proj.load_state_dict(ckpt["proj_state"])
    clf.load_state_dict(ckpt["clf_state"])
    proj.eval(); clf.eval()

    # ----- Test data -----
    samples = make_test_samples(args.data_root, cls2idx, paired_cids)
    ds_test = TestDataset(samples, preprocess)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    total = correct1 = correct5 = 0
    total_p = correct1_p = correct5_p = 0
    total_np = correct1_np = correct5_np = 0

    conf = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for x, y, is_paired, paths in dl_test:
            x = x.to(device)
            y = y.to(device)

            h = backbone(x)
            if h.dim() > 2:
                h = h.mean(dim=tuple(range(2, h.dim())))
            z = proj(h)
            logits = clf(z)
            probs = F.softmax(logits, dim=-1)
            top5_vals, top5_idx = probs.topk(k=min(5, num_classes), dim=-1)

            preds1 = top5_idx[:, 0]
            for yi, pi in zip(y, preds1):
                conf[int(yi.item()), int(pi.item())] += 1

            for i in range(y.size(0)):
                yi = int(y[i].item())
                pi1 = int(top5_idx[i, 0].item())
                top5_list = top5_idx[i].tolist()
                is_pair = bool(is_paired[i].item())

                total += 1
                if yi == pi1:
                    correct1 += 1
                if yi in top5_list:
                    correct5 += 1

                if is_pair:
                    total_p += 1
                    if yi == pi1:
                        correct1_p += 1
                    if yi in top5_list:
                        correct5_p += 1
                else:
                    total_np += 1
                    if yi == pi1:
                        correct1_np += 1
                    if yi in top5_list:
                        correct5_np += 1

    def pct(num, den):
        return 100.0 * num / den if den > 0 else 0.0

    metrics = {
        "overall": {
            "N": int(total),
            "top1": pct(correct1, total),
            "top5": pct(correct5, total),
        },
        "with_pairs": {
            "N": int(total_p),
            "top1": pct(correct1_p, total_p),
            "top5": pct(correct5_p, total_p),
        },
        "without_pairs": {
            "N": int(total_np),
            "top1": pct(correct1_np, total_np),
            "top5": pct(correct5_np, total_np),
        },
    }

    # ----- summary.txt -----
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Approach 3 – Domain-adaptive metric learning with DINOv2\n")
        f.write(f"Checkpoint: {args.ckpt}\n\n")
        f.write(f"OVERALL (test/photo): N={metrics['overall']['N']}\n")
        f.write(f"  Top-1: {metrics['overall']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['overall']['top5']:.2f}%\n\n")
        f.write(f"WITH PAIRS classes: N={metrics['with_pairs']['N']}\n")
        f.write(f"  Top-1: {metrics['with_pairs']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['with_pairs']['top5']:.2f}%\n\n")
        f.write(f"WITHOUT PAIRS classes: N={metrics['without_pairs']['N']}\n")
        f.write(f"  Top-1: {metrics['without_pairs']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['without_pairs']['top5']:.2f}%\n")
    print(f"[eval] Summary written to {summary_path}")

    # ----- metrics.json -----
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Metrics JSON written to {metrics_path}")

    # ----- confusion.png -----
    fig = plt.figure(figsize=(8, 7))
    import matplotlib.ticker as ticker
    ax = fig.add_subplot(111)
    im = ax.imshow(conf, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix (test/photo)")
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    plt.tight_layout()
    conf_path = os.path.join(args.outdir, "confusion.png")
    plt.savefig(conf_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Confusion matrix saved to {conf_path}")
