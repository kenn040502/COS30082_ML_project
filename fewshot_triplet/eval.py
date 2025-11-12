# fewshot_triplet/eval.py
from __future__ import annotations
import os, json, argparse, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import DEFAULT_MODEL, DEFAULT_BATCH_EVAL, DEFAULT_SHOTS
from .backbone import load_backbone, encode_images, estimate_out_dim
from .data import split_fewshot, make_loaders
from .plots import plot_per_class_accuracy, plot_confusion

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
    def forward(self, z): return F.normalize(self.net(z), dim=-1)

def _relpath_norm(root: str, p: str) -> str:
    ap = os.path.abspath(p)
    ar = os.path.abspath(root)
    rp = os.path.relpath(ap, ar).replace("\\", "/")
    return rp

def _load_pairs_csv(data_root: str) -> set[str]:
    """
    Returns a set of normalized relative paths that are listed as photo_path in list/pairs.csv.
    If file missing, returns empty set.
    """
    pairs_file = os.path.join(data_root, "list", "pairs.csv")
    relset: set[str] = set()
    if not os.path.isfile(pairs_file):
        print(f"ℹ️  No pairs file found at {pairs_file} — with/without pair metrics will be empty subsets.")
        return relset
    with open(pairs_file, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pp = (row.get("photo_path") or "").strip()
            if not pp:
                continue
            # Normalize relative path
            rel = pp.replace("\\", "/")
            relset.add(rel)
    print(f"🔗 Loaded {len(relset)} paired photo paths from list/pairs.csv")
    return relset

@torch.no_grad()
def build_class_means(backbone, head, ds_train, device):
    from torch.utils.data import DataLoader
    backbone.eval(); head.eval()
    C = max(y for _, y, _ in ds_train) + 1 if len(ds_train) > 0 else 0
    sums = [torch.zeros(256, device=device) for _ in range(C)]
    counts = [0 for _ in range(C)]
    dl = DataLoader(ds_train, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    for xb, yb, _ in dl:
        xb, yb = xb.to(device), yb.to(device)
        z = encode_images(backbone, device, xb)  # [B, D_backbone]
        f = head(z)                              # [B, 256]
        for i in range(f.shape[0]):
            sums[yb[i].item()] += f[i]
            counts[yb[i].item()] += 1
    means = []
    for c in range(C):
        if counts[c] > 0:
            v = F.normalize(sums[c] / counts[c], dim=0)
        else:
            v = torch.zeros(256, device=device)
        means.append(v)
    return torch.stack(means, 0)  # [C,256]

@torch.no_grad()
def eval_with_means(backbone, head, ds_eval, class_means, device, data_root: str, paired_relset: set[str]):
    from torch.utils.data import DataLoader
    backbone.eval(); head.eval()
    dl = DataLoader(ds_eval, batch_size=DEFAULT_BATCH_EVAL, shuffle=False, num_workers=2, pin_memory=True)

    C = class_means.shape[0]
    conf = np.zeros((C, C), dtype=np.int32)

    # metrics buckets
    overall = {"total": 0, "top1": 0, "top5": 0}
    withpair = {"total": 0, "top1": 0, "top5": 0}
    without = {"total": 0, "top1": 0, "top5": 0}

    per_image_rows = []

    for xb, yb, paths in dl:
        xb, yb = xb.to(device), yb.to(device)
        z = encode_images(backbone, device, xb)
        f = head(z)
        sims = f @ class_means.t()  # [B,C]
        vals, idxs = sims.topk(k=min(5, C), dim=-1)
        preds = idxs[:, 0]
        for i in range(xb.shape[0]):
            t = int(yb[i].item())
            p = int(preds[i].item())
            rp = _relpath_norm(data_root, paths[i])  # normalized rel path for split
            top5_list = idxs[i].tolist()
            score_list = vals[i].tolist()

            overall["total"] += 1
            if p == t: overall["top1"] += 1
            if t in top5_list: overall["top5"] += 1

            # with/without pair split (by photo_path set)
            if rp in paired_relset:
                withpair["total"] += 1
                if p == t: withpair["top1"] += 1
                if t in top5_list: withpair["top5"] += 1
            else:
                without["total"] += 1
                if p == t: without["top1"] += 1
                if t in top5_list: without["top5"] += 1

            conf[t, p] += 1

            # save per-image top3
            top3 = top5_list[:3]
            scores = score_list[:3]
            per_image_rows.append([
                paths[i],
                t,
                p, float(scores[0]),
                top3[1] if len(top3) > 1 else -1,
                float(scores[1]) if len(scores) > 1 else "",
                top3[2] if len(top3) > 2 else -1,
                float(scores[2]) if len(scores) > 2 else "",
                1 if p == t else 0,
                "with_pair" if rp in paired_relset else "without_pair"
            ])

    def _pct(x, y): return (100.0 * x / max(1, y))
    metrics = {
        "overall": {
            "top1": _pct(overall["top1"], overall["total"]),
            "top5": _pct(overall["top5"], overall["total"]),
            "total": int(overall["total"]),
        },
        "with_pair": {
            "top1": _pct(withpair["top1"], withpair["total"]),
            "top5": _pct(withpair["top5"], withpair["total"]),
            "total": int(withpair["total"]),
        },
        "without_pair": {
            "top1": _pct(without["top1"], without["total"]),
            "top5": _pct(without["top5"], without["total"]),
            "total": int(without["total"]),
        },
    }
    return metrics, conf, per_image_rows

def _save_metrics_bars(outdir: str, metrics: dict):
    # Top-1
    labels = ["Overall", "With Pair", "Without Pair"]
    t1 = [metrics["overall"]["top1"], metrics["with_pair"]["top1"], metrics["without_pair"]["top1"]]
    t5 = [metrics["overall"]["top5"], metrics["with_pair"]["top5"], metrics["without_pair"]["top5"]]

    def _bar(vals, title, fname):
        plt.figure(figsize=(5, 4))
        x = np.arange(len(labels))
        plt.bar(x, vals)
        plt.xticks(x, labels, rotation=0)
        plt.ylabel("Accuracy (%)")
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    p1 = _bar(t1, "Top-1 Accuracy — Overall / With Pair / Without Pair", "metrics_bar_top1.png")
    p5 = _bar(t5, "Top-5 Accuracy — Overall / With Pair / Without Pair", "metrics_bar_top5.png")
    return p1, p5

def main():
    ap = argparse.ArgumentParser("Few-shot Triplet - Evaluate & Report (with Pair/No-Pair splits)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--shots-per-class", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--train-domain", default="herbarium")
    ap.add_argument("--eval-domain",  default="photo")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--outdir", default="fewshot_results/report_${shots}")
    args = ap.parse_args()

    device = torch.device(args.device)
    backbone, preprocess = load_backbone(args.model, device)

    shots, eval_samples, cls2idx, class_ids = split_fewshot(
        args.data_root, args.shots_per_class, args.train_domain, args.eval_domain
    )
    ds_train, ds_eval, _, _ = make_loaders(shots, eval_samples, cls2idx, preprocess,
                                           batch_train=16, batch_eval=DEFAULT_BATCH_EVAL)

    # Load head
    obj = torch.load(args.head_ckpt, map_location=device)
    d_backbone = estimate_out_dim(backbone, device)
    head = ProjectionHead(d_backbone, out_dim=256).to(device)
    head.load_state_dict(obj["head"])

    # Build class means and evaluate
    means = build_class_means(backbone, head, ds_train, device)
    paired_relset = _load_pairs_csv(args.data_root)  # set of relative photo paths (if any)

    t0 = time.time()
    metrics, conf, rows = eval_with_means(backbone, head, ds_eval, means, device, args.data_root, paired_relset)
    t1 = time.time()

    outdir = args.outdir.replace("${shots}", f"{args.shots_per_class}shot")
    os.makedirs(outdir, exist_ok=True)

    # CSV (now with split tag column)
    per_img_csv = os.path.join(outdir, "fewshot_per_image_top3.csv")
    with open(per_img_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_idx", "pred1_idx", "score1",
                    "pred2_idx", "score2", "pred3_idx", "score3",
                    "top1_correct", "split"])
        for r in rows: w.writerow(r)

    # PNGs (per-class + confusion)
    per_class_png = os.path.join(outdir, "fewshot_per_class_top1.png")
    conf_png      = os.path.join(outdir, "fewshot_confusion.png")
    plot_per_class_accuracy(conf, per_class_png, class_ids)
    plot_confusion(conf, conf_png)

    # Metrics bars (Top-1 & Top-5)
    bar_top1_png, bar_top5_png = _save_metrics_bars(outdir, metrics)

    # TXT summary
    txt_path = os.path.join(outdir, "METRICS.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Few-shot Triplet — Evaluation Metrics\n")
        f.write(f"Shots per class: {args.shots_per_class}\n")
        f.write(f"Train domain:    {args.train_domain}\n")
        f.write(f"Eval domain:     {args.eval_domain}\n")
        f.write(f"Model:           {args.model}\n")
        f.write(f"Images evaluated: {metrics['overall']['total']}\n\n")
        f.write("Overall:\n")
        f.write(f"  Top-1: {metrics['overall']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['overall']['top5']:.2f}%\n\n")
        f.write("With Pair:\n")
        f.write(f"  Count: {metrics['with_pair']['total']}\n")
        f.write(f"  Top-1: {metrics['with_pair']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['with_pair']['top5']:.2f}%\n\n")
        f.write("Without Pair:\n")
        f.write(f"  Count: {metrics['without_pair']['total']}\n")
        f.write(f"  Top-1: {metrics['without_pair']['top1']:.2f}%\n")
        f.write(f"  Top-5: {metrics['without_pair']['top5']:.2f}%\n")
        f.write(f"\nCharts:\n  {bar_top1_png}\n  {bar_top5_png}\n  {per_class_png}\n  {conf_png}\n")

    # JSON/MD summary
    summary = {
        "mode": "fewshot_triplet",
        "shots_per_class": args.shots_per_class,
        "domains": {"train": args.train_domain, "eval": args.eval_domain},
        "model": args.model,
        "num_classes": len(class_ids),
        "num_images_evaluated": int(metrics["overall"]["total"]),
        "overall": {k: (float(v) if isinstance(v, (int,float)) else v) for k,v in metrics["overall"].items()},
        "with_pair": {k: (float(v) if isinstance(v, (int,float)) else v) for k,v in metrics["with_pair"].items()},
        "without_pair": {k: (float(v) if isinstance(v, (int,float)) else v) for k,v in metrics["without_pair"].items()},
        "elapsed_seconds": round(float(t1 - t0), 2),
        "per_class_png": per_class_png,
        "confusion_png": conf_png,
        "per_image_csv": per_img_csv,
        "metrics_bar_top1_png": bar_top1_png,
        "metrics_bar_top5_png": bar_top5_png,
        "metrics_txt": txt_path,
        "head_ckpt": args.head_ckpt,
    }
    with open(os.path.join(outdir, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("# Few-shot Triplet Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")

    print(f"✅ Report saved to: {outdir}")
    print(f"🎯 Overall Top-1: {metrics['overall']['top1']:.2f}%  |  🏅 Overall Top-5: {metrics['overall']['top5']:.2f}%")
    print(f"🔗 With Pair — Top-1: {metrics['with_pair']['top1']:.2f}%  |  Top-5: {metrics['with_pair']['top5']:.2f}%  |  N={metrics['with_pair']['total']}")
    print(f"🧩 Without Pair — Top-1: {metrics['without_pair']['top1']:.2f}%  |  Top-5: {metrics['without_pair']['top5']:.2f}%  |  N={metrics['without_pair']['total']}")
    print(f"📝 TXT: {txt_path}")
    print(f"🖼️ Bars: {bar_top1_png}, {bar_top5_png}")

if __name__ == "__main__":
    main()
