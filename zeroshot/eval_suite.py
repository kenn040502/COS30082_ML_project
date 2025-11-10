# zeroshot/eval_suite.py
from __future__ import annotations
import os, json, csv, time, argparse, pathlib
from typing import Dict, List, Tuple, Set
import torch
import numpy as np
import matplotlib.pyplot as plt

from .dino_model import load_dino
from .cache import load_text_checkpoint
from .data import (
    build_class_mapping_from_folders,
    load_id_to_name_csv,
    collect_images,
)
from .evaluate import evaluate_batched  # must be the version in this message

# Paired tools (optional; used only if pairs.csv exists)
try:
    from .paired_data import load_pairs_csv
    from .paired_eval import paired_retrieval_eval, paired_per_query_ranks_csv
    HAVE_PAIRED = True
except Exception:
    HAVE_PAIRED = False


# ---------- small utilities ----------

def _ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _subset_details_by_class(details, class_indices: Set[int]):
    """Filter evaluation details to a subset of classes (by true label)."""
    return [d for d in details if d[1] in class_indices]

def _topk_overall(details, k: int) -> float:
    """
    details: list of (path, true_idx, pred_idx, topk_idxs, topk_scores)
    returns overall top-k accuracy in percent
    """
    if not details:
        return 0.0
    correct = 0
    for _, t, p, topk, _scores in details:
        if k == 1:
            ok = (p == t)
        else:
            ok = (t in topk[:k])
        if ok:
            correct += 1
    return 100.0 * correct / len(details)

def _per_class_topk(details, num_classes: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    returns: (acc[C], totals[C]) for per-class top-k accuracy (%)
    """
    correct = np.zeros(num_classes, dtype=np.int32)
    total   = np.zeros(num_classes, dtype=np.int32)
    for _, t, p, topk, _scores in details:
        if 0 <= t < num_classes:
            total[t] += 1
            ok = (p == t) if k == 1 else (t in topk[:k])
            if ok:
                correct[t] += 1
    acc = np.zeros(num_classes, dtype=np.float32)
    mask = total > 0
    acc[mask] = 100.0 * correct[mask] / total[mask]
    return acc, total

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _save_csv_per_class_top123(path, class_ids, id2name, acc1, acc2, acc3, totals):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "top1(%)", "top2(%)", "top3(%)", "num_samples"])
        for i, cid in enumerate(class_ids):
            w.writerow([
                cid, id2name.get(cid, ""),
                round(float(acc1[i]), 4),
                round(float(acc2[i]), 4),
                round(float(acc3[i]), 4),
                int(totals[i]),
            ])

def _save_per_image_top3_csv(path, details, class_ids, id2name):
    """
    Save per-image rows:
    image_path, true_class_id, true_name,
    pred1_id, pred1_name, score1,
    pred2_id, pred2_name, score2,
    pred3_id, pred3_name, score3,
    top1_correct
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path", "true_class_id", "true_name",
            "pred1_id", "pred1_name", "score1",
            "pred2_id", "pred2_name", "score2",
            "pred3_id", "pred3_name", "score3",
            "top1_correct",
        ])
        for img_path, t, p, topk, scores in details:
            # class ids
            true_cid = class_ids[t] if 0 <= t < len(class_ids) else ""
            true_name = id2name.get(true_cid, "")

            def pack(j, score):
                if j is None or j < 0 or j >= len(class_ids):
                    return ("", "", "")
                cid = class_ids[j]
                cname = id2name.get(cid, "")
                return (cid, cname, f"{float(score):.6f}")

            # we requested top-5 inside evaluator; use the first three for this CSV
            j1 = p
            s1 = scores[0] if scores else ""
            j2 = topk[1] if len(topk) > 1 else -1
            s2 = scores[1] if len(scores) > 1 else ""
            j3 = topk[2] if len(topk) > 2 else -1
            s3 = scores[2] if len(scores) > 2 else ""

            pred1_id, pred1_name, score1 = pack(j1, s1)
            pred2_id, pred2_name, score2 = pack(j2, s2)
            pred3_id, pred3_name, score3 = pack(j3, s3)

            w.writerow([
                img_path, true_cid, true_name,
                pred1_id, pred1_name, score1,
                pred2_id, pred2_name, score2,
                pred3_id, pred3_name, score3,
                "1" if (p == t) else "0",
            ])

def _plot_bar_per_class(path, acc, class_ids, id2name, title):
    plt.figure(figsize=(12, 5))
    xs = np.arange(len(acc))
    plt.bar(xs, acc)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    if len(acc) <= 60:
        labels = [id2name.get(class_ids[i], class_ids[i]) for i in range(len(acc))]
        plt.xticks(xs, labels, rotation=90, fontsize=6)
    else:
        plt.xticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _plot_confusion(path, conf, title):
    # pure matplotlib (no seaborn dependency)
    plt.figure(figsize=(6, 5))
    conf = conf.astype(np.float32)
    row_sums = conf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    conf_norm = conf / row_sums
    plt.imshow(conf_norm, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- unpaired evaluation for a domain ----------

def evaluate_unpaired_for_domain(
    model, preprocess, device,
    data_root: str,
    domain: str,
    class_ids: List[str],
    cls2idx: Dict[str,int],
    text_cls_feats: torch.Tensor,
    img_batch: int,
    outdir: str,
    id2name: Dict[str,str],
):
    # Collect image paths
    samples = collect_images(data_root, split="train", domain=domain)

    # Evaluate (our evaluate_batched returns per-image details incl. top5 indices & scores)
    top1, top5, total, conf, details = evaluate_batched(
        model, preprocess, device, samples, cls2idx, text_cls_feats,
        img_batch=img_batch, use_logit_scale=False, want_confusion=True, return_details=True
    )

    # Top-1/2/3 overall
    top1_overall = _topk_overall(details, 1)
    top2_overall = _topk_overall(details, 2)
    top3_overall = _topk_overall(details, 3)

    # Per-class Top-1/2/3
    C = len(class_ids)
    acc1, totals = _per_class_topk(details, C, 1)
    acc2, _      = _per_class_topk(details, C, 2)
    acc3, _      = _per_class_topk(details, C, 3)

    # Save summaries and artifacts
    summary = {
        "domain": domain,
        "images_evaluated": int(total),
        "top1_percent": round(float(top1_overall), 4),
        "top2_percent": round(float(top2_overall), 4),
        "top3_percent": round(float(top3_overall), 4),
        "top5_percent_legacy": round(float(top5), 4),  # keep the original top-5 for reference
    }
    _save_json(os.path.join(outdir, f"unpaired_{domain}_summary.json"), summary)

    _save_per_image_top3_csv(
        os.path.join(outdir, f"unpaired_{domain}_per_image_top3.csv"),
        details, class_ids, id2name
    )
    _save_csv_per_class_top123(
        os.path.join(outdir, f"unpaired_{domain}_per_class_top123.csv"),
        class_ids, id2name, acc1, acc2, acc3, totals
    )
    _plot_bar_per_class(
        os.path.join(outdir, f"unpaired_{domain}_per_class_top1.png"),
        acc1, class_ids, id2name, f"Unpaired Top-1 accuracy per class — {domain}"
    )
    if conf is not None:
        _plot_confusion(
            os.path.join(outdir, f"unpaired_{domain}_confusion.png"),
            conf, f"Confusion — {domain} (unpaired)"
        )

    return {
        "summary": summary,
        "details": details,            # keep for subsets later
        "per_class_acc": acc1,         # Top-1 per-class
        "per_class_totals": totals,
        "confusion": conf,
    }


# ---------- main suite ----------

def main():
    ap = argparse.ArgumentParser("Full eval suite: unpaired + paired (optional) with pair/no-pair splits")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--proto-file", required=True, help="Prototypes checkpoint (.pt) from zeroshot.main")
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img-batch", type=int, default=8)
    ap.add_argument("--outdir", default=None, help="Output dir (default: zero_shot_results/report_<timestamp>)")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = args.outdir or os.path.join("zero_shot_results", f"report_{_timestamp()}")
    _ensure_dir(outdir)

    # Model + preprocess
    model, preprocess = load_dino(args.model, device)

    # Class order must match prototypes
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)
    id2name = load_id_to_name_csv(args.data_root)

    ck = load_text_checkpoint(args.proto_file)
    if ck is None or "text_cls_feats" not in ck:
        raise FileNotFoundError(f"Could not load prototypes from: {args.proto_file}")
    prototypes = ck["text_cls_feats"].to(device)  # [C,D]
    if prototypes.shape[0] != len(class_ids):
        raise RuntimeError(
            f"Prototype count ({prototypes.shape[0]}) != classes found now ({len(class_ids)}). "
            "Rebuild prototypes with zeroshot.main so they align."
        )

    # Unpaired eval for herbarium & photo
    results = {}
    for domain in ("herbarium", "photo"):
        print(f"\n=== Unpaired evaluation: {domain} ===")
        results[domain] = evaluate_unpaired_for_domain(
            model, preprocess, device,
            args.data_root, domain,
            class_ids, cls2idx, prototypes,
            img_batch=max(1, args.img_batch),
            outdir=outdir,
            id2name=id2name,
        )

    # Paired (optional) — requires zeroshot/pairs.csv and paired modules available
    pairs_path = os.path.join(args.data_root, "list", "pairs.csv")
    have_pairs_file = os.path.isfile(pairs_path) and HAVE_PAIRED
    paired_summary = {}

    if have_pairs_file:
        print("\n=== Paired retrieval evaluation (photo -> herbarium) ===")
        pairs = load_pairs_csv(args.data_root)

        # Overall paired metrics
        paired_overall = paired_retrieval_eval(
            model, preprocess, device,
            data_root=args.data_root, pairs=pairs,
            img_batch=max(1, args.img_batch),
        )
        _save_json(os.path.join(outdir, "paired_overall.json"), paired_overall)
        paired_summary["overall"] = paired_overall

        # Also save per-query ranks CSV (shows rank of the paired herb for each photo)
        paired_csv_path = os.path.join(outdir, "paired_per_query_ranks.csv")
        paired_per_query_ranks_csv(
            model, preprocess, device,
            data_root=args.data_root, pairs=pairs,
            img_batch=max(1, args.img_batch),
            out_csv=paired_csv_path
        )

        # Split unpaired photo accuracy for classes with vs. without pairs
        classes_with_pairs: Set[str] = {cid for _, _, cid in pairs if cid}
        classes_with_pairs_idx = {cls2idx[cid] for cid in classes_with_pairs if cid in cls2idx}
        classes_without_pairs_idx = set(range(len(class_ids))) - classes_with_pairs_idx

        photo_details = results["photo"]["details"]
        subset_with = _subset_details_by_class(photo_details, classes_with_pairs_idx)
        subset_without = _subset_details_by_class(photo_details, classes_without_pairs_idx)

        def _acc_from_subset(details_subset):
            if not details_subset:
                return 0.0, 0
            correct = sum(int(p == t) for _, t, p, *_ in details_subset)
            total = len(details_subset)
            return 100.0 * correct / max(1, total), total

        with_top1, with_total = _acc_from_subset(subset_with)
        wout_top1, wout_total = _acc_from_subset(subset_without)

        splits = {
            "classes_with_pairs_count": len(classes_with_pairs_idx),
            "classes_without_pairs_count": len(classes_without_pairs_idx),
            "photo_unpaired_top1_with_pairs(%)": round(with_top1, 4),
            "photo_unpaired_samples_with_pairs": with_total,
            "photo_unpaired_top1_without_pairs(%)": round(wout_top1, 4),
            "photo_unpaired_samples_without_pairs": wout_total,
        }
        _save_json(os.path.join(outdir, "paired_splits_unpaired_photo.json"), splits)
        paired_summary["splits_on_unpaired_photo"] = splits

    # Global summary
    global_summary = {
        "unpaired": { dom: results[dom]["summary"] for dom in ("herbarium", "photo") },
        "paired": paired_summary if have_pairs_file else "pairs.csv not found (skipped)",
    }
    _save_json(os.path.join(outdir, "GLOBAL_SUMMARY.json"), global_summary)
    print(f"\n✅ Saved full report to: {outdir}")

if __name__ == "__main__":
    main()
