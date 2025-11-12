# zeroshot/main.py
from __future__ import annotations
import os, json, time, argparse, torch
from .config import (
    DEFAULT_MODEL, DEFAULT_IMG_BATCH, DEFAULT_MAX_PER_CLASS,
    DEFAULT_TEMPLATES, DEFAULT_TEXT_BATCH, DEFAULT_PRETRAINED
)
from .data import (
    build_class_mapping_from_folders, load_id_to_name_csv,
    collect_images, collect_class_images_dict
)
from .dino_model import load_dino, build_class_prototypes
from .cache import hash_config, save_text_checkpoint, load_text_checkpoint
from .evaluate import evaluate_batched
from .plots import plot_per_class_accuracy, plot_confusion

def main():
    p = argparse.ArgumentParser("Zero-shot (prototype) with DINOv2")
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--domain", default="photo", choices=["photo", "herbarium"],
                   help="Which split to EVALUATE (prototypes always from herbarium)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="timm model id for DINOv2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-batch",  type=int, default=DEFAULT_IMG_BATCH)
    p.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
                   help="Max herbarium images per class to build prototypes")
    # outputs / extras
    p.add_argument("--save-confusion", action="store_true")
    p.add_argument("--save-mistakes", action="store_true")
    p.add_argument("--mistakes-limit", type=int, default=200)
    p.add_argument("--summary-json", default="zero_shot_results/summary_${domain}.json")
    p.add_argument("--summary-md",   default="zero_shot_results/summary_${domain}.md")
    p.add_argument("--acc-chart",    default="zero_shot_results/acc_per_class_${domain}.png")
    p.add_argument("--confusion-png", default="zero_shot_results/confusion_${domain}.png")
    p.add_argument("--ckpt-dir", default="zero_shot_checkpoints")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Classes
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)
    print(f"🔢 Evaluating domain: {args.domain}")

    # Load DINOv2 model + preprocess
    model, preprocess = load_dino(args.model, device)

    # ===== Build or load prototypes from HERBARIUM images =====
    src_domain = "herbarium"
    class_to_paths = collect_class_images_dict(args.data_root, src_domain)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    # hash includes model + class_ids + max_per_class to keep caches separate
    sig = hash_config(args.model, "dino_proto", [f"max={args.max_per_class}"], class_ids)
    proto_ckpt = os.path.join(args.ckpt_dir, f"prototypes_{sig}.pt")

    cached = load_text_checkpoint(proto_ckpt)  # reuse same simple IO (tensor + meta)
    if cached is not None and "text_cls_feats" in cached:
        prototypes = cached["text_cls_feats"].to(device)
        print(f"🧠 Using cached prototypes (C={prototypes.shape[0]}).")
    else:
        print(f"🧠 Building class prototypes from herbarium (≤{args.max_per_class} imgs/class)…")
        t0 = time.time()
        prototypes = build_class_prototypes(
            model, preprocess, device,
            class_to_paths=class_to_paths,
            class_ids_in_order=class_ids,
            max_per_class=args.max_per_class,
            img_batch=max(1, args.img_batch),
        )
        meta = {"model": args.model, "num_classes": len(class_ids),
                "src_domain": src_domain, "max_per_class": args.max_per_class}
        save_text_checkpoint(proto_ckpt, prototypes, meta)
        print(f"⏱️  Prototype build took {time.time()-t0:.2f}s")

    # ===== Collect evaluation samples (target domain) =====
    print(f"🖼️ Scanning images in: train/{args.domain}")
    samples = collect_images(args.data_root, args.domain)

    # --- FIX: evaluate_batched expects a list of image PATHS (strings), not (path, class_id) tuples ---
    image_paths = [p for p, _ in samples]  # <--- key fix

    # ===== Evaluate (cosine sim to prototypes) =====
    print("🚀 Running prototype zero-shot inference …")
    t0 = time.time()
    # We pass `prototypes` in place of "text_cls_feats" and set use_logit_scale=False
    top1, top5, n, conf, details = evaluate_batched(
        model, preprocess, device,
        image_paths, cls2idx, prototypes,
        img_batch=max(1, args.img_batch),
        use_logit_scale=False,
        want_confusion=True,
        return_details=True,
    )
    t1 = time.time()

    print(f"\n✅ Done. Domain={args.domain} | Images evaluated={n}")
    print(f"🎯 Top-1: {top1:.2f}%")
    print(f"🏅 Top-5: {top5:.2f}%")
    print(f"⏱️  Elapsed: {t1 - t0:.2f}s")

    # ===== Outputs =====
    os.makedirs("zero_shot_results", exist_ok=True)
    sum_json = args.summary_json.replace("${domain}", args.domain)
    sum_md   = args.summary_md.replace("${domain}", args.domain)
    acc_png  = args.acc_chart.replace("${domain}", args.domain)
    conf_png = args.confusion_png.replace("${domain}", args.domain)

    if conf is not None:
        plot_per_class_accuracy(conf, acc_png, class_ids)
        if args.save_confusion:
            plot_confusion(conf, conf_png)

    if args.save_mistakes:
        import shutil
        errs_dir = os.path.join("zero_shot_results", f"errors_{args.domain}")
        os.makedirs(errs_dir, exist_ok=True)
        copied = 0
        for path, t_idx, p_idx, top5_idx, top1_score in details:
            if p_idx != t_idx:
                try:
                    base = os.path.basename(path)
                    out = os.path.join(errs_dir, f"wrong_t{t_idx}_p{p_idx}__{base}")
                    shutil.copy2(path, out)
                    copied += 1
                    if copied >= max(1, args.mistakes_limit):
                        break
                except Exception as e:
                    print(f"⚠️  Could not copy {path}: {e}")
        print(f"📁 Misclassified images copied: {copied} → {errs_dir}")

    summary = {
        "mode": "dino_prototype_zero_shot",
        "domain": args.domain,
        "model": args.model,
        "num_classes": len(class_ids),
        "num_images_evaluated": int(n),
        "top1": round(top1, 4), "top5": round(top5, 4),
        "elapsed_seconds": round(t1 - t0, 2),
        "prototype_ckpt": proto_ckpt,
        "acc_chart": acc_png if conf is not None else "",
        "confusion_png": conf_png if conf is not None and os.path.exists(conf_png) else "",
    }
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(sum_md, "w", encoding="utf-8") as f:
        f.write("# Prototype Zero-shot (DINOv2) Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
    print(f"🧾 Saved summary JSON: {sum_json}")
    print(f"📝 Saved summary Markdown: {sum_md}")

if __name__ == "__main__":
    main()
