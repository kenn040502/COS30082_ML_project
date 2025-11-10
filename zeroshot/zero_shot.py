# zero_shot.py
# Self-contained zero-shot classifier using OpenCLIP
# Works with folder structure:
#   <data_root>/train/herbarium/<class_id>/*.jpg
#   <data_root>/train/photo/<class_id>/*.jpg
#
# Optional: <data_root>/list/class_names.csv with headers:
#   class_id,name

from __future__ import annotations
import os, csv, argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image


# -------- Prompt templates (prompt ensembling) --------
TEMPLATES = [
    "a photo of {name}.",
    "a picture of {name}.",
    "herbarium specimen of {name}.",
    "field photo of {name}.",
    "plant species {name}.",
    "macro photo of {name}.",
]


# -------- Data helpers --------
def _list_class_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    # Class ids are folder names
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def build_class_mapping_from_folders(data_root: str) -> tuple[list[str], dict[str, int]]:
    herb = _list_class_dirs(os.path.join(data_root, "train", "herbarium"))
    photo = _list_class_dirs(os.path.join(data_root, "train", "photo"))
    union = sorted(set(herb) | set(photo))
    cls2idx = {cid: i for i, cid in enumerate(union)}
    print(f"📁 Found {len(union)} classes (union of herbarium & photo).")
    return union, cls2idx

def load_id_to_name_csv(data_root: str) -> dict[str, str]:
    """
    Optional file: <data_root>/list/class_names.csv
    CSV headers: class_id,name
    """
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
    """
    Scans <data_root>/train/<domain>/<class_id>/**/*.(jpg|png) and
    returns list of (image_path, class_id).
    """
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

def build_prompts(class_ids: list[str], id2name: dict[str, str]) -> list[str]:
    prompts: list[str] = []
    for cid in class_ids:
        name = id2name.get(cid, cid)  # if no CSV provided, use folder name
        for t in TEMPLATES:
            prompts.append(t.format(name=name))
    return prompts


# -------- Zero-shot core --------
@torch.no_grad()
def encode_text_classes(model, tokenizer, device, prompts: list[str], num_classes: int) -> torch.Tensor:
    import math
    feats = []
    bs = 64
    for i in range(0, len(prompts), bs):
        toks = tokenizer(prompts[i:i+bs]).to(device)
        z = model.encode_text(toks)
        feats.append(F.normalize(z, dim=-1))
    feats = torch.cat(feats, 0)  # [C*T, D]

    T = len(TEMPLATES)
    if num_classes * T != feats.shape[0]:
        raise RuntimeError(f"Prompt count mismatch: classes={num_classes}, T={T}, total={feats.shape[0]}")

    # Average prompts per class
    per_class = []
    for ci in range(num_classes):
        per_class.append(feats[ci*T:(ci+1)*T].mean(0))
    per_class = torch.stack(per_class, 0)  # [C, D]
    return F.normalize(per_class, dim=-1)

@torch.no_grad()
def evaluate(
    model, preprocess, device,
    samples: list[tuple[str, str]],
    cls2idx: dict[str, int],
    text_cls_feats: torch.Tensor,
    csv_out: Optional[str] = None,
) -> tuple[float, float, int]:
    writer = None
    fcsv = None
    if csv_out:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        fcsv = open(csv_out, "w", encoding="utf-8", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["path", "true_class_id", "pred_top1_global_idx", "top1_score", "top5_global_idxs"])

    total = top1 = top5 = 0
    for path, cid in samples:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Skipping {path}: {e}")
            continue

        x = preprocess(img).unsqueeze(0).to(device)
        z = F.normalize(model.encode_image(x), dim=-1)  # [1, D]
        logits = z @ text_cls_feats.T                # [1, C]

        vals, idxs = logits.topk(5, dim=-1)
        pred1 = idxs[0, 0].item()
        top5set = set(idxs[0].tolist())

        true_idx = cls2idx.get(cid)  # None if not in mapping
        if true_idx is None:
            continue

        total += 1
        if pred1 == true_idx:
            top1 += 1
        if true_idx in top5set:
            top5 += 1

        if writer is not None:
            writer.writerow([
                path,
                cid,
                pred1,
                float(vals[0, 0].item()),
                " ".join(map(str, idxs[0].tolist()))
            ])

    if fcsv is not None:
        fcsv.close()

    top1_acc = 100.0 * top1 / max(1, total)
    top5_acc = 100.0 * top5 / max(1, total)
    return top1_acc, top5_acc, total


# -------- CLI --------
def main():
    parser = argparse.ArgumentParser("Zero-shot image classification with OpenCLIP")
    parser.add_argument("--data-root", required=True, type=str, help="Root of dataset")
    parser.add_argument("--domain", default="photo", choices=["photo", "herbarium"], help="Which domain to evaluate")
    parser.add_argument("--model", default="ViT-B-16", help="OpenCLIP model name (e.g., ViT-B-32, ViT-L-14)")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k", help="OpenCLIP pretrained tag")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-csv", default="zero_shot_results/preds_${domain}.csv")
    parser.add_argument("--use-names-csv", action="store_true",
                        help="Use <data_root>/list/class_names.csv to map IDs to names in prompts.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1) Build class mapping from folders
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)

    # 2) Optional: better names for prompts
    id2name = load_id_to_name_csv(args.data_root) if args.use_names_csv else {}

    # 3) Load OpenCLIP + preprocess + tokenizer
    import open_clip
    print(f"🔤 Loading OpenCLIP: {args.model} ({args.pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # 4) Build and encode prompts
    prompts = build_prompts(class_ids, id2name)
    print(f"🧠 Encoding {len(prompts)} prompts for {len(class_ids)} classes …")
    text_cls_feats = encode_text_classes(model, tokenizer, device, prompts, len(class_ids))

    # 5) Collect images & evaluate
    print(f"🖼️ Scanning images in: train/{args.domain}")
    samples = collect_images(args.data_root, args.domain)

    out_csv = args.results_csv.replace("${domain}", args.domain) if args.results_csv else None
    print("🚀 Running zero-shot inference …")
    top1, top5, n = evaluate(model, preprocess, device, samples, cls2idx, text_cls_feats, csv_out=out_csv)

    print(f"\n✅ Done. Domain={args.domain} | Images evaluated={n}")
    print(f"🎯 Top-1: {top1:.2f}%")
    print(f"🏅 Top-5: {top5:.2f}%")
    if out_csv:
        print(f"📝 Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
