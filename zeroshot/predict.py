# zeroshot/predict.py
from __future__ import annotations
import os, argparse, torch
from PIL import Image
import torch.nn.functional as F
from .config import DEFAULT_MODEL, DEFAULT_PRETRAINED, DEFAULT_TEMPLATES
from .prompts import TEMPLATES_ALL, build_prompts
from .data import build_class_mapping_from_folders, load_id_to_name_csv
from .cache import hash_config, load_text_checkpoint, save_text_checkpoint
from .clip_model import load_openclip, encode_text_classes

def main():
    ap = argparse.ArgumentParser("Zero-shot single-image prediction")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--image", required=True, help="Path to a single JPG/PNG")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--templates", type=int, default=DEFAULT_TEMPLATES)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--use-names-csv", action="store_true")
    ap.add_argument("--ckpt-dir", default="zero_shot_checkpoints")
    args = ap.parse_args()

    device = torch.device(args.device)
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)
    id2name = load_id_to_name_csv(args.data_root) if args.use_names_csv else {}

    model, preprocess, tokenizer = load_openclip(args.model, args.pretrained, device)

    templates = TEMPLATES_ALL[:max(1, min(args.templates, len(TEMPLATES_ALL)))]
    prompts = build_prompts(class_ids, id2name, templates)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    sig = hash_config(args.model, args.pretrained, templates, class_ids)
    ckpt_path = os.path.join(args.ckpt_dir, f"text_feats_{sig}.pt")

    cached = load_text_checkpoint(ckpt_path)
    if cached is not None and "text_cls_feats" in cached:
        text_cls_feats = cached["text_cls_feats"].to(device)
        print(f"🧠 Using cached text features (C={text_cls_feats.shape[0]})")
    else:
        print(f"🧠 Encoding {len(prompts)} prompts …")
        text_cls_feats = encode_text_classes(model, tokenizer, device, prompts, len(class_ids), bs=32)
        save_text_checkpoint(ckpt_path, text_cls_feats, {"model": args.model, "pretrained": args.pretrained})

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    z = F.normalize(model.encode_image(x), dim=-1)
    logits = z @ text_cls_feats.T
    vals, idxs = logits.topk(min(args.topk, len(class_ids)), dim=-1)
    idxs = idxs[0].tolist()
    vals = vals[0].tolist()

    print("\n🔮 Top predictions:")
    for rank, (gi, score) in enumerate(zip(idxs, vals), start=1):
        cid = class_ids[gi]
        cname = id2name.get(cid, cid)
        print(f"{rank:>2}. {cname}  (id={cid}, global_idx={gi})  score={score:.4f}")

if __name__ == "__main__":
    main()
