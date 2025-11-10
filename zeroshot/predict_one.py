# zeroshot/predict_one.py
from __future__ import annotations
import os, argparse, torch
import torch.nn.functional as F
from PIL import Image

from .dino_model import load_dino
from .data import build_class_mapping_from_folders, load_id_to_name_csv
from .cache import load_text_checkpoint

@torch.no_grad()
def _encode_any(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):      # CLIP-style
        z = model.encode_image(x)
    else:                                   # timm (DINOv2)
        z = model(x)
        if z.dim() > 2:
            z = z.mean(dim=tuple(range(2, z.dim())))
    return F.normalize(z, dim=-1)

def _infer_truth_from_path(data_root: str, image_path: str) -> tuple[str|None, str|None]:
    """
    Try to infer (domain, class_id) from a path like:
      <root>/train/photo/123/xxx.jpg  -> ("photo", "123")
      <root>/train/herbarium/456/y.jpg -> ("herbarium", "456")
    Returns (domain, class_id) or (None, None) if not under expected tree.
    """
    rp = os.path.relpath(os.path.abspath(image_path), os.path.abspath(data_root))
    parts = rp.replace("\\", "/").split("/")
    # expect ["train", "<domain>", "<class_id>", ...]
    if len(parts) >= 3 and parts[0] == "train" and parts[1] in ("photo", "herbarium"):
        return parts[1], parts[2]
    return None, None

def main():
    ap = argparse.ArgumentParser("Predict one image with DINOv2 prototypes")
    ap.add_argument("--data-root", required=True, help="Dataset root containing train/herbarium and train/photo")
    ap.add_argument("--image", required=True, help="Absolute or relative path to the image to test")
    ap.add_argument("--proto-file", required=True, help="Path to cached prototypes (e.g. zero_shot_checkpoints/prototypes_xxx.pt)")
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, preprocess = load_dino(args.model, device)

    # Class order (must match what prototypes were built with — i.e., current folder union)
    class_ids, cls2idx = build_class_mapping_from_folders(args.data_root)

    ck = load_text_checkpoint(args.proto_file)
    if ck is None or "text_cls_feats" not in ck:
        raise FileNotFoundError(f"Could not load prototypes from: {args.proto_file}")
    prototypes = ck["text_cls_feats"].to(device)   # [C,D]
    C = prototypes.shape[0]
    if C != len(class_ids):
        raise RuntimeError(
            f"Prototype count ({C}) != number of classes found now ({len(class_ids)}). "
            f"Rebuild prototypes (run zeroshot.main) or ensure classes haven’t changed."
        )

    # Optional friendly names
    id2name = load_id_to_name_csv(args.data_root)   # empty if file not present

    # Load and encode image
    img_path = os.path.abspath(args.image)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)     # [1,3,H,W]
    z = _encode_any(model, x)                       # [1,D]
    logits = z @ prototypes.T                       # [1,C]

    vals, idxs = logits.topk(k=max(1, args.topk), dim=-1)
    vals = vals.squeeze(0).cpu().tolist()
    idxs = idxs.squeeze(0).cpu().tolist()

    # Show predictions
    print(f"\n🔍 Image: {img_path}")
    print("🏁 Top predictions:")
    for rank, (j, s) in enumerate(zip(idxs, vals), start=1):
        cid = class_ids[j]
        cname = id2name.get(cid, "")
        label = f"{cid} ({cname})" if cname else cid
        print(f"  {rank:>2}. {label:<24}  score={s:.4f}")

    # Try to infer ground-truth from path
    dom, gt_cid = _infer_truth_from_path(args.data_root, img_path)
    if gt_cid is not None:
        print(f"\n🧾 Inferred ground-truth: domain={dom}, class_id={gt_cid}")
        pred_top1_cid = class_ids[idxs[0]]
        correct = (pred_top1_cid == gt_cid)
        print(f"🎯 Top-1 {'CORRECT' if correct else 'WRONG'} → predicted={pred_top1_cid}, gt={gt_cid}")
    else:
        print("\nℹ️  Ground-truth not inferred (image not under train/<domain>/<class_id>/...).")

if __name__ == "__main__":
    main()
