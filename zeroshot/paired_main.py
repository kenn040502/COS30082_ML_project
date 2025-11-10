# zeroshot/paired_main.py
from __future__ import annotations
import argparse, torch
from .paired_data import load_pairs_csv
from .dino_model import load_dino
from .paired_eval import paired_retrieval_eval

def main():
    ap = argparse.ArgumentParser("Paired retrieval eval (DINOv2)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img-batch", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, preprocess = load_dino(args.model, device)

    pairs = load_pairs_csv(args.data_root)
    res = paired_retrieval_eval(
        model, preprocess, device,
        data_root=args.data_root,
        pairs=pairs,
        img_batch=max(1, args.img_batch),
    )
    print("\n📈 Paired retrieval metrics:")
    for k, v in res.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
