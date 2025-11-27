from __future__ import annotations
import argparse
from .train_loop import train_main, set_seed

def parse_args():
    p = argparse.ArgumentParser("Approach 3 - Domain-adaptive metric learning with DINOv2")
    p.add_argument("--data-root", required=True, help="Root of AML herbarium dataset")
    p.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--emb-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--w-triplet", type=float, default=0.5)
    p.add_argument("--w-da", type=float, default=0.1)
    p.add_argument("--da-lambda", type=float, default=1.0)
    p.add_argument("--outdir", default="runs_approach3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train_main(args)
