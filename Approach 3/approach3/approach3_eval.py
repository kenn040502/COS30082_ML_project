from __future__ import annotations
import argparse
from .eval_core import eval_main

def parse_args():
    p = argparse.ArgumentParser("Approach 3 eval – overall / with-pairs / without-pairs")
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", required=True, help="Path to best_model.pt from approach3_train")
    p.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="runs_approach3_report")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval_main(args)
