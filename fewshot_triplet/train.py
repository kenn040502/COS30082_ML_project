# fewshot_triplet/train.py
from __future__ import annotations
import os, time, argparse, json, random, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .config import DEFAULT_MODEL, DEFAULT_BATCH_TRAIN, DEFAULT_EPOCHS, DEFAULT_MARGIN, DEFAULT_LR, DEFAULT_SHOTS
from .backbone import load_backbone, encode_images, estimate_out_dim
from .data import split_fewshot, make_loaders

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
    def forward(self, z): return F.normalize(self.net(z), dim=-1)

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def batch_hard_triplets(emb, y):
    with torch.no_grad():
        S = emb @ emb.t()  # cosine similarities
        y = y.view(-1,1)
        eq = (y == y.t())
        ne = ~eq
        pos_sim = S.clone(); pos_sim[~eq] = -1e9; pos_sim.fill_diagonal_(-1e9)
        p_idx = pos_sim.argmax(dim=1)
        neg_sim = S.clone(); neg_sim[~ne] = -1e9
        n_idx = neg_sim.argmax(dim=1)
        a_idx = torch.arange(emb.shape[0], device=emb.device)
    return a_idx, p_idx, n_idx

def triplet_loss(emb, a_idx, p_idx, n_idx, margin=0.2):
    ap = (emb[a_idx] * emb[p_idx]).sum(dim=-1)
    an = (emb[a_idx] * emb[n_idx]).sum(dim=-1)
    return F.relu(margin - (ap - an)).mean()

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
        z = encode_images(backbone, device, xb)  # [B, D]
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
def quick_eval(backbone, head, ds_train, ds_eval, device):
    """Evaluate Top-1/Top-5 using class-means built from few-shot train set."""
    from torch.utils.data import DataLoader
    means = build_class_means(backbone, head, ds_train, device)  # [C,256]
    dl = DataLoader(ds_eval, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    C = means.shape[0]
    total = top1 = top5 = 0
    for xb, yb, _ in dl:
        xb, yb = xb.to(device), yb.to(device)
        z = encode_images(backbone, device, xb)
        f = head(z)                          # [B,256]
        sims = f @ means.t()                 # [B,C]
        vals, idxs = sims.topk(k=min(5, C), dim=-1)
        preds = idxs[:, 0]
        for i in range(xb.shape[0]):
            t = int(yb[i].item()); p = int(preds[i].item())
            total += 1
            if p == t: top1 += 1
            if t in idxs[i].tolist(): top5 += 1
    top1_pct = 100.0 * top1 / max(1, total)
    top5_pct = 100.0 * top5 / max(1, total)
    return top1_pct, top5_pct, int(total)

def _save_leaderboard(outdir, leaderboard):
    os.makedirs(outdir, exist_ok=True)
    # JSON
    with open(os.path.join(outdir, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    # TXT
    lines = ["Few-shot Triplet — Leaderboard (ranked by Overall Top-1 on eval domain)\n"]
    for r in leaderboard:
        lines.append(f"#{r['rank']:>1}  epoch={r['epoch']:>2}  top1={r['top1']:.2f}%  top5={r['top5']:.2f}%  ckpt={r['ckpt']}")
    with open(os.path.join(outdir, "LEADERBOARD.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def _maybe_update_top3(outdir, epoch, top1, top5, head_state_dict, backbone_dim):
    """Maintain top-3 by top1. Saves copies under outdir/ranked/."""
    ranked_dir = os.path.join(outdir, "ranked")
    os.makedirs(ranked_dir, exist_ok=True)
    # Load current leaderboard if any
    lb_path = os.path.join(outdir, "leaderboard.json")
    if os.path.isfile(lb_path):
        with open(lb_path, "r", encoding="utf-8") as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Save a temp ckpt for this epoch
    ckpt_name = f"head_ep{epoch}.pt"
    ckpt_path = os.path.join(outdir, ckpt_name)
    torch.save({"head": head_state_dict, "meta": {"epoch": epoch, "embedding_dim": 256}}, ckpt_path)

    leaderboard.append({"epoch": epoch, "top1": float(top1), "top5": float(top5), "ckpt": ckpt_path})
    # Sort by Top-1 desc, then Top-5 desc, then lower epoch wins tie
    leaderboard.sort(key=lambda r: (-r["top1"], -r["top5"], r["epoch"]))
    # Keep only top-3
    leaderboard = leaderboard[:3]

    # Save rank copies
    for i, r in enumerate(leaderboard, start=1):
        rank_path = os.path.join(ranked_dir, f"head_rank{i}.pt")
        shutil.copy2(r["ckpt"], rank_path)
        r["rank"] = i
        r["rank_path"] = rank_path

    _save_leaderboard(outdir, leaderboard)
    return leaderboard

def main():
    ap = argparse.ArgumentParser("Few-shot Triplet - Train projection head (frozen backbone) + rank Top-3 checkpoints")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--shots-per-class", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--train-domain", default="herbarium")
    ap.add_argument("--eval-domain",  default="photo")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-train", type=int, default=DEFAULT_BATCH_TRAIN)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="fewshot_results/run_${shots}")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    backbone, preprocess = load_backbone(args.model, device)

    shots, eval_samples, cls2idx, class_ids = split_fewshot(
        args.data_root, args.shots_per_class, args.train_domain, args.eval_domain
    )
    ds_train, ds_eval, dl_train, _ = make_loaders(
        shots, eval_samples, cls2idx, preprocess,
        batch_train=max(4, args.batch_train), batch_eval=64
    )

    d_backbone = estimate_out_dim(backbone, device)
    head = ProjectionHead(d_backbone, out_dim=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    outdir = args.outdir.replace("${shots}", f"{args.shots_per_class}shot")
    os.makedirs(outdir, exist_ok=True)
    print(f"🎯 Shots total: {len(shots)} across {len(class_ids)} classes")
    print(f"💾 Outputs will be in: {outdir}")

    backbone.eval()
    train_loss_last_epoch = None

    for ep in range(1, args.epochs+1):
        head.train()
        pbar = tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}")
        losses = []
        for xb, yb, _ in pbar:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                z = encode_images(backbone, device, xb)
            f = head(z)
            a_idx, p_idx, n_idx = batch_hard_triplets(f, yb)
            loss = triplet_loss(f, a_idx, p_idx, n_idx, margin=args.margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{sum(losses)/len(losses):.4f}")
        train_loss_last_epoch = sum(losses)/max(1,len(losses))

        # === quick eval & rank ===
        top1, top5, total = quick_eval(backbone, head.eval(), ds_train, ds_eval, device)
        leaderboard = _maybe_update_top3(outdir, ep, top1, top5, head.state_dict(), d_backbone)
        print(f"🧪 Epoch {ep}: Eval Top-1={top1:.2f}%  Top-5={top5:.2f}%  N={total}")
        if leaderboard:
            best = leaderboard[0]
            print(f"🏆 Current #1: epoch={best['epoch']}  Top-1={best['top1']:.2f}%  Top-5={best['top5']:.2f}%")

    # Save last head as well
    last_path = os.path.join(outdir, "fewshot_triplet_head.pt")
    torch.save({
        "head": head.state_dict(),
        "meta": {
            "model": args.model,
            "shots_per_class": args.shots_per_class,
            "train_domain": args.train_domain,
            "eval_domain": args.eval_domain,
            "embedding_dim": 256,
            "class_count": len(class_ids),
            "seed": args.seed,
            "train_loss_last_epoch": float(train_loss_last_epoch) if train_loss_last_epoch is not None else None,
        }
    }, last_path)
    # small training summary
    with open(os.path.join(outdir, "TRAIN_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump({"loss_last_epoch": train_loss_last_epoch}, f, indent=2)

    print("✅ Training finished.")
    print(f"📌 Last head: {last_path}")
    print(f"📚 Leaderboard & ranked copies are in: {os.path.join(outdir, 'leaderboard.json')} and {os.path.join(outdir, 'ranked')}")
    print("➡ You can now evaluate the best head (rank1) with:")
    print(f"   python -m fewshot_triplet.eval --data-root <root> --shots-per-class {args.shots_per_class} "
          f"--train-domain {args.train_domain} --eval-domain {args.eval_domain} "
          f"--head-ckpt \"{os.path.join(outdir, 'ranked', 'head_rank1.pt')}\" --outdir fewshot_results\\report_{args.shots_per_class}shot")
if __name__ == "__main__":
    main()
