# Few-Shot + Triplet Loss (DINOv2 backbone)

This is a **self-contained** pipeline (separate from your `zeroshot` code) that:
- Uses a **frozen DINOv2** visual backbone from `timm`.
- Trains a small **projection head** with **Triplet Loss** on **K shots per class** (default K=5) from one domain (e.g., `herbarium`).
- Evaluates on another domain (e.g., `photo`) using **class-means** in the learned embedding space.
- Produces the **same style of outputs** you asked for: per-class accuracy PNG, confusion PNG, per-image Top-3 CSV, JSON and MD summaries.

## Data Layout (expected)

```
<data_root>/
  train/
    herbarium/<class_id>/*.jpg|png
    photo/<class_id>/*.jpg|png
```

Optional: you may add `test/<domain>/...` and point the evaluator to another folder via `--eval-split-path`, but by default it uses `train/photo` for evaluation.

## Quickstart (Windows PowerShell)

```powershell
# 0) Install deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm pillow matplotlib numpy tqdm

# 1) Paths
$ROOT = "C:\path\to\your\AML_project_herbarium_dataset"

# 2) Train a few-shot head with triplet loss (frozen DINOv2 backbone)
python -m fewshot_triplet.train `
  --data-root "$ROOT" `
  --shots-per-class 5 `
  --train-domain herbarium `
  --eval-domain  photo `
  --epochs 10 `
  --outdir "fewshot_results\run_5shot"

# 3) Evaluate and generate report (PNG charts, CSV, JSON/MD)
python -m fewshot_triplet.eval `
  --data-root "$ROOT" `
  --shots-per-class 5 `
  --train-domain herbarium `
  --eval-domain  photo `
  --head-ckpt  "fewshot_results\run_5shot\fewshot_triplet_head.pt" `
  --outdir     "fewshot_results\report_5shot"
```

## Outputs

- `fewshot_per_class_top1.png` — per-class Top-1 accuracy (bar chart).
- `fewshot_confusion.png` — confusion matrix (row-normalized).
- `fewshot_per_image_top3.csv` — per-image Top-3 predictions w/ scores.
- `SUMMARY.json` and `SUMMARY.md` — aggregate metrics and meta info.

## Notes

- Backbones: default `vit_base_patch14_reg4_dinov2.lvd142m`. You can change via `--model`.
- VRAM: If you hit OOM, reduce `--batch-train` and `--batch-eval` flags.
- Repro: Shots are randomly sampled once per run. Set `--seed` for reproducibility.
