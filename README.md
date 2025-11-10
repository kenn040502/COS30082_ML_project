# Zero‑Shot Evaluation — Quick Start (Windows/PowerShell)

This project evaluates **zero‑shot plant classification** using **DINOv2** image features and **text prototypes**.
It supports:
- **Unpaired** per‑domain accuracy (herbarium & photo): Top‑1/2/3/5, per‑class charts, confusion matrices, per‑image CSV.
- **Paired** photo→herbarium retrieval (optional if `list\pairs.csv` exists): R@1/R@5, mean rank, per‑query rank CSV.
- Safe GPU presets for **RTX 4060** and CPU mode.

> **Paths with spaces** are fine — keep the quotes in the commands.

---

## 0) One‑time setup

```powershell
# (Optional) Activate your venv first

# Install dependencies (minimal set)
pip install torch torchvision timm pillow matplotlib numpy
```

---

## 1) Point to your dataset and prototype checkpoint

```powershell
# Dataset root that has: train\herbarium\*, train\photo\*, list\*
$ROOT  = "C:\Users\User\Documents\GitHub\COS30082_ML_project\AML_project_herbarium_dataset"

# Text prototype checkpoint (.pt) built by zeroshot.main (step 2 below)
$PROTO = "zero_shot_checkpoints\prototypes_089aa5cdaa75694b.pt"

# (Optional) Choose a fixed output directory (otherwise a timestamped one is created)
$OUT   = "zero_shot_results\report_run"
```

**Expected dataset structure** (example):
```
AML_project_herbarium_dataset  train    herbarium\<class_id>\*.jpg
    photo\<class_id>\*.jpg
  list    groundtruth.txt           # (for older code paths; not required by eval_suite)
    pairs.csv                 # (optional; enables paired retrieval evaluation)
  class_names.csv             # (optional; maps class_id -> readable name)
```

---

## 2) (Re)build text prototypes (required if your classes/folders changed)

Run this once whenever class folders are updated so prototype order matches class order.

```powershell
python -m zeroshot.main --data-root "$ROOT" `
  --model vit_base_patch14_reg4_dinov2.lvd142m `
  --save-proto "$PROTO"
```

This writes a `.pt` file containing normalized text embeddings for all classes detected under `train\herbarium` and `train\photo` (the union).

---

## 3) Full evaluation (unpaired + optional paired)

The **same** command handles both unpaired and (if present) paired retrieval.

```powershell
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_base_patch14_reg4_dinov2.lvd142m `
  --img-batch 8 `
  --outdir "$OUT"
```

Outputs (under `--outdir`):
- `GLOBAL_SUMMARY.json`
- `unpaired_herbarium_summary.json` (+ per‑image/per‑class CSVs and charts)
- `unpaired_photo_summary.json` (+ per‑image/per‑class CSVs and charts)
- If `list\pairs.csv` exists:
  - `paired_overall.json` (R@1/R@5/mean_rank)
  - `paired_per_query_ranks.csv` (rank of true herb for each photo)
  - `paired_splits_unpaired_photo.json` (photo Top‑1 split: classes **with** vs **without** pairs)

---

## 4) Presets and variants

### Safer settings for **RTX 4060** (cooler / lower VRAM)
```powershell
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_base_patch14_reg4_dinov2.lvd142m `
  --img-batch 6 `
  --outdir "$OUT"
# If OOM persists, try --img-batch 4
```

### CPU mode (no GPU heat)
```powershell
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_base_patch14_reg4_dinov2.lvd142m `
  --device cpu `
  --img-batch 4 `
  --outdir "$OUT"
```

### Try a different DINOv2 checkpoint (timm model names)
```powershell
# Smaller (cooler)
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_small_patch14_reg4_dinov2.lvd142m `
  --img-batch 8 `
  --outdir "$OUTit_s14"

# Larger (heavier; reduce batch)
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_large_patch14_reg4_dinov2.lvd142m `
  --img-batch 4 `
  --outdir "$OUTit_l14"
```

---

## 5) Unpaired‑only (force skip paired)

If `pairs.csv` exists but you want to skip paired evaluation for a run:
```powershell
Rename-Item "$ROOT\list\pairs.csv" "pairs.skip.csv"
python -m zeroshot.eval_suite --data-root "$ROOT" --proto-file "$PROTO" `
  --model vit_base_patch14_reg4_dinov2.lvd142m `
  --img-batch 8 `
  --outdir "$OUT"
Rename-Item "$ROOT\list\pairs.skip.csv" "pairs.csv"
```

---

## 6) Troubleshooting

- **Prototype count != classes**  
  Rebuild prototypes (Step 2) so the order matches current folders.
- **OOM on GPU**  
  Lower `--img-batch` to 6 or 4; close other GPU apps.
- **No `pairs.csv`**  
  Paired metrics are skipped automatically.
- **Class names on charts**  
  If `class_names.csv` exists in dataset root (`class_id,name`), charts/CSVs will use readable names.
- **Long paths / spaces**  
  Always keep the double quotes around `$ROOT` and `$OUT`.
- **Confusion matrix looks empty**  
  It is normalized by row; classes with few samples may appear very dark/light.

---

## 7) Expected artifacts (example)

```
zero_shot_results
eport_run  GLOBAL_SUMMARY.json
  unpaired_herbarium_summary.json
  unpaired_herbarium_per_image_top3.csv
  unpaired_herbarium_per_class_top123.csv
  unpaired_herbarium_per_class_top1.png
  unpaired_herbarium_confusion.png
  unpaired_photo_summary.json
  unpaired_photo_per_image_top3.csv
  unpaired_photo_per_class_top123.csv
  unpaired_photo_per_class_top1.png
  unpaired_photo_confusion.png
  paired_overall.json                 # if pairs.csv exists
  paired_per_query_ranks.csv          # if pairs.csv exists
  paired_splits_unpaired_photo.json   # if pairs.csv exists
```
