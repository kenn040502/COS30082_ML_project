# COS30082 ML Project

Cross-domain plant species identification across herbarium sheets and field photos. The repo bundles the dataset splits/images, three training approaches, and GUI demos for inference.

## Repository Map
- `AML_project_herbarium_dataset/` — dataset with `train/`, `test/`, and `list/` (train.txt, test.txt, groundtruth.txt, species lists).
- `Approach 1/` — ConvNeXt baseline training/eval helpers (`src/` for improved_baseline1_* plus diagnostics and plotting scripts).
- `Approach 2/` — DINOv2 feature-extraction pipeline with a lightweight scikit-learn head (`src/`); precomputed features in `features/`, pretrained heads in `weights/`, logs in `process/` and `results/`.
- `Approach 3/` — DINOv2 + metric/triplet learning package in `approach3/` with run outputs under `runs_approach3*`.
- `Gui/` — Tkinter and Streamlit inference demos that load `improved_baseline1_best.pth`.
- `commands.txt` — quick-start commands; `requirements.txt` — top-level dependencies.

## Path Handling
- All approaches look for `AML_DATA_ROOT` first; if unset, they fall back to `./AML_project_herbarium_dataset`.
  - PowerShell: `$env:AML_DATA_ROOT = (Resolve-Path "AML_project_herbarium_dataset")`
  - Bash: `export AML_DATA_ROOT="$PWD/AML_project_herbarium_dataset"`
- Approach 1 also honors `--data_dir` or `CONFIG['data_dir']` if you need a custom path.
- Approach 3 is a package under `Approach 3/approach3`. Add `Approach 3` to `PYTHONPATH` or call the train/eval scripts by path.
  - PowerShell: `$env:PYTHONPATH = "$PWD\\Approach 3"`
  - Bash: `export PYTHONPATH="$PWD/Approach 3"`

## Approach Overviews
- **Approach 1 - ConvNeXt baseline** (`Approach 1/src`): two-phase ConvNeXt-S with mixup, optional class-balanced/focal loss, Albumentations, cosine/OneCycle LR, and TTA at final eval. `improved_baseline1_train.py` trains and evaluates; `test_model.py` re-evaluates checkpoints.
- **Approach 2 - DINOv2 features + sklearn head** (`Approach 2/src`): `build_features.py` extracts embeddings from a DINOv2 reg4 checkpoint, `train_sklearn.py` fits logreg/SVC/KNN heads, `eval_topk.py` reports Top-K. Precomputed `features/*.npz` and `weights/*.pkl` are provided; rebuild features only if you have the DINOv2 checkpoint.
- **Approach 3 - DINOv2 metric/triplet** (`Approach 3/approach3`): frozen DINOv2 backbone with projection head, classifier, triplet loss, and optional domain-adversarial term. Use `approach3_train.py`/`approach3_eval.py`.

## Quick Start (repo root)
- Set dataset env (optional if using bundled dataset):
  - PowerShell: `$env:AML_DATA_ROOT = (Resolve-Path "AML_project_herbarium_dataset")`
  - Bash: `export AML_DATA_ROOT="$PWD/AML_project_herbarium_dataset"`
- Approach 1: `python "Approach 1/src/improved_baseline1_train.py"`
- Approach 2 (precomputed features):
  ```
  python "Approach 2/src/train_sklearn.py" --train "Approach 2/features/train.npz" --out_dir "Approach 2/weights"
  python "Approach 2/src/eval_topk.py" --clf "Approach 2/weights/sklearn_model.pkl" --train "Approach 2/features/train.npz" --test "Approach 2/features/test.npz" --species-list "AML_project_herbarium_dataset/list/species_list.txt" --topk 1 5
  ```
- Approach 2 (rebuild features; requires your DINOv2 checkpoint path in `$CKPT`):
  ```
  python "Approach 2/src/build_features.py" --data-list "AML_project_herbarium_dataset/list/train.txt" --weights "$CKPT" --out "Approach 2/features/train.npz" --batch 16 --device cuda
  python "Approach 2/src/build_features.py" --data-list "AML_project_herbarium_dataset/list/test.txt"  --weights "$CKPT" --out "Approach 2/features/test.npz"  --batch 16 --device cuda --groundtruth "AML_project_herbarium_dataset/list/groundtruth.txt"
  python "Approach 2/src/train_sklearn.py"  --train "Approach 2/features/train.npz" --out_dir "Approach 2/weights"
  python "Approach 2/src/eval_topk.py"      --clf "Approach 2/weights/sklearn_model.pkl" --train "Approach 2/features/train.npz" --test "Approach 2/features/test.npz" --species-list "AML_project_herbarium_dataset/list/species_list.txt" --topk 1 5
  ```
- Approach 3 (add PYTHONPATH or call scripts directly):
  ```
  python -m approach3.approach3_train --model vit_base_patch14_reg4_dinov2.lvd142m --epochs 20 --batch-size 32 --lr 1e-3 --margin 0.3 --outdir "Approach 3/runs_approach3"
  python -m approach3.approach3_eval  --ckpt "Approach 3/runs_approach3/best_model.pt" --outdir "Approach 3/runs_approach3/report"
  ```
- GUI demos (Approach 1 inference; uses `improved_baseline1_best.pth`):
  - Tkinter: `python "Gui/tk_app.py"` (flags: `--checkpoint`, `--data-root`, `--device`)
  - Streamlit: `streamlit run Gui/streamlit_app.py`
  - The GUIs rebuild the label map from `AML_project_herbarium_dataset/list`, so keep that folder intact or point `--data-root` / `AML_DATA_ROOT` to it.

## Notes on Data and Outputs
- Dataset splits live in `AML_project_herbarium_dataset/list`.
- Outputs stay under each approach (`runs_*`, `weights/`, `results/`, plots). Prune locally if you need space.
