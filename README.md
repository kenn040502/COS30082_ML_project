# COS30082 ML Project

Cross-domain plant species identification across herbarium sheets and field photos. The repository bundles three approaches plus the shared dataset lists and images.

## Repository Map
- `AML_project_herbarium_dataset/` - train/test splits, class lists, and all images.
- `Approach 1/` - CDAN-based domain adaptation pipeline (`cdna_pipeline`).
- `Approach 2/` - improved deep baseline (rich augmentation + class balancing), feature-extraction pipeline (`Approach2_v1`), and a copy of the CDAN pipeline for comparison.
- `Approach 3/` - DINOv2 transfer learning with metric learning and light domain-adversarial losses; multiple run outputs in `runs_*`.

## Path Handling
- All scripts look for the dataset at `AML_project_herbarium_dataset/` relative to the repo root.
- To override, set `AML_DATA_ROOT` to your dataset path before running (e.g., `export AML_DATA_ROOT=/data/AML_project_herbarium_dataset` or `set AML_DATA_ROOT=C:\data\AML_project_herbarium_dataset`).

## Approach Overviews
- **Approach 1 - CDAN domain adaptation** (`Approach 1/cdna_pipeline`): DINO/ResNet backbone, classifier head, and domain discriminator (CDAN + gradient reversal). Entry point `main.py`; `test.py` evaluates saved checkpoints. Both respect `AML_DATA_ROOT`.
- **Approach 2 - Improved baselines**:
  - Deep baseline (`Approach 2/src`): `improved_baseline1_train.py` uses Albumentations, mixup, class-balanced/focal loss, cosine/OneCycle LR, and TTA. The CONFIG block defaults to `AML_DATA_ROOT` or the repo-local dataset.
  - Feature pipeline (`Approach 2/Approach2_v1`): `src/` builds DINOv2 features (`build_features.py`), trains a scikit-learn head (`train_sklearn.py`), and evaluates (`eval_topk.py`). Precomputed features are in `features/`, a trained model in `weights/`, and process notes/results in `process/` and `results/`. For fresh extraction, provide a DINOv2 checkpoint (e.g., `weights/model_best.pth.tar`) via `--weights` to `build_features.py` (see `commands.txt`).
  - CDAN copy (`Approach 2/cdna_pipeline`) for side-by-side experiments.
- **Approach 3 - DINOv2 + metric learning** (`Approach 3/approach3`): frozen DINOv2 backbone with projection head, classifier, triplet loss, and optional domain-adversarial term. `approach3_train.py` and `approach3_eval.py` default to `AML_DATA_ROOT` or the repo-local dataset.

## Running Commands
- See `commands.txt` for a shell-agnostic sequence to train/evaluate each approach from the repo root. Paths are relative; set `AML_DATA_ROOT` if your dataset lives elsewhere.

## Notes on Data
- Split definitions live in `AML_project_herbarium_dataset/list`.
- Run outputs (plots, logs, checkpoints) stay under each approach's `runs_*` or `checkpoints/`; prune locally if you need a lighter workspace.
