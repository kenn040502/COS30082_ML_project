# dataset_lists.py
# ---------------------------------------------------------
# Utilities to inspect/validate dataset list files.
#
# Typical layout:
#   <dataset_root>/
#     ├─ train/
#     │   ├─ herbarium/...
#     │   └─ field/...
#     ├─ test/...
#     └─ list/
#         ├─ train.txt
#         ├─ test.txt
#         └─ species_list.txt
#
# Usage:
#   # Quick check (warn about missing files, show stats)
#   python src/dataset_lists.py --root "Herbarium_Field dataset" --list "list/train.txt"
#
#   # Or check both train/test at once
#   python src/dataset_lists.py --root "Herbarium_Field dataset"
# ---------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

def parse_list(root: Path, list_path: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rel, lab = ln.rsplit(maxsplit=1)
                lab = int(lab)
            except ValueError:
                print(f"[WARN] {list_path.name}:{i} cannot parse line: {ln}")
                continue
            p = Path(rel)
            if not p.is_absolute():
                p = root / rel
            if not p.exists():
                print(f"[WARN] missing file: {p}")
                continue
            items.append((p, lab))
    if not items:
        print(f"[WARN] no valid entries found in {list_path}")
    return items


def show_stats(name: str, items: List[Tuple[Path, int]]) -> None:
    print(f"[{name}] total entries: {len(items)}")
    if not items:
        return
    labels = [lab for _, lab in items]
    uniq  = sorted(set(labels))
    print(f"[{name}] unique classes: {len(uniq)} (min={min(uniq)} max={max(uniq)})")

    # quick domain breakdown if paths contain herbarium/field/photo
    dom_counts = {"herbarium": 0, "field": 0, "photo": 0, "other": 0}
    for p, _ in items:
        s = str(p.as_posix())
        if "/herbarium/" in s or s.endswith("/herbarium"):
            dom_counts["herbarium"] += 1
        elif "/field/" in s:
            dom_counts["field"] += 1
        elif "/photo/" in s:
            dom_counts["photo"] += 1
        else:
            dom_counts["other"] += 1
    print(f"[{name}] by domain: {dom_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (the folder containing train/, test/, list/)")
    ap.add_argument("--list", default=None, help="Optional: a single list file to check (relative to root)")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Dataset root not found: {root}"

    if args.list:
        list_path = (root / args.list) if not Path(args.list).is_absolute() else Path(args.list)
        assert list_path.exists(), f"List file not found: {list_path}"
        items = parse_list(root, list_path)
        show_stats(list_path.name, items)
    else:
        # default: check both train/test lists if present
        for rel in ["list/train.txt", "list/test.txt"]:
            lp = root / rel
            if lp.exists():
                items = parse_list(root, lp)
                show_stats(rel, items)
            else:
                print(f"[INFO] missing list file (skip): {lp}")


if __name__ == "__main__":
    main()
