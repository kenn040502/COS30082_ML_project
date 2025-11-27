from __future__ import annotations
import os, csv
from typing import List, Tuple

def load_pairs_csv(data_root: str) -> list[tuple[str, str, str]]:
    """
    Load herbarium-photo pairs from:
      <data_root>/list/pairs.csv

    Expected columns:
      photo_path, herb_path, class_id

    Returns:
      list of (photo_path, herb_path, class_id) as strings.
    """
    path = os.path.join(data_root, "list", "pairs.csv")
    pairs: list[tuple[str, str, str]] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing pairs file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pp = row.get("photo_path", "").strip()
            hp = row.get("herb_path", "").strip()
            cid = row.get("class_id", "").strip()
            if not pp or not hp:
                continue
            pairs.append((pp, hp, cid))
    if not pairs:
        raise RuntimeError(f"No valid rows in {path}")
    return pairs
