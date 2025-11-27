"""
Compare two eval metrics JSON files and print deltas for overall / with_pairs / without_pairs.
Usage:
  python tools/compare_metrics.py old_metrics.json new_metrics.json
"""
import json
import sys

if len(sys.argv) != 3:
    print("Usage: python tools/compare_metrics.py old_metrics.json new_metrics.json")
    sys.exit(1)

old_path, new_path = sys.argv[1], sys.argv[2]
with open(old_path, 'r', encoding='utf-8') as f:
    old = json.load(f)
with open(new_path, 'r', encoding='utf-8') as f:
    new = json.load(f)

keys = ['overall', 'with_pairs', 'without_pairs']
for k in keys:
    old_top1 = old.get(k, {}).get('top1', 0.0)
    new_top1 = new.get(k, {}).get('top1', 0.0)
    old_top5 = old.get(k, {}).get('top5', 0.0)
    new_top5 = new.get(k, {}).get('top5', 0.0)
    print(f"{k}:")
    print(f"  Top-1: {old_top1:.2f}% -> {new_top1:.2f}%  (Δ {new_top1-old_top1:+.2f}%)")
    print(f"  Top-5: {old_top5:.2f}% -> {new_top5:.2f}%  (Δ {new_top5-old_top5:+.2f}%)")
    print()
print("Compare complete.")
