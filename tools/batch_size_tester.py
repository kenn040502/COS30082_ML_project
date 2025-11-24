"""
Batch size tester

Usage:
  python tools/batch_size_tester.py --data-root <DATA_ROOT> --batch-sizes 32 24 16

This script runs short 1-epoch trials for the given batch sizes and reports which sizes completed without OOM.
It runs `python -m approach3.train` as a subprocess so it uses the same virtualenv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_trial(data_root, outdir, batch_size, num_workers, amp, augment):
    cmd = [sys.executable, "-m", "approach3.train",
           "--data-root", data_root,
           "--outdir", outdir,
           "--epochs", "1",
           "--batch-size", str(batch_size),
           "--num-workers", str(num_workers)]
    if amp:
        cmd.append("--amp")
    if augment:
        cmd.append("--augment")
    print("Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = proc.stdout + proc.stderr
    # Detect OOM
    if "out of memory" in out.lower() or "cuda out of memory" in out.lower():
        return False, "oom"
    if proc.returncode != 0:
        return False, f"error (code {proc.returncode})"
    return True, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--outdir", default="runs_batch_tester")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32,24,16])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()

    results = {}
    for b in args.batch_sizes:
        outdir = f"{args.outdir}_b{b}"
        ok, reason = run_trial(args.data_root, outdir, b, args.num_workers, args.amp, args.augment)
        results[b] = (ok, reason)
        print(f"Batch {b}: {ok} ({reason})")

    print("\nSummary:")
    for b, (ok, reason) in results.items():
        print(f"  {b}: {'OK' if ok else 'FAILED'} ({reason})")

if __name__ == '__main__':
    main()
