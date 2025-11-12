# zeroshot/rank_checkpoints.py
import argparse, json, shutil, time
from pathlib import Path
from typing import Dict, Any, List

def _safe_load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _safe_write_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def pick_metric(global_summary: Dict[str, Any], which: str) -> float:
    """
    which in {"photo","herbarium","mean"}
    Expects global_summary like:
      {
        "unpaired": {
          "photo": {"top1": float, ...},
          "herbarium": {"top1": float, ...}
        },
        "paired": {...}  # optional
      }
    """
    up = global_summary.get("unpaired", {})
    ph = (up.get("photo", {}) or {}).get("top1", None)
    hb = (up.get("herbarium", {}) or {}).get("top1", None)

    if which == "photo" and ph is not None:
        return float(ph)
    if which == "herbarium" and hb is not None:
        return float(hb)
    # mean (fallbacks gracefully)
    vals = [v for v in [ph, hb] if v is not None]
    if not vals:
        raise ValueError("No unpaired top1 metrics found in GLOBAL_SUMMARY.json.")
    return float(sum(vals) / len(vals))

def format_sidecar(meta: Dict[str, Any]) -> str:
    lines = [
        f"timestamp       : {meta['timestamp']}",
        f"score_metric    : {meta['score_metric']} = {meta['score']:.4f}",
        f"proto_file      : {meta['proto_file']}",
        f"model_name      : {meta.get('model_name','(unknown)')}",
        f"run_outdir      : {meta['run_outdir']}",
        f"global_summary  : {meta['summary_file']}",
    ]
    # optional extras:
    if 'paired' in meta:
        lines.append(f"paired_present  : {bool(meta['paired'])}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser(
        description="Rank and checkpoint top-3 prototype files from a zero-shot run."
    )
    ap.add_argument("--summary-file", required=True,
                    help="Path to GLOBAL_SUMMARY.json produced by eval_suite.")
    ap.add_argument("--proto-file", required=True,
                    help="The prototype .pt file used for this run.")
    ap.add_argument("--model-name", default="vit_base_patch14_reg4_dinov2.lvd142m",
                    help="Model identifier for record-keeping.")
    ap.add_argument("--metric", default="mean", choices=["photo","herbarium","mean"],
                    help="Which score to rank by (Top-1 %).")
    ap.add_argument("--leaderboard-db", default="zero_shot_checkpoints/leaderboard.json",
                    help="Path to leaderboard JSON DB.")
    ap.add_argument("--dst-dir", default="zero_shot_checkpoints/ranked",
                    help="Directory where model_1st.pt / 2nd / 3rd are materialized.")
    args = ap.parse_args()

    summary_file = Path(args.summary_file).resolve()
    proto_file   = Path(args.proto_file).resolve()
    outdir       = summary_file.parent
    lb_path      = Path(args.leaderboard_db)
    dst_dir      = Path(args.dst_dir)

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found: {summary_file}")
    if not proto_file.exists():
        raise FileNotFoundError(f"Proto file not found: {proto_file}")

    global_summary = _safe_load_json(summary_file)
    score = pick_metric(global_summary, args.metric)

    # Build entry
    entry = {
        "timestamp": int(time.time()),
        "score_metric": args.metric,
        "score": score,
        "proto_file": str(proto_file),
        "model_name": args.model_name,
        "run_outdir": str(outdir),
        "summary_file": str(summary_file),
        "paired": "paired" in global_summary
    }

    # Load DB, append, sort desc by score
    db = _safe_load_json(lb_path)
    runs: List[Dict[str, Any]] = db.get("runs", [])
    runs.append(entry)
    runs.sort(key=lambda r: r["score"], reverse=True)
    db["runs"] = runs
    db["updated_at"] = int(time.time())

    # Write DB
    _safe_write_json(lb_path, db)

    # Materialize Top-3
    dst_dir.mkdir(parents=True, exist_ok=True)
    rank_targets = [
        ("model_1st.pt", "model_1st.txt"),
        ("model_2nd.pt", "model_2nd.txt"),
        ("model_3rd.pt", "model_3rd.txt"),
    ]

    for i, (pt_name, txt_name) in enumerate(rank_targets):
        if i >= len(runs):
            break
        src_proto = Path(runs[i]["proto_file"])
        if not src_proto.exists():
            # Skip missing historical files
            continue
        dst_pt  = dst_dir / pt_name
        dst_txt = dst_dir / txt_name
        shutil.copy2(src_proto, dst_pt)
        dst_txt.write_text(format_sidecar(runs[i]), encoding="utf-8")

    print(f"✅ Leaderboard updated: {lb_path}")
    print(f"🥇/🥈/🥉 materialized under: {dst_dir}")

if __name__ == "__main__":
    main()
