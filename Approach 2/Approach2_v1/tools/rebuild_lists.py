from pathlib import Path
import argparse

EXTS = {".jpg", ".jpeg", ".png"}

def gather_train(root: Path):
    out = []
    for domain in ["herbarium", "field", "photo"]:
        droot = root / "train" / domain
        if not droot.exists(): continue
        for cls_dir in sorted(droot.iterdir()):
            if not cls_dir.is_dir(): continue
            try:
                cls_id = int(cls_dir.name)
            except ValueError:
                continue
            for img in sorted(cls_dir.rglob("*")):
                if img.suffix.lower() in EXTS:
                    rel = img.relative_to(root).as_posix()
                    out.append(f"{rel} {cls_id}")
    return out

def gather_test_with_gt(root: Path):
    test_dir = root / "test"
    gt_file  = root / "list" / "groundtruth.txt"
    gt = {}
    if gt_file.exists():
        for line in gt_file.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            name, cid = line.strip().split()
            gt[name] = int(cid)
    out = []
    for img in sorted(test_dir.glob("*")):
        if img.suffix.lower() in EXTS:
            rel = img.relative_to(root).as_posix()
            name = img.name
            out.append(f"{rel} {gt.get(name, -1)}" if gt else rel)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="e.g. 'Herbarium_Field dataset' or 'data'")
    args = ap.parse_args()
    root = args.root

    (root/"list").mkdir(parents=True, exist_ok=True)
    (root/"list"/"train.txt").write_text("\n".join(gather_train(root))+"\n", encoding="utf-8")
    (root/"list"/"test.txt").write_text("\n".join(gather_test_with_gt(root))+"\n", encoding="utf-8")
    print("Wrote:", root/"list"/"train.txt")
    print("Wrote:", root/"list"/"test.txt")

if __name__ == "__main__":
    main()
