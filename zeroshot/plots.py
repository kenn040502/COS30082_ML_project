import os, numpy as np, matplotlib.pyplot as plt

def plot_per_class_accuracy(conf: np.ndarray, out_png: str, class_ids: list[str]):
    correct = np.diag(conf).astype(float)
    support = conf.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(support > 0, (correct / support) * 100.0, 0.0)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 5))
    idx = np.arange(len(class_ids))
    plt.bar(idx, acc)
    plt.xlabel("Class (global index)"); plt.ylabel("Accuracy (%)")
    plt.title("Per-class Accuracy (Zero-shot)")
    step = max(1, len(class_ids) // 20)
    plt.xticks(idx[::step], [class_ids[i] for i in idx[::step]], rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"📊 Saved per-class accuracy chart: {out_png}")

def plot_confusion(conf: np.ndarray, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(8, 6))
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_norm = conf / conf.sum(axis=1, keepdims=True)
        conf_norm = np.nan_to_num(conf_norm)
    plt.imshow(conf_norm, aspect='auto', interpolation='nearest')
    plt.colorbar(); plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"🧩 Saved confusion heatmap: {out_png}")
