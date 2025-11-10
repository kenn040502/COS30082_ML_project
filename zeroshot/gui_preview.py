# zeroshot/gui_preview.py
from __future__ import annotations
import os, argparse, io, time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F

from .dino_model import load_dino
from .data import build_class_mapping_from_folders, load_id_to_name_csv
from .cache import load_text_checkpoint

# ---- small helpers ----

@torch.no_grad()
def _encode_any(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):           # CLIP-style
        z = model.encode_image(x)
    else:                                        # timm / DINOv2
        z = model(x)
        if z.dim() > 2:
            z = z.mean(dim=tuple(range(2, z.dim())))
    return F.normalize(z, dim=-1)

def _infer_truth_from_path(data_root: str, image_path: str):
    rp = os.path.relpath(os.path.abspath(image_path), os.path.abspath(data_root))
    parts = rp.replace("\\", "/").split("/")
    # expect train/<domain>/<class_id>/file.jpg
    if len(parts) >= 3 and parts[0] == "train" and parts[1] in ("photo", "herbarium"):
        return parts[1], parts[2]
    return None, None

# ---- GUI app ----

class PreviewApp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.preprocess = load_dino(args.model, self.device)

        # class order must match prototypes
        self.class_ids, self.cls2idx = build_class_mapping_from_folders(args.data_root)
        self.id2name = load_id_to_name_csv(args.data_root)  # optional

        ck = load_text_checkpoint(args.proto_file)
        if ck is None or "text_cls_feats" not in ck:
            raise FileNotFoundError(f"Could not load prototypes from: {args.proto_file}")
        self.prototypes = ck["text_cls_feats"].to(self.device)  # [C,D]
        if self.prototypes.shape[0] != len(self.class_ids):
            raise RuntimeError(
                f"Prototype count ({self.prototypes.shape[0]}) != classes found ({len(self.class_ids)}). "
                "Rebuild prototypes (run zeroshot.main) to refresh cache."
            )

        # ---- tk UI ----
        self.root = tk.Tk()
        self.root.title("DINOv2 Prototype Zero-shot — Single Image Preview")

        top = tk.Frame(self.root); top.pack(fill="x", padx=10, pady=8)
        tk.Label(top, text=f"Model: {args.model}  |  Device: {self.device}").pack(anchor="w")
        tk.Label(top, text=f"Prototypes: {os.path.basename(args.proto_file)}  |  Classes: {len(self.class_ids)}").pack(anchor="w")
        tk.Label(top, text=f"Data root: {args.data_root}").pack(anchor="w")

        btns = tk.Frame(self.root); btns.pack(fill="x", padx=10, pady=6)
        tk.Button(btns, text="Select Image…", command=self.select_image).pack(side="left")
        self.status = tk.Label(btns, text="Ready", fg="#444")
        self.status.pack(side="left", padx=10)

        mid = tk.Frame(self.root); mid.pack(fill="both", expand=True, padx=10, pady=8)

        # Image panel
        self.img_panel = tk.Label(mid, bd=1, relief="sunken", width=420, height=320, anchor="center")
        self.img_panel.pack(side="left", padx=(0,10))
        self.img_panel_text = tk.Label(mid, text="(image preview)")
        self.img_panel_text.pack(side="left", anchor="n")

        # Results panel
        right = tk.Frame(mid)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Top-k Predictions", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.results = tk.Text(right, width=64, height=18, wrap="none")
        self.results.pack(fill="both", expand=True)
        self.results.configure(state="disabled")

        # Footer hint
        foot = tk.Frame(self.root); foot.pack(fill="x", padx=10, pady=(0,8))
        tk.Label(foot, text="Tip: If the image path is under train/<domain>/<class_id>/…, ground truth is auto-inferred.", fg="#666").pack(anchor="w")

    def run(self):
        self.root.mainloop()

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self._predict_and_show(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    @torch.no_grad()
    def _predict_and_show(self, path: str):
        self.status.config(text="Embedding…")
        self.root.update_idletasks()

        # load & show preview (resized)
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {e}")

        preview = pil.copy()
        preview.thumbnail((420, 320))
        tkimg = ImageTk.PhotoImage(preview)
        self.img_panel.configure(image=tkimg)
        self.img_panel.image = tkimg
        self.img_panel_text.config(text=os.path.basename(path))

        # encode
        x = self.preprocess(pil).unsqueeze(0).to(self.device)  # [1,3,H,W]
        z = _encode_any(self.model, x)                         # [1,D]
        logits = z @ self.prototypes.T                         # [1,C]

        vals, idxs = logits.topk(k=max(1, self.args.topk), dim=-1)
        vals = vals.squeeze(0).tolist()
        idxs = [int(i) for i in idxs.squeeze(0).tolist()]

        # inferred ground truth (optional)
        dom, gt_cid = _infer_truth_from_path(self.args.data_root, path)
        gt_idx = self.cls2idx.get(gt_cid, -1) if gt_cid else -1

        # render results
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("end", f"Image: {path}\n")
        if gt_cid:
            self.results.insert("end", f"Inferred GT: domain={dom}, class_id={gt_cid}\n")

        self.results.insert("end", "\nTop-k:\n")
        for r, (j, s) in enumerate(zip(idxs, vals), start=1):
            cid = self.class_ids[j]
            cname = self.id2name.get(cid, "")
            label = f"{cid} ({cname})" if cname else cid
            extra = ""
            if gt_idx >= 0 and j == gt_idx and r == 1:
                extra = "  ← CORRECT (Top-1)"
            elif gt_idx >= 0 and j == gt_idx:
                extra = "  ← contains GT"
            self.results.insert("end", f"{r:>2}. {label:<28} score={s:.4f}{extra}\n")

        if gt_idx >= 0:
            top1_correct = (idxs[0] == gt_idx)
            self.results.insert("end", f"\nTop-1 {'CORRECT ✅' if top1_correct else 'WRONG ❌'}\n")
        self.results.configure(state="disabled")

        self.status.config(text="Done")

def parse_args():
    ap = argparse.ArgumentParser("GUI: browse an image and see Top-k predictions")
    ap.add_argument("--data-root", required=True, help="Dataset root (contains train/herbarium and train/photo)")
    ap.add_argument("--proto-file", required=True, help="Path to prototypes .pt (from zero_shot_checkpoints)")
    ap.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--topk", type=int, default=5)
    return ap.parse_args()

def main():
    args = parse_args()
    app = PreviewApp(args)
    app.run()

if __name__ == "__main__":
    main()
