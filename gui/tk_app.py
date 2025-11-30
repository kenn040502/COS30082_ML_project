import os
import sys
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F

# Ensure 'Approach 3' is importable
ROOT = os.getcwd()
APP3_ROOT = os.path.join(ROOT, 'Approach 3')
if APP3_ROOT not in sys.path:
    sys.path.insert(0, APP3_ROOT)

try:
    from dino_model import load_dino
except Exception as e:
    raise ImportError(f"Could not import load_dino from Approach 3: {e}")

try:
    from approach3.models import ProjectionHead, SpeciesClassifier
except Exception as e:
    raise ImportError(f"Could not import approach3.models: {e}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InferenceApp:
    def __init__(self, master):
        self.master = master
        master.title('Herbarium — Inference GUI (Tk)')

        self.ckpt_var = tk.StringVar()
        self.approach_var = tk.StringVar(value='Approach 3')
        self.extractor_weights_var = tk.StringVar()
        self.classifier_var = tk.StringVar()
        self.image_path = None
        self.model = None
        self.preprocess = None
        self.meta = None
        self.data_root_var = tk.StringVar(value=os.path.join(ROOT, 'AML_project_herbarium_dataset'))

        # Top frame: model selection
        frm = ttk.Frame(master, padding=8)
        frm.grid(row=0, column=0, sticky='ew')

        ttk.Label(frm, text='Approach:').grid(row=0, column=0, sticky='w')
        self.approach_combo = ttk.Combobox(frm, values=['Approach 1', 'Approach 2', 'Approach 3'], textvariable=self.approach_var, state='readonly', width=14)
        self.approach_combo.grid(row=0, column=1, sticky='w')
        self.approach_combo.bind('<<ComboboxSelected>>', lambda e: self.on_approach_change())

        ttk.Label(frm, text='Checkpoint:').grid(row=0, column=2, sticky='w')
        self.ckpt_entry = ttk.Entry(frm, textvariable=self.ckpt_var, width=46)
        self.ckpt_entry.grid(row=0, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_ckpt).grid(row=0, column=2)
        # Model picker (scan runs_*/ folders)
        ttk.Label(frm, text='Models:').grid(row=1, column=3, sticky='w')
        self.model_combo = ttk.Combobox(frm, values=self.find_checkpoints(), state='readonly', width=56)
        self.model_combo.grid(row=1, column=1, sticky='ew')
        ttk.Button(frm, text='Refresh', command=self.refresh_model_list).grid(row=1, column=2)

        # Data root and Image selection
        ttk.Label(frm, text='Data root:').grid(row=2, column=0, sticky='w')
        self.data_root_entry = ttk.Entry(frm, textvariable=self.data_root_var, width=48)
        self.data_root_entry.grid(row=2, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_data_root).grid(row=2, column=2)

        # Approach-specific fields (Approach 2: extractor weights + classifier file)
        ttk.Label(frm, text='Extractor weights:').grid(row=3, column=0, sticky='w')
        self.extractor_entry = ttk.Entry(frm, textvariable=self.extractor_weights_var, width=48)
        self.extractor_entry.grid(row=3, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_extractor).grid(row=3, column=2)

        ttk.Label(frm, text='Classifier file:').grid(row=3, column=3, sticky='w')
        self.classifier_entry = ttk.Entry(frm, textvariable=self.classifier_var, width=48)
        self.classifier_entry.grid(row=3, column=4, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_classifier).grid(row=3, column=5)

        ttk.Button(frm, text='Load Image', command=self.browse_image).grid(row=3, column=0, pady=6)
        self.img_label = ttk.Label(frm, text='No image selected')
        self.img_label.grid(row=3, column=1, sticky='w')

        # Top-K selector
        ttk.Label(frm, text='Top-K:').grid(row=4, column=0, sticky='w')
        self.topk_spin = ttk.Spinbox(frm, from_=1, to=10, width=5)
        self.topk_spin.set(5)
        self.topk_spin.grid(row=2, column=1, sticky='w')

        # Run button
        ttk.Button(frm, text='Run Inference', command=self.run_inference_thread).grid(row=4, column=2)

        # Results area
        self.left = ttk.LabelFrame(master, text='Image', padding=8)
        self.left.grid(row=1, column=0, sticky='nsw')
        self.canvas = tk.Canvas(self.left, width=400, height=400)
        self.canvas.pack()

        self.right = ttk.LabelFrame(master, text='Predictions', padding=8)
        self.right.grid(row=1, column=1, sticky='nsew')

        self.pred_text = tk.Text(self.right, width=50, height=15)
        self.pred_text.pack()

        self.examples_frame = ttk.Frame(master, padding=8)
        self.examples_frame.grid(row=2, column=0, columnspan=2, sticky='ew')

        master.columnconfigure(1, weight=1)

        # status bar and menu
        self.status_var = tk.StringVar(value='Ready')
        self.status = ttk.Label(master, textvariable=self.status_var, relief='sunken', anchor='w')
        self.status.grid(row=99, column=0, columnspan=2, sticky='ew')

        self.progress = ttk.Progressbar(master, orient='horizontal', mode='determinate')
        self.progress.grid(row=100, column=0, columnspan=2, sticky='ew', pady=(2,6))

        self.create_menu()

        # storage for last batch results for export
        self.last_results = []

        # initialize approach-specific UI states (after status/menu created)
        try:
            self.on_approach_change()
        except Exception:
            pass

    def find_checkpoints(self):
        # Search for any folder named runs* anywhere under the repo (including Approach 3 subfolders)
        cks = []
        for p in Path(ROOT).rglob('runs*'):
            if p.is_dir():
                for f in p.rglob('*.pt'):
                    cks.append(str(f))
                for f in p.rglob('*.pth'):
                    cks.append(str(f))
        # Also include any top-level .pt/.pth for convenience
        for f in Path(ROOT).rglob('*.pt'):
            cks.append(str(f))
        for f in Path(ROOT).rglob('*.pth'):
            cks.append(str(f))
        # unique & sorted
        seen = set()
        out = []
        for x in sorted(cks):
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def refresh_model_list(self):
        vals = self.find_checkpoints()
        self.model_combo['values'] = vals
    def browse_ckpt(self):
        path = filedialog.askopenfilename(title='Select checkpoint (.pt/.pth)', filetypes=[('PyTorch', '*.pt;*.pth'), ('All files', '*.*')])
        if path:
            self.ckpt_var.set(path)
            vals = list(self.model_combo['values'])
            if path not in vals:
                vals = vals + [path]
                self.model_combo['values'] = vals

    def browse_extractor(self):
        path = filedialog.askopenfilename(title='Select extractor weights', filetypes=[('PyTorch', '*.pt;*.pth'), ('All files', '*.*')])
        if path:
            self.extractor_weights_var.set(path)

    def browse_classifier(self):
        path = filedialog.askopenfilename(title='Select classifier (joblib/pkl)', filetypes=[('Joblib', '*.pkl;*.joblib'), ('All files', '*.*')])
        if path:
            self.classifier_var.set(path)

    def browse_image(self):
        path = filedialog.askopenfilename(title='Select image', filetypes=[('Images', '*.jpg;*.jpeg;*.png'), ('All files', '*.*')])
        if path:
            self.image_path = path
            self.img_label.config(text=os.path.basename(path))
            self.show_image(path)

    def infer_image(self, image_path):
        # Core inference logic that returns (lines, examples_map) without touching UI widgets
        from PIL import Image as PILImage
        with PILImage.open(image_path) as pil:
            img = pil.convert('RGB')

        # ensure model loaded
        ckpt_path = self.ckpt_var.get().strip() or ''
        if self.model is None or getattr(self, 'loaded_ckpt', None) != ckpt_path or getattr(self, 'loaded_approach', None) != self.approach_var.get():
            # attempt to load (this will validate required files)
            self.load_model_if_needed(ckpt_path)
            if self.model is None:
                raise RuntimeError('Model failed to load')

        approach = self.approach_var.get()
        preprocess = self.preprocess

        if approach == 'Approach 3':
            backbone, proj, clf = self.model
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            backbone.to(DEVICE); backbone.eval()
            with torch.no_grad():
                h = backbone(tensor)
                if h.dim() > 2:
                    h = h.mean(dim=tuple(range(2, h.dim())))
                z = proj(h)
                logits = clf(z)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        elif approach == 'Approach 1':
            F_backbone, C = self.model
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            F_backbone.to(DEVICE); C.to(DEVICE)
            F_backbone.eval(); C.eval()
            with torch.no_grad():
                feats = F_backbone(tensor)
                if feats.dim() > 2:
                    feats = feats.mean(dim=tuple(range(2, feats.dim())))
                logits = C(feats)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        elif approach == 'Approach 2':
            feat_model, clf = self.model
            tf_img = preprocess(img)
            if isinstance(tf_img, torch.Tensor):
                tensor = tf_img.unsqueeze(0).to(DEVICE)
            else:
                from torchvision import transforms
                tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

            feat_model.to(DEVICE); feat_model.eval()
            with torch.no_grad():
                feats = feat_model(tensor)
                if feats.ndim > 2:
                    feats = torch.flatten(feats, 1)
                X = feats.cpu().numpy()

            import numpy as _np
            if hasattr(clf, 'predict_proba'):
                P = clf.predict_proba(X)
            else:
                S = clf.decision_function(X)
                if S.ndim == 1:
                    S = _np.stack([-S, S], axis=1)
                E = _np.exp(S - S.max(axis=1, keepdims=True))
                P = E / E.sum(axis=1, keepdims=True)
            probs = P[0]

        else:
            raise RuntimeError(f'Unknown approach: {approach}')

        k = int(self.topk_spin.get())
        topk_idx = probs.argsort()[::-1][:k]
        lines = []
        classes_ordered = []
        for rank, idx in enumerate(topk_idx, start=1):
            clsname = self.meta['classes'][int(idx)]
            score = probs[int(idx)] * 100.0
            lines.append(f"Top-{rank}: {clsname} — {score:.2f}%")
            classes_ordered.append(str(clsname))

        # build examples_map: key -> list of example files
        examples_map = {}
        data_root = self.data_root_var.get() or os.path.join(ROOT, 'AML_project_herbarium_dataset')
        for cls in classes_ordered:
            exs = []
            for sub in ('train/herbarium', 'train/photo'):
                d = os.path.join(data_root, sub, str(cls))
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d))[:4]:
                        exs.append(os.path.join(d, f))
            examples_map[cls] = exs

        return lines, examples_map

    def run_batch(self):
        folder = filedialog.askdirectory(title='Select folder of images to process')
        if not folder:
            return
        # gather images
        imgs = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            imgs.extend(sorted(Path(folder).rglob(ext)))
        imgs = [str(p) for p in imgs]
        if not imgs:
            messagebox.showinfo('Batch', 'No images found in folder')
            return

        ckpt_path = self.ckpt_var.get().strip() or ''
        self.set_status(f'Loading model for batch...')
        self.load_model_if_needed(ckpt_path)
        if self.model is None:
            self.set_status('Model load failed')
            return

        self.progress['maximum'] = len(imgs)
        self.progress['value'] = 0
        self.last_results = []
        for i, im in enumerate(imgs, start=1):
            try:
                lines, examples_map = self.infer_image(im)
                top1 = lines[0] if lines else ''
                try:
                    _, rest = top1.split(':', 1)
                    cls, score = rest.split('—')
                    cls = cls.strip(); score = float(score.strip().rstrip('%'))
                except Exception:
                    cls = ''; score = 0.0
                self.last_results.append({'image': im, 'top1': cls, 'top1_score': score, 'topk': '|'.join([l.split(':',1)[1].strip() for l in lines])})
            except Exception as e:
                self.last_results.append({'image': im, 'top1': '', 'top1_score': 0.0, 'topk': ''})
            self.progress['value'] = i
            self.set_status(f'Processed {i}/{len(imgs)}')

        self.set_status(f'Batch complete: {len(imgs)} images')

    def browse_data_root(self):
        path = filedialog.askdirectory(title='Select dataset root')
        if path:
            self.data_root_var.set(path)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label='Run Batch...', command=self.run_batch_thread)
        file_menu.add_command(label='Export Last Results...', command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.master.quit)
        menubar.add_cascade(label='File', menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label='How to use', command=self.show_help)
        menubar.add_cascade(label='Help', menu=help_menu)

        try:
            self.master.config(menu=menubar)
        except Exception:
            pass

    def set_status(self, text):
        self.status_var.set(text)

    def on_approach_change(self):
        ap = self.approach_var.get()
        # show/hide approach-specific fields if needed (simple enable/disable)
        if ap == 'Approach 2':
            try:
                self.extractor_entry.config(state='normal')
                self.classifier_entry.config(state='normal')
            except Exception:
                pass
        else:
            try:
                self.extractor_entry.config(state='disabled')
                self.classifier_entry.config(state='disabled')
            except Exception:
                pass
        # update status with a quick validation
        ok, msg = self.validate_files_for_approach(ap)
        if ok:
            self.set_status(msg)
        else:
            self.set_status(msg)

    def validate_files_for_approach(self, ap):
        """Return (ok:bool, message:str). Also try to auto-fill a classifier for Approach 2 if found."""
        # Approach 2: needs a classifier file (joblib .pkl). extractor optional.
        if ap == 'Approach 2':
            clf = self.classifier_var.get().strip()
            if clf and os.path.exists(clf):
                return True, 'Approach 2 ready (classifier found)'
            # try to auto-detect a classifier in the repo
            cand_dir = os.path.join(ROOT, 'Approach 2', 'Approach2_v1', 'weights')
            if os.path.isdir(cand_dir):
                for f in os.listdir(cand_dir):
                    if f.endswith('.pkl') or f.endswith('.joblib'):
                        p = os.path.join(cand_dir, f)
                        self.classifier_var.set(p)
                        return True, f'Approach 2 ready (auto-filled classifier: {f})'
            return False, 'Approach 2: classifier not set (use File->Browse or place .pkl in Approach 2/Approach2_v1/weights)'

        # Approach 1 & 3: require checkpoint or discovered model
        if ap in ('Approach 1', 'Approach 3'):
            ck = self.ckpt_var.get().strip()
            if ck and os.path.exists(ck):
                return True, f'{ap} ready (checkpoint set)'
            # fall back to scanning known runs
            vals = self.find_checkpoints()
            if vals:
                # prefill first value if none set
                if not ck:
                    self.ckpt_var.set(vals[0])
                return True, f'{ap} ready (found {len(vals)} checkpoint(s))'
            return False, f'{ap}: no checkpoint found (select via Browse or place .pt under runs_*)'

        return False, 'Unknown approach'

    def show_image(self, path):
        img = Image.open(path).convert('RGB')
        img.thumbnail((400, 400))
        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.create_image(200, 200, image=self.tkimg)

    def run_inference_thread(self):
        t = threading.Thread(target=self.run_inference)
        t.start()

    def run_batch_thread(self):
        t = threading.Thread(target=self.run_batch)
        t.start()

    def export_results(self):
        if not self.last_results:
            messagebox.showinfo('Export', 'No results to export yet. Run a batch or inference first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['image', 'top1', 'top1_score', 'topk'])
                writer.writeheader()
                for r in self.last_results:
                    writer.writerow(r)
            messagebox.showinfo('Export', f'Exported {len(self.last_results)} rows to {path}')
        except Exception as e:
            messagebox.showerror('Export Error', str(e))

    def show_help(self):
        msg = (
            'Usage:\n'
            '- Select an Approach (1/2/3).\n'
            "- For Approach 2, provide a classifier (.pkl) and optionally extractor weights.\n"
            '- Choose a checkpoint (Approach 1/3) or leave blank when using Approach 2 fallback.\n'
            '- Load an image and press "Run Inference".\n'
            '- Use "File -> Run Batch" to process a folder of images and export results.\n'
        )
        messagebox.showinfo('Help', msg)

    def load_model_if_needed(self, ckpt_path):
        # Avoid reload if already loaded for the same path
        if self.model is not None and getattr(self, 'loaded_ckpt', None) == ckpt_path and getattr(self, 'loaded_approach', None) == self.approach_var.get():
            return

        approach = self.approach_var.get()

        # For Approach 1 and 3, require a valid checkpoint path; Approach 2 doesn't need a checkpoint
        if approach in ('Approach 1', 'Approach 3'):
            if not ckpt_path or not os.path.exists(ckpt_path):
                messagebox.showerror('Error', f'Checkpoint not found: {ckpt_path}')
                return

        approach = self.approach_var.get()

        if approach == 'Approach 3':
            ckpt = torch.load(ckpt_path, map_location='cpu')
            backbone_name = ckpt.get('backbone_name', None) or ckpt.get('model', 'vit_base_patch14_reg4_dinov2.lvd142m')
            class_ids = ckpt['class_ids']
            emb_dim = ckpt.get('emb_dim', 512)
            num_classes = len(class_ids)

            backbone, preprocess = load_dino(backbone_name, DEVICE)
            in_dim = getattr(backbone, 'num_features', 768)
            proj = ProjectionHead(in_dim, emb_dim).to(DEVICE)
            clf = SpeciesClassifier(emb_dim, num_classes).to(DEVICE)
            proj.load_state_dict(ckpt['proj_state'])
            clf.load_state_dict(ckpt['clf_state'])
            proj.eval(); clf.eval()

            self.model = (backbone, proj, clf)
            self.preprocess = preprocess
            self.meta = {'classes': class_ids}
            self.loaded_ckpt = ckpt_path
            self.loaded_approach = approach

        elif approach == 'Approach 1':
            # Approach 1 uses cdna_pipeline models
            APP1_ROOT = os.path.join(ROOT, 'Approach 1', 'cdna_pipeline')
            if APP1_ROOT not in sys.path:
                sys.path.insert(0, APP1_ROOT)
            try:
                from models.feature_extractor import get_backbone
                from models.classifier import ClassifierHead
                from utils.transforms import get_transforms
            except Exception as e:
                messagebox.showerror('Import Error', f'Could not import Approach 1 modules: {e}')
                return

            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Build backbone
            backbone_name = ckpt.get('backbone_name', 'dinov2')
            F, feat_dim = get_backbone(backbone_name)
            num_classes = ckpt.get('num_classes', None)
            class_ids = ckpt.get('class_ids', None)
            if num_classes is None and class_ids is not None:
                num_classes = len(class_ids)
            if num_classes is None:
                messagebox.showerror('Error', 'Could not determine num_classes from checkpoint')
                return

            C = ClassifierHead(feat_dim, num_classes)
            # Move to device and load
            F_backbone = F.to(DEVICE)
            C = C.to(DEVICE)
            try:
                F_backbone.load_state_dict(ckpt['feature_extractor_state_dict'])
                C.load_state_dict(ckpt['classifier_state_dict'])
            except Exception as e:
                messagebox.showerror('Error', f'Failed to load Approach 1 checkpoint: {e}')
                return
            F_backbone.eval(); C.eval()
            preprocess = get_transforms(train=False, backbone='dinov2' if 'dinov2' in backbone_name else 'resnet')
            # Setup meta
            if class_ids is None:
                class_ids = [str(i) for i in range(num_classes)]

            self.model = (F_backbone, C)
            self.preprocess = preprocess
            self.meta = {'classes': class_ids}
            self.loaded_ckpt = ckpt_path
            self.loaded_approach = approach

        elif approach == 'Approach 2':
            # Approach 2: extractor (DINOv2) + sklearn classifier
            APP2_ROOT = os.path.join(ROOT, 'Approach 2', 'Approach2_v1', 'src')
            if APP2_ROOT not in sys.path:
                sys.path.insert(0, APP2_ROOT)
            try:
                from extractor_dinov2 import load_dinov2_feature_extractor, get_transform
            except Exception as e:
                messagebox.showerror('Import Error', f'Could not import Approach 2 extractor: {e}')
                return

            # extractor weights and classifier file provided separately
            extractor_path = self.extractor_weights_var.get().strip() or None
            clf_path = self.classifier_var.get().strip()
            if not clf_path or not os.path.exists(clf_path):
                messagebox.showerror('Error', f'Classifier file not found: {clf_path}')
                return

            # load extractor (returns model on device). If user didn't provide extractor weights,
            # fall back to using the Approach 3 DINOv2 backbone available in the repo for demo purposes.
            if extractor_path and os.path.exists(extractor_path):
                feat_model, feat_dim = load_dinov2_feature_extractor(extractor_path, device=str(DEVICE))
            else:
                # fallback: use Approach 3's load_dino
                try:
                    from dino_model import load_dino as load_dino_fallback
                    # use default model name
                    feat_model, _ = load_dino_fallback('vit_base_patch14_reg4_dinov2.lvd142m', DEVICE)
                    feat_dim = getattr(feat_model, 'num_features', None) or 768
                except Exception as e:
                    messagebox.showerror('Error', f'No extractor provided and fallback failed: {e}')
                    return

            # load classifier (joblib)
            try:
                import joblib
                pack = joblib.load(clf_path)
            except Exception as e:
                messagebox.showerror('Error', f'Failed to load classifier (joblib): {e}')
                return

            clf = pack.get('model', pack)
            # classifier classes
            if 'classes' in pack:
                clf_classes = list(pack['classes'])
            elif hasattr(clf, 'classes_'):
                clf_classes = [int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x for x in clf.classes_]
            else:
                messagebox.showerror('Error', 'Could not determine classifier classes')
                return

            # Try to build ID remapping if classifier classes look like compact 0..K-1 and
            # Approach2 features/train.npz exists containing original species ids in 'y'.
            try:
                import numpy as _np
                feats_path = os.path.join(ROOT, 'Approach 2', 'Approach2_v1', 'features', 'train.npz')
                if os.path.exists(feats_path):
                    arr = _np.load(feats_path)
                    y_train = arr.get('y', None)
                    if y_train is not None:
                        # unique ordering
                        uniq = sorted(_np.unique(y_train).tolist())
                        # if classifier classes are compact 0..K-1 and lengths match, map
                        if all(isinstance(x, int) for x in clf_classes) and min(clf_classes) >= 0:
                            k = len(clf_classes)
                            if len(uniq) == k and set(clf_classes) == set(range(k)):
                                # map index -> original species id
                                mapped = [int(u) for u in uniq]
                                clf_classes = mapped
            except Exception:
                pass

            # preprocess transform
            preprocess = get_transform(518)

            self.model = (feat_model, clf)
            self.preprocess = preprocess
            self.meta = {'classes': clf_classes}
            self.loaded_ckpt = ckpt_path
            self.loaded_approach = approach

        else:
            messagebox.showerror('Error', f'Unknown approach: {approach}')
            return

    

    def show_examples_for_classes(self, class_list):
        # Clear previous
        for w in self.examples_frame.winfo_children():
            w.destroy()

        data_root = self.data_root_var.get() or os.path.join(ROOT, 'AML_project_herbarium_dataset')
        if not os.path.isdir(data_root):
            ttk.Label(self.examples_frame, text=f'Data root not found: {data_root}').pack()
            return

        # For each class, show up to 4 examples in a row with a label
        for row_idx, clsname in enumerate(class_list):
            # class identifiers may be numeric (e.g. numpy int64); ensure they are strings for path joining
            cls_str = str(clsname)
            ttk.Label(self.examples_frame, text=f'Top-{row_idx+1}: {cls_str}', font=('TkDefaultFont', 10, 'bold')).grid(row=row_idx*2, column=0, columnspan=6, sticky='w', pady=(6,0))
            examples = []
            for sub in ('train/herbarium', 'train/photo'):
                d = os.path.join(data_root, sub, cls_str)
                if os.path.isdir(d):
                    for f in os.listdir(d)[:4]:
                        examples.append(os.path.join(d, f))

            if not examples:
                ttk.Label(self.examples_frame, text='  No example images found for class').grid(row=row_idx*2+1, column=0, sticky='w')
                continue

            for i, ex in enumerate(examples[:4]):
                try:
                    im = Image.open(ex).convert('RGB')
                    im.thumbnail((120, 120))
                    tkim = ImageTk.PhotoImage(im)
                    lbl = ttk.Label(self.examples_frame, image=tkim)
                    lbl.image = tkim
                    lbl.grid(row=row_idx*2+1, column=i, padx=4, pady=4)
                except Exception:
                    ttk.Label(self.examples_frame, text=ex).grid(row=row_idx*2+1, column=i)


def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
