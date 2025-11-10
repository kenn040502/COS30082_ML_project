from __future__ import annotations
import os, hashlib, torch

def hash_config(model: str, pretrained: str, templates: list[str], class_ids: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model.encode()); h.update(pretrained.encode())
    h.update(("||".join(templates)).encode()); h.update(("||".join(class_ids)).encode())
    return h.hexdigest()[:16]

def save_text_checkpoint(path: str, text_feats: torch.Tensor, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"text_cls_feats": text_feats.cpu(), "meta": meta}, path)
    print(f"💾 Saved text checkpoint: {path}")

def load_text_checkpoint(path: str):
    if os.path.isfile(path):
        obj = torch.load(path, map_location="cpu")
        print(f"📦 Loaded text checkpoint: {path}")
        return obj
    return None
