from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def load_openclip(model_name: str, pretrained: str, device: torch.device):
    import open_clip
    print(f"🔤 Loading OpenCLIP: {model_name} ({pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.to(device).eval(), preprocess, tokenizer

@torch.no_grad()
def encode_text_classes(model, tokenizer, device, prompts: list[str], num_classes: int, bs: int):
    feats = []
    for i in range(0, len(prompts), bs):
        toks = tokenizer(prompts[i:i+bs]).to(device)
        z = model.encode_text(toks)
        feats.append(F.normalize(z, dim=-1))
    feats = torch.cat(feats, 0)               # [C*T, D]
    T = len(prompts) // max(1, num_classes)
    if num_classes * T != feats.shape[0]:
        raise RuntimeError(f"Prompt count mismatch: C={num_classes}, T={T}, total={feats.shape[0]}")
    per_class = []
    for ci in range(num_classes):
        per_class.append(feats[ci*T:(ci+1)*T].mean(0))
    per_class = torch.stack(per_class, 0)     # [C, D]
    return torch.nn.functional.normalize(per_class, dim=-1)
