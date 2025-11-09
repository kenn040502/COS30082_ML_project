# cdna_pipeline/models/feature_extractor.py
import torch
import torch.nn as nn
import timm

class FeatureWrapper(nn.Module):
    """Wrap timm model so forward(x) returns a pooled feature vector (B, D)."""
    def __init__(self, model, pool_type: str = "mean"):
        super().__init__()
        self.model = model
        self.pool_type = pool_type

    def forward(self, x):
        out = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 3:
            if self.pool_type == "cls":
                return out[:, 0, :]
            return out.mean(dim=1)
        if out.dim() == 2:
            return out
        return out.view(out.size(0), -1)

def get_backbone(name: str = "dinov2", pretrained: bool = True, freeze_all: bool = True):
    """
    Returns a DINOv2 feature extractor backbone and its feature dimension.
    (ResNet support removed.)
    """
    if "dinov2" not in name.lower():
        raise NotImplementedError("Only 'dinov2' backbone is supported now.")

    print("✅ Loading DINOv2 (timm) ViT-B/14 backbone…")
    model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
    base = timm.create_model(model_name, pretrained=pretrained)

    if freeze_all:
        for p in base.parameters():
            p.requires_grad = False
        print("🛑 DINOv2 backbone frozen.")
    else:
        # optional partial unfreeze
        for n, p in base.named_parameters():
            p.requires_grad = ("blocks.11" in n) or ("norm" in n)

    feature_dim = getattr(base, "embed_dim", 768)
    projector = nn.Linear(feature_dim, 512)

    wrapper = nn.Sequential(
        FeatureWrapper(base, pool_type="mean"),
        projector,
    )
    print("✅ DINOv2 backbone ready. Feature dim = 512 (after projection)")
    return wrapper, 512
