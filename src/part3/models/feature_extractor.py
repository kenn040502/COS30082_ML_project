# models/feature_extractor.py
import torch
import torch.nn as nn
import timm

class FeatureWrapper(nn.Module):
    """
    Wrap a timm model so forward(x) returns a pooled feature vector (B, D).
    Handles ViT outputs: (B, N, D) → pooled (CLS or mean over patches).
    """
    def __init__(self, model, pool_type="cls"):
        super().__init__()
        self.model = model
        self.pool_type = pool_type

    def forward(self, x):
        out = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
        
        if isinstance(out, tuple):
            out = out[0]

        if out.dim() == 3:
            if self.pool_type == "cls":
                pooled = out[:, 0, :]
            else:
                pooled = out[:, 1:, :].mean(dim=1)
            return pooled
        elif out.dim() == 2:
            return out
        else:
            return out.view(out.size(0), -1)

def get_backbone(freeze_mode="partial", pretrained=True):
    """
    Returns DINOv2 backbone and feature dim.
    
    Args:
        freeze_mode (str): "full", "partial", or "none" (fully trainable)
        pretrained (bool): use pre-trained weights
    """
    print(f"✅ Loading DINOv2 ViT-B/14 backbone (pretrained={pretrained})...")
    model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
    base = timm.create_model(model_name, pretrained=pretrained)

    feature_dim = getattr(base, "embed_dim", 768)

    # ===== Freezing Logic =====
    if freeze_mode == "full":
        for param in base.parameters():
            param.requires_grad = False
        print("🛑 Backbone fully frozen.")
    elif freeze_mode == "partial":
        for name, param in base.named_parameters():
            if "blocks.11" in name or "blocks.10" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ Backbone partially unfrozen (last 2 blocks + norm layers).")
    elif freeze_mode == "none":
        for param in base.parameters():
            param.requires_grad = True
        print("✅ Backbone fully trainable.")
    else:
        raise ValueError(f"Invalid freeze_mode '{freeze_mode}'. Choose from 'full', 'partial', 'none'.")

    wrapper = FeatureWrapper(base, pool_type="cls")
    print(f"✅ DINOv2 backbone ready. Feature dim = {feature_dim}")
    return wrapper, feature_dim
