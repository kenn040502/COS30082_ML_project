# models/feature_extractor.py
import torch
import torch.nn as nn
import timm

class FeatureWrapper(nn.Module):
    """
    Wrap a timm model so forward(x) returns a pooled feature vector (B, D).
    If timm model returns (B, D) already, pass through.
    If returns (B, N, D), we pool (mean) or use cls token.
    """
    def __init__(self, model, pool_type="mean"):
        super().__init__()
        self.model = model
        self.pool_type = pool_type

    def forward(self, x):
        out = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
        # Some timm models return (B, D) after forward_features; some return (B, N, D)
        if isinstance(out, tuple):
            # sometimes it returns (feat, ...)
            out = out[0]
        if out.dim() == 3:
            # (B, N, D) -> pool
            if self.pool_type == "cls":
                pooled = out[:, 0, :]     # CLS token
            else:
                pooled = out.mean(dim=1)
            return pooled
        elif out.dim() == 2:
            return out
        else:
            # fallback: flatten
            return out.view(out.size(0), -1)


def get_backbone(name="dinov2", pretrained=True):
    """
    Returns a feature extractor backbone and its feature dimension.
    Supports: DINOv2 (timm) and ResNet50.
    """
    name = name.lower()
    if "dinov2" in name:
        print("✅ Loading DINOv2 (timm) ViT-B/14 backbone from Hugging Face...")
        model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
        base = timm.create_model(model_name, pretrained=pretrained)

        # Keep classifier head but we'll use forward_features/pooling
        # Determine feature dim robustly
        # Try to get an attribute that tells features
        if hasattr(base, "num_features") and base.num_features is not None:
            feature_dim = base.num_features
        elif hasattr(base, "embed_dim"):
            feature_dim = base.embed_dim
        else:
            # compute via dummy pass (on CPU to be safe)
            base.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feat = base.forward_features(dummy) if hasattr(base, "forward_features") else base(dummy)
                if isinstance(feat, tuple): feat = feat[0]
                if feat.dim() == 3:
                    feature_dim = feat.size(-1)
                elif feat.dim() == 2:
                    feature_dim = feat.size(1)
                else:
                    feature_dim = 768

        wrapper = FeatureWrapper(base, pool_type="mean")
        print(f"✅ Hugging Face DINOv2 backbone loaded. Feature dim = {feature_dim}")
        return wrapper, feature_dim

    elif name == "resnet50":
        print("✅ Loading ResNet50 backbone...")
        from torchvision import models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        layers = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*layers)
        feature_dim = 2048

        class ResNetWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor

            def forward(self, x):
                x = self.extractor(x)          # [B, 2048, 1, 1]
                x = torch.flatten(x, 1)        # -> [B, 2048]
                return x

        wrapper = ResNetWrapper(feature_extractor)
        print("✅ ResNet50 backbone ready. Feature dim = 2048")
        return wrapper, feature_dim


    else:
        raise NotImplementedError(f"Backbone '{name}' not supported.")
