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
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 3:
            if self.pool_type == "cls":
                pooled = out[:, 0, :]
            else:
                pooled = out.mean(dim=1)
            return pooled
        elif out.dim() == 2:
            return out
        else:
            return out.view(out.size(0), -1)

def freeze_bn(model):
    """Sets all Batch Norm layers to eval mode and freezes parameters."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

def get_backbone(name="dinov2", pretrained=True, freeze_all=False):
    """
    Returns a feature extractor backbone and its feature dimension.
    Supports: DINOv2 (timm) and ResNet50.
    
    Args:
        freeze_all (bool): If True, freezes all parameters in the backbone.
                           This is critical for small-batch adversarial training stability.
    """
    name = name.lower()
    if "dinov2" in name:
        print(f"✅ Loading DINOv2 (timm) ViT-B/14 backbone...")
        model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
        base = timm.create_model(model_name, pretrained=pretrained)

        # ===== FREEZING LOGIC =====
        if freeze_all:
            # Full freeze: set all base model parameters to not require gradient
            for param in base.parameters():
                param.requires_grad = False
            print("🛑 DINOv2 Backbone fully frozen (requires_grad = False).")
        else:
            # Original partial fine-tuning logic (unfreezing last two blocks)
            for name, param in base.named_parameters():
                if "blocks.11" in name or "norm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("✅ DINOv2 Backbone partially unfrozen (last 2 blocks).")

        # Project 768 → 512 for better adaptation
        feature_dim = getattr(base, "embed_dim", 768)
        projector = nn.Linear(feature_dim, 512)

        # The projector is the only new layer, and it is trainable by default.
        # This layer will handle the gradient flow for the feature path.
        wrapper = nn.Sequential(
            FeatureWrapper(base, pool_type="mean"),
            projector
        )

        print("✅ Hugging Face DINOv2 backbone loaded. Feature dim = 512 (after projection)")
        return wrapper, 512

    elif name == "resnet50":
        print("✅ Loading ResNet50 backbone...")
        from torchvision import models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        layers = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*layers)
        feature_dim = 2048

        # ===== FREEZING BN FIX =====
        # Freeze BN layers to prevent small-batch instability (critical for small B=2/16)
        freeze_bn(feature_extractor)
        print("⚠️ ResNet50 BN layers frozen for small batch stability.")

        class ResNetWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor

            def forward(self, x):
                x = self.extractor(x)      # [B, 2048, 1, 1]
                x = torch.flatten(x, 1)    # -> [B, 2048]
                return x

        wrapper = ResNetWrapper(feature_extractor)
        print("✅ ResNet50 backbone ready. Feature dim = 2048")
        return wrapper, feature_dim

    else:
        raise NotImplementedError(f"Backbone '{name}' not supported.")