import torch
import torch.nn as nn
from torch.autograd import Function

# ===== Gradient Reversal Layer =====
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# ===== Conditional Feature Combination =====
def conditional_adversarial_features(features, outputs, mode="outer"):
    """
    Compute class-conditional domain features for CDAN.
    features: (B, D)
    outputs: (B, C)
    mode: "outer" for multilinear (official CDAN), "concat" for lightweight version.
    Returns: (B, D*C) or (B, D+C)
    """
    softmax_out = torch.softmax(outputs, dim=1)

    if mode == "outer":
        # Full multilinear conditioning (f ⊗ p)
        op = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))  # (B, C, D)
        return op.view(features.size(0), -1)
    elif mode == "concat":
        # Simpler concatenation (f || p)
        return torch.cat([features, softmax_out], dim=1)
    else:
        raise ValueError("Unknown mode: choose 'outer' or 'concat'.")


# ===== Domain Discriminator =====
class DomainDiscriminator(nn.Module):
    """
    Discriminator that accepts the actual input dimension.
    For DANN: input_dim = feat_dim
    For CDAN: input_dim = feat_dim * num_classes
    """
    def __init__(self, input_dim, hidden_dim=512, use_layernorm=False, dropout=0.3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        # Normalization choice
        if self.use_layernorm:
            norm = nn.LayerNorm(self.input_dim)
        else:
            try:
                norm = nn.BatchNorm1d(self.input_dim)
            except Exception:
                norm = nn.LayerNorm(self.input_dim)

        # Discriminator layers with dropout
        self.net = nn.Sequential(
            norm,
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, features, outputs=None, lambda_=1.0, mode="outer"):
        """
        features: feature tensor (B, D)
        outputs: logits tensor (B, C) if using CDAN
        lambda_: gradient reversal factor
        mode: 'outer' for CDAN, None for DANN
        """
        if outputs is not None:
            x = conditional_adversarial_features(features, outputs, mode)
        else:
            x = features

        x = grad_reverse(x, lambda_)
        return self.net(x)
