import torch
import torch.nn as nn
from torch.autograd import Function

# ====================================================
#  Gradient Reversal Layer (same as CDAN, used by DANN)
# ====================================================

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


# ====================================================
#  DANN Domain Discriminator (NO conditional features)
# ====================================================

class DomainDiscriminator(nn.Module):
    """
    Pure DANN Discriminator.
    Input: feature vector (B, D)
    Output: domain prediction logits (B, 1)
    """

    def __init__(self, input_dim, hidden_dim=512, use_layernorm=False, dropout=0.3):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        # Normalization
        if self.use_layernorm:
            norm_layer = nn.LayerNorm(self.input_dim)
        else:
            try:
                norm_layer = nn.BatchNorm1d(self.input_dim)
            except Exception:
                norm_layer = nn.LayerNorm(self.input_dim)

        # Simple discriminator for DANN (features → hidden → 1)
        self.net = nn.Sequential(
            norm_layer,
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, features, lambda_=1.0):
        """
        DANN Forward:
        - reverse gradients
        - classify domain
        """
        reversed_features = grad_reverse(features, lambda_)
        out = self.net(reversed_features)
        return out
