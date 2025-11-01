# models/cdan_module.py
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


# ===== Domain Discriminator =====
class DomainDiscriminator(nn.Module):
    """
    Discriminator that accepts the actual input dimension.
    Use `input_dim = feat_dim` for DANN (plain features)
    or `input_dim = feat_dim * num_classes` for CDAN (conditional features).
    """
    def __init__(self, input_dim, hidden_dim=512, use_layernorm=False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        # Use LayerNorm for transformer-style inputs (small batch, dependent dims)
        if self.use_layernorm:
            norm = nn.LayerNorm(self.input_dim)
        else:
            # For very large input_dim BatchNorm1d expects that many channels,
            # but BatchNorm1d doesn't play well for single-sample batches; use LayerNorm fallback.
            # If input_dim is huge and you have tiny batch, prefer LayerNorm.
            try:
                norm = nn.BatchNorm1d(self.input_dim)
            except Exception:
                norm = nn.LayerNorm(self.input_dim)

        self.net = nn.Sequential(
            norm,
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, lambda_):
        # x expected shape: (B, input_dim)
        x = grad_reverse(x, lambda_)
        return self.net(x)
