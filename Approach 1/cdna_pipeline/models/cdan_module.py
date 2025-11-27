import torch
import torch.nn as nn
from torch.autograd import Function


# Gradient Reversal Layer
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


# Domain Discriminator (updated for ViT features)
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),  # ✅ ViT features are already normalized — use LayerNorm
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, lambda_):
        x = grad_reverse(x, lambda_)
        return self.net(x)
