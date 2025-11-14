from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ Gradient Reversal ------------

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)

# ------------ Projection + Heads ------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, emb_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=-1)

class SpeciesClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

class DomainDiscriminator(nn.Module):
    def __init__(self, emb_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        z_rev = grad_reverse(z, lambd)
        return self.net(z_rev)
