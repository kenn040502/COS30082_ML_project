from __future__ import annotations
from typing import List
import random
import numpy as np
import torch
import torch.nn.functional as F

# ----- Class weights for imbalance -----

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)

# ----- Simple triplet mining -----

def triplet_loss_random(z: torch.Tensor, y: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Simple supervised triplet:
      for each class, construct random triplets (anchor, positive, negative).
    """
    device = z.device
    N = z.size(0)
    if N < 2:
        return torch.tensor(0.0, device=device)

    z = F.normalize(z, dim=-1)

    # group indices by class
    indices_by_class = {}
    for i in range(N):
        c = int(y[i].item())
        indices_by_class.setdefault(c, []).append(i)

    anchors, positives, negatives = [], [], []
    all_indices = list(range(N))

    for c, idxs in indices_by_class.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            pos_candidates = [k for k in idxs if k != i]
            if not pos_candidates:
                continue
            j = random.choice(pos_candidates)
            neg_candidates = [k for k in all_indices if y[k].item() != c]
            if not neg_candidates:
                continue
            k = random.choice(neg_candidates)
            anchors.append(i)
            positives.append(j)
            negatives.append(k)

    if not anchors:
        return torch.tensor(0.0, device=device)

    a = torch.tensor(anchors, device=device, dtype=torch.long)
    p = torch.tensor(positives, device=device, dtype=torch.long)
    n = torch.tensor(negatives, device=device, dtype=torch.long)

    za, zp, zn = z[a], z[p], z[n]
    d_ap = 1.0 - (za * zp).sum(dim=-1)  # cosine distance
    d_an = 1.0 - (za * zn).sum(dim=-1)

    loss = torch.relu(d_ap - d_an + margin).mean()
    return loss
