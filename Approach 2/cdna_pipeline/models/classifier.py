# models/classifier.py
import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # Use LayerNorm (works better with small batch sizes / transformer features)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),        # <- switched from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),         # reduced dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Ensure x is (B, D)
        if x.dim() == 3:
            # sequence-like, pool by mean
            x = x.mean(dim=1)
        elif x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.classifier(x)
