# models/classifier.py
import torch
import torch.nn as nn

class LogisticRegressionHead(nn.Module):
    """
    Simple linear classifier (logistic regression).
    Used for frozen feature extractors like DINOv2.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Pool or flatten if feature map is not already (B, D)
        if x.dim() == 3:
            x = x.mean(dim=1)  # pool tokens if ViT
        elif x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.fc(x)
