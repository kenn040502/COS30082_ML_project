import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Added for stability
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, 256),  # Additional layer
            nn.BatchNorm1d(256),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle different input shapes
        if x.dim() == 4:
            # [batch, channels, height, width] -> [batch, channels]
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            # [batch, channels, 1] -> [batch, channels]
            x = x.squeeze(-1)
        
        return self.classifier(x)