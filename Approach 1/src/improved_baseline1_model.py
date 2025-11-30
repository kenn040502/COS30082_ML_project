import torch
import torch.nn as nn
import timm


class ImprovedMixStreamCNN(nn.Module):
    """
    Improved mix-stream CNN for baseline-1.
    Still a single-stream architecture (valid baseline!), just better executed.
    """
    def __init__(self, model_name='convnext_small', num_classes=100, pretrained=True, dropout=0.5):
        super(ImprovedMixStreamCNN, self).__init__()
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=dropout
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Improved classification head with better regularization
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        print(f"Improved Mix-Stream CNN created:")
        print(f"   Backbone: {model_name}")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Classes: {num_classes}")
        print(f"   Dropout: {dropout}")
    
    def forward(self, x):
        """
        Forward pass - all images (herbarium and field) go through same network.
        This is what makes it "mix-stream"!
        """
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone for phase 1 training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        print("Backbone frozen, head trainable")
    
    def unfreeze_all(self):
        """Unfreeze everything for phase 2 training"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")
    
    def get_backbone_params(self):
        """Get backbone parameters (for differential learning rates)"""
        return self.backbone.parameters()
    
    def get_head_params(self):
        """Get head parameters (for differential learning rates)"""
        return self.head.parameters()


def create_improved_model(model_name='convnext_small', num_classes=100, pretrained=True, dropout=0.5):
    """
    Args:
        model_name: Model architecture
        num_classes: Number of plant species
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate for regularization
    
    Returns:
        model: ImprovedMixStreamCNN
    """
    model = ImprovedMixStreamCNN(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_improved_model(num_classes=100)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nModel test:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: (4, 100)")
    
    # Test freeze/unfreeze
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params (frozen): {trainable:,}")
    
    model.unfreeze_all()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params (unfrozen): {trainable:,}")