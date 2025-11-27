import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.bn(self.conv(spatial)))
        x = x * spatial_att
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1./self.p)


class MixStreamCNN(nn.Module):
    """Mix-Stream CNN with CBAM and GeM improvements"""
    def __init__(self, model_name='convnext_small', num_classes=100, 
                 pretrained=True, dropout=0.2, use_cbam=True, use_gem=True):
        super(MixStreamCNN, self).__init__()
        
        print(f"\n{'='*60}")
        print(f"CREATING IMPROVED MIX-STREAM CNN")
        print(f"{'='*60}")
        
        # Create backbone with features_only=True to get spatial features!
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True,  # ← KEY FIX: Get spatial features!
            out_indices=[-1]     # Only get last layer
        )
        
        # Get feature dimension from the last layer
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, list):
                features = features[-1]
            self.has_spatial = len(features.shape) == 4
            self.feature_dim = features.shape[1]
        
        print(f"Backbone: {model_name}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Has spatial features: {self.has_spatial}")
        print(f"Dropout: {dropout}")
        
        # Attention module (CBAM)
        self.use_cbam = use_cbam and self.has_spatial
        if self.use_cbam:
            self.cbam = CBAM(channels=self.feature_dim, reduction=16)
            print(f"✅ CBAM attention added (+1.5-2% expected)")
        else:
            print(f"⚠️  CBAM NOT added (no spatial features)")
        
        # Pooling strategy
        self.use_gem = use_gem and self.has_spatial
        if self.has_spatial:
            if self.use_gem:
                self.pool = GeM(p=3)
                print(f"✅ GeM pooling added (+0.5-1% expected)")
            else:
                self.pool = nn.AdaptiveAvgPool2d(1)
                print(f"   Standard pooling used")
        else:
            print(f"⚠️  No pooling (backbone already pooled)")
        
        pool_features = self.feature_dim
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Count improvements
        improvements = []
        if self.use_cbam:
            improvements.append("CBAM")
        if self.use_gem:
            improvements.append("GeM")
        
        print(f"\nImprovements: {', '.join(improvements) if improvements else 'None'}")
        print(f"Expected gain: +{len(improvements) * 1.5:.1f}% vs baseline")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        if isinstance(features, list):
            features = features[-1]
        
        # Apply attention if available
        if self.use_cbam:
            features = self.cbam(features)
        
        # Apply pooling if needed
        if self.has_spatial:
            features = self.pool(features)
            if len(features.shape) > 2:
                features = features.flatten(1)
        
        # Classification
        return self.head(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.use_cbam:
            for param in self.cbam.parameters():
                param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        print("✅ Backbone + CBAM frozen, head trainable")
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("✅ All parameters unfrozen")
    
    def get_backbone_params(self):
        params = list(self.backbone.parameters())
        if self.use_cbam:
            params += list(self.cbam.parameters())
        return params
    
    def get_head_params(self):
        params = list(self.head.parameters())
        if self.has_spatial and self.use_gem:
            params += list(self.pool.parameters())
        return params


def create_improved_model(model_name='convnext_small', num_classes=100, 
                          pretrained=True, dropout=0.2, use_cbam=True, use_gem=True):
    return MixStreamCNN(model_name, num_classes, pretrained, dropout, use_cbam, use_gem)


if __name__ == "__main__":
    print("Testing model...")
    model = create_improved_model(num_classes=100)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Has CBAM: {model.use_cbam}")
    print(f"Has GeM: {model.use_gem}")