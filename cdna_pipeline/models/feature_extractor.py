import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(name="dinov2", pretrained=True):
    if name == "dinov2":
        # Load DINOv2 - you might have a different version
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Check what model you actually loaded
        print(f"✅ DINO model class: {model.__class__.__name__}")
        
        # DINO models output features directly, but let's verify the dimension
        # Create a test forward pass to get the actual feature dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = model(test_input)
            actual_feature_dim = test_output.shape[1]
            print(f"✅ Actual feature dimension: {actual_feature_dim}")
        
        feature_dim = actual_feature_dim
        return model, feature_dim

    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        layers = list(model.children())[:-1]  # remove classification layer
        feature_extractor = nn.Sequential(*layers)
        feature_dim = 2048
        print("✅ Using ResNet50 backbone")
        return feature_extractor, feature_dim

    else:
        raise NotImplementedError(f"Backbone {name} not supported.")