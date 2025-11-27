from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(train=True, backbone="resnet"):
    """
    Returns preprocessing transforms depending on the model backbone.
    Supports:
      - 'resnet': standard ImageNet preprocessing
      - 'dinov2': DINOv2 ViT preprocessing (larger crop + normalized to [-1, 1])
    """

    if backbone.lower() == "dinov2":
        if train:
            return transforms.Compose([
                transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(518, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # [-1, 1] range
            ])
        else:
            return transforms.Compose([
                transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

    # Default → ResNet / CNN-style
    else:
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
