# cdna_pipeline/utils/transforms.py
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_transforms(train: bool = True, backbone: str = "dinov2"):
    """DINOv2-only preprocessing."""
    if backbone.lower() != "dinov2":
        raise NotImplementedError("Only 'dinov2' transforms are supported now.")

    if train:
        return transforms.Compose([
            transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(518, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
