# cdna_pipeline/test_simple.py  (or replace your existing test.py)

import torch
from torch.utils.data import DataLoader

# Use package imports so `python -m cdna_pipeline.test_simple` works
from cdna_pipeline.models.feature_extractor import get_backbone
from cdna_pipeline.models.classifier import LogisticRegressionHead
from cdna_pipeline.utils.transforms import get_transforms
from cdna_pipeline.utils.metrics import evaluate_model
from cdna_pipeline.datasets import PlantFolderDataset, PlantTestDataset


def simple_model_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # ---- Data (DINOv2 transforms) ----
    data_root = "./AML_project_herbarium_dataset"   # <-- set to your real path if different
    test_tf = get_transforms(train=False, backbone="dinov2")

    # Initialize global class mapping (needed to know num_classes)
    _ = PlantFolderDataset(data_root, domain="herbarium", split="train", transform=test_tf)
    num_classes = len(PlantFolderDataset.global_class_to_idx)

    # Test set (uses list/groundtruth.txt or whatever your PlantTestDataset expects)
    test_dataset = PlantTestDataset(data_root, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ---- Model: DINOv2 backbone + logistic head ----
    F, feat_dim = get_backbone("dinov2", freeze_all=True) 
    C = LogisticRegressionHead(feat_dim, num_classes=num_classes)

    # Move to device BEFORE loading weights
    F = F.to(device)
    C = C.to(device)

    # ---- Load saved weights ----
    print("🧪 Loading best model...")
    try:
        ckpt_path = "checkpoints/final_model.pth"   # or "checkpoints/best_model.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)
        F.load_state_dict(checkpoint["feature_extractor_state_dict"])
        C.load_state_dict(checkpoint["classifier_state_dict"])
        print(f"✅ Model loaded from {ckpt_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ---- Evaluate ----
    print("🧪 Evaluating model...")
    top1, top5 = evaluate_model(F, C, test_loader, device)
    print(f"📊 Results — Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")


if __name__ == "__main__":
    simple_model_test()
