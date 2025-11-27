import os
import torch
from torch.utils.data import DataLoader
from models.feature_extractor import get_backbone
from models.classifier import ClassifierHead
from utils.transforms import get_transforms
from utils.metrics import evaluate_model
from datasets import PlantFolderDataset, PlantTestDataset


def get_data_root():
    """Resolve dataset root from env or default to repo-local AML_project_herbarium_dataset."""
    env_root = os.environ.get("AML_DATA_ROOT")
    if env_root:
        return env_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "AML_project_herbarium_dataset"))

def simple_model_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Load test data
    data_root = get_data_root()
    print(f"Dataset root: {data_root}")
    test_tf = get_transforms(train=False)

    # Build shared class mapping before loading test set
    _ = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=test_tf)

    test_dataset = PlantTestDataset(data_root, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Recreate model architecture
    F, feat_dim = get_backbone("resnet50")
    C = ClassifierHead(feat_dim, num_classes=100)
    
    # FIX: Move models to device BEFORE loading weights
    F = F.to(device)
    C = C.to(device)
    
    # Load saved weights
    print("🧪 Loading best model...")
    try:
        checkpoint = torch.load("checkpoints/final_model.pth", map_location=device)  # map_location is key!
        F.load_state_dict(checkpoint['feature_extractor_state_dict'])
        C.load_state_dict(checkpoint['classifier_state_dict'])
        print("✅ Best model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load best model: {e}")
        return
    
    # Test the model
    print("🧪 Evaluating model...")
    target_top1, target_top5 = evaluate_model(F, C, test_loader, device)
    print(f"📊 Best Model Results - Top-1: {target_top1:.2f}%, Top-5: {target_top5:.2f}%")

if __name__ == '__main__':
    simple_model_test()
