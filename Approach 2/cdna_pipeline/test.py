import torch
from torch.utils.data import DataLoader
from models.feature_extractor import get_backbone
from models.classifier import ClassifierHead
from utils.transforms import get_transforms
from utils.metrics import evaluate_model
from datasets import PlantDomainDataset

def simple_model_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Load test data
    data_root = "./AML_project_herbarium_dataset"
    test_tf = get_transforms(train=False)
    test_dataset = PlantDomainDataset(data_root, split='test', transform=test_tf)
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