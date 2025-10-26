import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.feature_extractor import get_backbone
from models.classifier import ClassifierHead
from models.cdan_module import DomainDiscriminator
from utils.losses import get_losses
from utils.trainer import Trainer
from utils.transforms import get_transforms
from utils.metrics import evaluate_model
from datasets import PlantFolderDataset, PlantTestDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # === Data Loading ===
    data_root = "./AML_project_herbarium_dataset"
    train_tf = get_transforms(train=True)
    test_tf = get_transforms(train=False)

    # Source: Herbarium training set (100 classes)
    source_dataset = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)
    
    # Target: Photo images (60 classes - subset of source)
    target_dataset = PlantFolderDataset(data_root, domain='photo', split='train', transform=train_tf)
    
    # Validation: Herbarium validation set (100 classes)
    val_dataset = PlantFolderDataset(data_root, domain='photo', split='train', transform=test_tf)

    
    # Test set (unlabeled)
    test_dataset = PlantTestDataset(data_root, transform=test_tf)

    # DataLoaders
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_shared = len(PlantFolderDataset.global_class_to_idx)
    print(f"✅ Source (herbarium train): {len(source_dataset)} samples")
    print(f"✅ Target (photo train): {len(target_dataset)} samples")
    print(f"✅ Shared classes used (label space): {num_shared}")
    print(f"✅ Validation (domain={val_dataset.domain}) samples: {len(val_dataset)}")

    # === Model Setup ===
    F, feat_dim = get_backbone("resnet50")
    C = ClassifierHead(feat_dim, num_classes=len(PlantFolderDataset.global_class_to_idx))
    D = DomainDiscriminator(feat_dim)
    F, C, D = F.to(device), C.to(device), D.to(device)

    # === Training Setup ===
    losses = get_losses()
    optimizer = optim.Adam(
        list(F.parameters()) + list(C.parameters()) + list(D.parameters()), 
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = Trainer(F, C, D, losses, optimizer, device)

    # === Training ===
    num_epochs = 50
    best_accuracy = 0.0

    print("\n🚀 Starting Partial Domain Adaptation Training...")
    print("💡 Strategy: Train on herbarium (source), adapt to photo (target)")

    for epoch in range(num_epochs):
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_train_acc = 0.0
        batches = 0

        for s_batch, t_batch in zip(source_loader, target_loader):
            stats = trainer.train_step(s_batch, t_batch)
            total_cls_loss += stats['train_loss']
            total_dom_loss += stats['domain_loss']
            total_train_acc += stats['train_acc']
            batches += 1

        avg_cls_loss = total_cls_loss / batches
        avg_dom_loss = total_dom_loss / batches
        avg_train_acc = total_train_acc / batches

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # === Validation (Herbarium val) ===
        val_top1, _ = evaluate_model(F, C, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_cls_loss:.3f}, Domain Loss: {avg_dom_loss:.3f}")
        print(f"  Train Acc: {avg_train_acc:.2f}% | Val Acc: {val_top1:.2f}%")

        # === Save Best Model ===
        if val_top1 > best_accuracy:
            best_accuracy = val_top1
            trainer.save_checkpoint(epoch+1, "checkpoints/best_model.pth", stats)
            print(f"🎉 New best model saved ({val_top1:.2f}%)")

        # Save intermediate checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch+1, checkpoint_path, stats)

        print("-" * 50)

    # === Final Evaluation ===
    print("\n🎯 Final Evaluation:")
    val_top1, val_top5 = evaluate_model(F, C, val_loader, device)
    test_top1, test_top5 = evaluate_model(F, C, test_loader, device)

    print(f"🌿 Validation (100 classes): Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")
    print(f"🌳 Test (Field Images): Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}%")
    print(f"🏆 Best Val Accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
