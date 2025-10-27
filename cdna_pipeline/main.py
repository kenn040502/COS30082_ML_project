import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from models.feature_extractor import get_backbone
from models.classifier import ClassifierHead
from models.cdan_module import DomainDiscriminator
from utils.losses import get_losses
from utils.trainer import Trainer
from utils.transforms import get_transforms
from utils.metrics import evaluate_model
from datasets import PlantFolderDataset, PlantTestDataset


@torch.no_grad()
def evaluate_per_class(F, C, loader, device, class_names=None):
    F.eval()
    C.eval()
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        feats = F(images)
        outputs = C(feats)
        preds = torch.argmax(outputs, dim=1)

        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            total_per_class[label] += 1
            if label == pred:
                correct_per_class[label] += 1

    all_acc = []
    for c in sorted(total_per_class.keys()):
        acc = 100.0 * correct_per_class[c] / total_per_class[c]
        all_acc.append(acc)
        cname = class_names[c] if class_names is not None else f"Class {c}"
        print(f"{cname:<25} | Acc: {acc:6.2f}% | Samples: {total_per_class[c]}")

    print("-" * 50)
    print(f"📊 Mean per-class accuracy: {np.mean(all_acc):.2f}%")
    return all_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # === Data Loading ===
    data_root = "./AML_project_herbarium_dataset"
    train_tf = get_transforms(train=True)
    test_tf = get_transforms(train=False)

    # Initialize global class mapping
    _ = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)

    # Datasets
    source_dataset = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)
    target_dataset = PlantFolderDataset(data_root, domain='photo', split='train', transform=train_tf)
    val_dataset = PlantFolderDataset(data_root, domain='herbarium', split='val', transform=test_tf)
    test_dataset = PlantTestDataset(data_root, transform=test_tf)

    # DataLoaders
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_shared = len(PlantFolderDataset.global_class_to_idx)
    class_names = getattr(PlantFolderDataset, 'global_idx_to_class', None)

    print(f"✅ Source (herbarium train): {len(source_dataset)} samples")
    print(f"✅ Target (photo train): {len(target_dataset)} samples")
    print(f"✅ Shared classes used (label space): {num_shared}")
    print(f"✅ Validation (domain={val_dataset.domain}) samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")

    # === Model Setup ===
    F, feat_dim = get_backbone("resnet50")
    C = ClassifierHead(feat_dim, num_classes=num_shared)
    D = DomainDiscriminator(feat_dim)
    F, C, D = F.to(device), C.to(device), D.to(device)

    # === Training Setup ===
    losses = get_losses()
    optimizer = optim.Adam(list(F.parameters()) + list(C.parameters()) + list(D.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = Trainer(F, C, D, losses, optimizer, device)
    lambda_domain = 0.1

    num_epochs = 100
    best_accuracy = 0.0

    print("\n🚀 Starting Partial Domain Adaptation Training...")

    for epoch in range(num_epochs):
        total_cls_loss, total_dom_loss, total_train_acc, batches = 0.0, 0.0, 0.0, 0

        # Progress bar for current epoch
        num_batches = max(len(source_loader), len(target_loader))
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for s_batch, t_batch in itertools.zip_longest(source_loader, target_loader):
                if s_batch is None or t_batch is None:
                    continue
                stats = trainer.train_step(s_batch, t_batch, lambda_=lambda_domain)
                total_cls_loss += stats['train_loss']
                total_dom_loss += stats['domain_loss']
                total_train_acc += stats['train_acc']
                batches += 1

                avg_cls = total_cls_loss / batches
                avg_dom = total_dom_loss / batches
                avg_acc = total_train_acc / batches
                pbar.set_postfix({
                    "Cls Loss": f"{avg_cls:.3f}",
                    "Dom Loss": f"{avg_dom:.3f}",
                    "Train Acc": f"{avg_acc:.2f}%"
                })
                pbar.update(1)

        avg_cls_loss = total_cls_loss / batches
        avg_dom_loss = total_dom_loss / batches
        avg_train_acc = total_train_acc / batches
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_top1, _ = evaluate_model(F, C, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_cls_loss:.3f}, Domain Loss: {avg_dom_loss:.3f}")
        print(f"  Train Acc: {avg_train_acc:.2f}% | Val Acc: {val_top1:.2f}%")

        if val_top1 > best_accuracy:
            best_accuracy = val_top1
            trainer.save_checkpoint(epoch+1, "checkpoints/best_model.pth", stats)
            print(f"🎉 New best model saved ({val_top1:.2f}%)")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch+1, checkpoint_path, stats)

        print("-" * 50)

    # === Final Evaluation (Validation + Test) ===
    print("\n🎯 Final Validation Evaluation:")
    val_top1, val_top5 = evaluate_model(F, C, val_loader, device)
    print(f"🌿 Validation Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")

    print("\n🌳 Test Evaluation (Photo Domain):")
    F.eval(); C.eval()
    per_class_acc = evaluate_per_class(F, C, test_loader, device, class_names)

    # Show bottom-10 performing classes
    print("\n🐢 Lowest 10 performing classes:")
    sorted_acc = sorted(list(enumerate(per_class_acc)), key=lambda x: x[1])
    for i, acc in sorted_acc[:10]:
        cname = class_names[i] if class_names else f"Class {i}"
        print(f"{cname:<25} | {acc:6.2f}%")

    print(f"\n🏆 Best Val Accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
