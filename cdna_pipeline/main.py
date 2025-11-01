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
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        feats = F(images)
        outputs = C(feats)
        preds = torch.argmax(outputs, dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

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
    mean_acc = np.mean(all_acc)
    overall_acc = 100.0 * total_correct / total_samples
    print(f"📊 Mean per-class accuracy: {mean_acc:.2f}%")
    print(f"🎯 Overall test accuracy:   {overall_acc:.2f}%")

    return all_acc, overall_acc


def main():
    # =============== CONFIGURATION ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # ===== Choose backbone =====
    backbone_name = "dinov2"  # "resnet50" or "dinov2"

    # ===== Dataset and transforms =====
    data_root = "./AML_project_herbarium_dataset"
    train_tf = get_transforms(train=True, backbone=backbone_name)
    test_tf = get_transforms(train=False, backbone=backbone_name)

    # Initialize class mapping
    _ = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)

    # ===== Datasets =====
    source_dataset = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)
    target_dataset = PlantFolderDataset(data_root, domain='photo', split='train', transform=train_tf)
    val_dataset = PlantFolderDataset(data_root, domain='herbarium', split='val', transform=test_tf)
    test_dataset = PlantTestDataset(data_root, transform=test_tf)

    # ===== Dataloaders =====
    batch_size = 16 if "resnet50" in backbone_name else 2
    num_workers = 2

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_shared = len(PlantFolderDataset.global_class_to_idx)
    class_names = getattr(PlantFolderDataset, 'global_idx_to_class', None)

    print(f"✅ Source (herbarium train): {len(source_dataset)} samples")
    print(f"✅ Target (photo train): {len(target_dataset)} samples")
    print(f"✅ Shared classes: {num_shared}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")

    # =============== MODEL ===============
    F, feat_dim = get_backbone(backbone_name)
    C = ClassifierHead(feat_dim, num_classes=num_shared)
    # after F, feat_dim, C creation
    # if using CDAN (i.e. ViT / dinov2), the CDAN input per sample is feat_dim * num_classes
    if "dinov2" in backbone_name.lower() or "vit" in backbone_name.lower():
        # CDAN
        input_dim_for_D = feat_dim * num_shared
        use_ln = True
    else:
        # DANN
        input_dim_for_D = feat_dim
        use_ln = False

    D = DomainDiscriminator(input_dim=input_dim_for_D, hidden_dim=512 if use_ln else 1024, use_layernorm=use_ln)

    F, C, D = F.to(device), C.to(device), D.to(device)

    # =============== TRAINING =======
    losses = get_losses()
    optimizer = optim.Adam(
        itertools.chain(F.parameters(), C.parameters(), D.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Pass backbone_type to Trainer for CDAN
    trainer = Trainer(F, C, D, losses, optimizer, device, backbone_type=backbone_name)
    lambda_domain = 0.3

    num_epochs = 30
    best_accuracy = 0.0

    print("\n🚀 Starting Partial Domain Adaptation Training...")

    for epoch in range(num_epochs):
        total_cls_loss, total_dom_loss, total_train_acc, batches = 0.0, 0.0, 0.0, 0
        num_batches = max(len(source_loader), len(target_loader))

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for s_batch, t_batch in itertools.zip_longest(source_loader, target_loader):
                if s_batch is None or t_batch is None:
                    continue

                stats = trainer.train_step(
                    s_batch, t_batch,
                    epoch=epoch, batch_idx=batches,
                    num_batches=num_batches,
                    num_epochs=num_epochs,
                    max_lambda=lambda_domain
                )

                total_cls_loss += stats['train_loss']
                total_dom_loss += stats['domain_loss']
                total_train_acc += stats['train_acc']
                batches += 1

                pbar.set_postfix({
                    "Cls Loss": f"{total_cls_loss / batches:.3f}",
                    "Dom Loss": f"{total_dom_loss / batches:.3f}",
                    "Train Acc": f"{total_train_acc / batches:.2f}%"
                })
                pbar.update(1)

        scheduler.step()

        # Evaluate source and target validation
        val_src_top1, _ = evaluate_model(F, C, val_loader, device)
        val_tgt_top1, _ = evaluate_model(F, C, target_loader, device)

        print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train Acc: {total_train_acc / batches:.2f}% | Source Val Acc: {val_src_top1:.2f}% | Target Val Acc: {val_tgt_top1:.2f}%")

        # Save best model based on target validation
        if val_tgt_top1 > best_accuracy:
            best_accuracy = val_tgt_top1
            trainer.save_checkpoint(epoch+1, "checkpoints/best_model.pth", stats)
            print(f"🎉 New best model saved (Target Val Acc: {val_tgt_top1:.2f}%)")

        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch+1, f"checkpoints/epoch_{epoch+1}.pth", stats)

        print("-" * 50)

    # =============== FINAL EVALUATION ===============
    print("\n🎯 Loading Best Model for Evaluation...")
    best_ckpt_path = "checkpoints/best_model.pth"

    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        if "feature_extractor_state_dict" in checkpoint and "classifier_state_dict" in checkpoint:
            F.load_state_dict(checkpoint["feature_extractor_state_dict"])
            C.load_state_dict(checkpoint["classifier_state_dict"])
        else:
            print("⚠️ Checkpoint missing expected keys; using current weights.")
        print(f"✅ Loaded best model from epoch {checkpoint.get('epoch', '?')} with Val Acc = {best_accuracy:.2f}%")
    else:
        print("⚠️ No best_model.pth found, using final epoch weights.")

    F.eval()
    C.eval()

    print("\n🌿 Validation Evaluation:")
    val_top1, val_top5 = evaluate_model(F, C, val_loader, device)
    print(f"🌿 Validation Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")

    print("\n🌳 Test Evaluation (Photo Domain):")
    per_class_acc, test_acc = evaluate_per_class(F, C, test_loader, device, class_names)
    print(f"\n🎯 Overall Test Accuracy: {test_acc:.2f}%")

    print("\n🐢 Lowest 10 performing classes:")
    sorted_acc = sorted(list(enumerate(per_class_acc)), key=lambda x: x[1])
    for i, acc in sorted_acc[:10]:
        cname = class_names[i] if class_names else f"Class {i}"
        print(f"{cname:<25} | {acc:6.2f}%")

    print(f"\n🏆 Best Val Accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
