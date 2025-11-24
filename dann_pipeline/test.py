import torch
import argparse
from torch.utils.data import DataLoader
from models.feature_extractor import get_backbone
from models.classifier import LogisticRegressionHead
from datasets import PlantFolderDataset, PlantTestDataset
from utils.transforms import get_transforms
import numpy as np
from collections import defaultdict


# ------------------------------
# Compute Top-k Accuracy
# ------------------------------
def topk_accuracy(outputs, labels, k=5):
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.sum().item()


# ------------------------------
# Per-class Accuracy + Top-1 & Top-5
# ------------------------------
def evaluate_test(F, C, loader, device, class_names=None):

    F.eval()
    C.eval()

    total = 0
    top1_correct = 0
    top5_correct = 0

    # Per class tracking
    correct_c = defaultdict(int)
    total_c = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            feats = F(images)
            outputs = C(feats)

            # -------- Top-1 --------
            _, pred1 = outputs.max(1)
            top1_correct += pred1.eq(labels).sum().item()

            # -------- Top-5 --------
            top5_correct += topk_accuracy(outputs, labels, k=5)

            total += labels.size(0)

            # -------- Per-class --------
            for y, p in zip(labels.cpu().numpy(), pred1.cpu().numpy()):
                total_c[y] += 1
                if y == p:
                    correct_c[y] += 1

    # Compute final metrics
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total

    print("\n==============================")
    print("📷 Test Evaluation (Photo Domain)")
    print("==============================\n")

    per_class_acc = []
    for cls_id in sorted(total_c.keys()):
        acc = 100.0 * correct_c[cls_id] / total_c[cls_id]
        cname = class_names[cls_id] if class_names else f"Class {cls_id}"
        print(f"{cname:<25} | Acc: {acc:6.2f}% | Samples: {total_c[cls_id]}")
        per_class_acc.append(acc)

    print("\n📊 Mean Per-Class Accuracy: {:.2f}%".format(np.mean(per_class_acc)))
    print("🎯 Overall Top-1 Accuracy: {:.2f}%".format(top1))
    print("🎯 Overall Top-5 Accuracy: {:.2f}%".format(top5))

    return top1, top5, per_class_acc


# ------------------------------
# Main testing script
# ------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth",
                        help="Path to checkpoint (.pth)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "./AML_project_herbarium_dataset"
    backbone_name = "dinov2"

    # Test transform (same as validation)
    test_tf = get_transforms(train=False, backbone=backbone_name)

    # ----------------------------------
    # IMPORTANT: initialize class mapping
    # ----------------------------------
    _ = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=test_tf)

    # Load dataset
    test_dataset = PlantTestDataset(data_root, transform=test_tf)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Class names
    num_classes = len(PlantFolderDataset.global_class_to_idx)
    class_names = PlantFolderDataset.global_idx_to_class

    # Load backbone + classifier
    F, feat_dim = get_backbone(backbone_name, freeze_all=True)
    C = LogisticRegressionHead(feat_dim, num_classes)

    F, C = F.to(device), C.to(device)

    # Load checkpoint
    print(f"🔍 Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    F.load_state_dict(checkpoint["feature_extractor_state_dict"])
    C.load_state_dict(checkpoint["classifier_state_dict"])

    # Run evaluation
    evaluate_test(F, C, test_loader, device, class_names)


if __name__ == "__main__":
    main()
