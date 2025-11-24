import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.feature_extractor import get_backbone
from models.classifier import LogisticRegressionHead
from models.cdan_module import DomainDiscriminator   # DANN version
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
        cname = class_names[c] if class_names else f"Class {c}"
        print(f"{cname:<25} | Acc: {acc:6.2f}%")

    mean_acc = np.mean(all_acc)
    overall_acc = 100.0 * total_correct / total_samples
    print(f"\n📊 Mean per-class accuracy: {mean_acc:.2f}%")
    print(f"🎯 Overall test accuracy:   {overall_acc:.2f}%")

    return all_acc, overall_acc


def plot_metric(values, epochs, title, ylabel):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    filename = f"metric_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"📈 Saved: {filename}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    backbone_name = "dinov2"
    data_root = "./AML_project_herbarium_dataset"

    train_tf = get_transforms(train=True, backbone=backbone_name)
    test_tf = get_transforms(train=False, backbone=backbone_name)

    _ = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)

    source_dataset = PlantFolderDataset(data_root, domain='herbarium', split='train', transform=train_tf)
    target_dataset = PlantFolderDataset(data_root, domain='photo', split='train', transform=train_tf)
    val_dataset    = PlantFolderDataset(data_root, domain='herbarium', split='val', transform=test_tf)
    test_dataset   = PlantTestDataset(data_root, transform=test_tf)

    batch_size = 2
    num_workers = 2

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_shared = len(PlantFolderDataset.global_class_to_idx)
    class_names = PlantFolderDataset.global_idx_to_class

    # =============== MODEL ===============
    F, feat_dim = get_backbone(backbone_name, freeze_all=True)
    C = LogisticRegressionHead(feat_dim, num_classes=num_shared)

    input_dim_for_D = feat_dim  # DANN uses raw feature dimension
    D = DomainDiscriminator(input_dim=input_dim_for_D, hidden_dim=512, use_layernorm=True)

    F, C, D = F.to(device), C.to(device), D.to(device)

    # =============== OPTIM ===============
    losses = get_losses()
    optimizer = optim.Adam(
        itertools.chain(F.parameters(), C.parameters(), D.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = Trainer(F, C, D, losses, optimizer, device, backbone_type=backbone_name)

    lambda_domain = 1.0
    num_epochs = 50
    best_acc = 0.0

    # ======================================
    # METRIC HISTORY
    # ======================================
    history = {
        "cls_loss": [],
        "dom_loss": [],
        "src_acc": [],
        "disc_acc": [],
        "val_src": [],
        "val_tgt": [],
    }

    print("\n🚀 Starting DANN Training...")

    for epoch in range(num_epochs):

        total_cls, total_dom, total_src, total_disc, total_lambda = 0, 0, 0, 0, 0
        batches = 0
        num_batches = max(len(source_loader), len(target_loader))

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:

            for s_batch, t_batch in itertools.zip_longest(source_loader, target_loader):
                if s_batch is None or t_batch is None:
                    continue

                stats = trainer.train_step(
                    s_batch, t_batch,
                    epoch=epoch, batch_idx=batches,
                    num_batches=num_batches, num_epochs=num_epochs,
                    max_lambda=lambda_domain
                )

                total_cls    += stats["train_loss"]
                total_dom    += stats["domain_loss"]
                total_src    += stats["train_acc"]
                total_disc   += stats["disc_acc"]
                total_lambda += stats["lambda_val"]

                batches += 1

                pbar.set_postfix({
                    "Cls": f"{total_cls / batches:.3f}",
                    "Dom": f"{total_dom / batches:.3f}",
                    "SrcAcc": f"{total_src / batches:.2f}%",
                    "DiscAcc": f"{total_disc / batches:.2f}%",
                    "λ": f"{total_lambda / batches:.3f}"
                })
                pbar.update(1)

        scheduler.step()

        # =====================
        # VALIDATION
        # =====================
        val_src, _ = evaluate_model(F, C, val_loader, device)
        val_tgt, _ = evaluate_model(F, C, target_loader, device)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Source Train Acc: {total_src/batches:.2f}%")
        print(f"  Domain Disc Acc: {total_disc/batches:.2f}%")
        print(f"  Source Val Acc: {val_src:.2f}%")
        print(f"  Target Val Acc: {val_tgt:.2f}%")

        # =====================
        # SAVE BEST
        # =====================
        if val_tgt > best_acc:
            best_acc = val_tgt
            trainer.save_checkpoint(epoch+1, "checkpoints/best_model.pth", stats)

        # =====================
        # STORE HISTORY
        # =====================
        history["cls_loss"].append(total_cls / batches)
        history["dom_loss"].append(total_dom / batches)
        history["src_acc"].append(total_src / batches)
        history["disc_acc"].append(total_disc / batches)
        history["val_src"].append(val_src)
        history["val_tgt"].append(val_tgt)

    # =============== FINAL EVAL ===============
    print("\n🎯 Loading Best Model...")
    ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        F.load_state_dict(checkpoint["feature_extractor_state_dict"])
        C.load_state_dict(checkpoint["classifier_state_dict"])

    print("\n🌿 Validation Evaluation:")
    val_top1, val_top5 = evaluate_model(F, C, val_loader, device)
    print(f"Val Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")

    print("\n📷 Test Evaluation (Photo Domain):")
    per_class_acc, test_acc = evaluate_per_class(F, C, test_loader, device, class_names)

    print(f"\n🏆 Best Target Val Accuracy: {best_acc:.2f}%")

    # ======================================
    # GENERATE METRIC GRAPHS
    # ======================================
    epochs = list(range(1, num_epochs + 1))

    plot_metric(history["cls_loss"], epochs, "Classification Loss", "Loss")
    plot_metric(history["dom_loss"], epochs, "Domain Loss", "Loss")
    plot_metric(history["src_acc"], epochs, "Source Train Accuracy", "Accuracy (%)")
    plot_metric(history["disc_acc"], epochs, "Domain Discriminator Accuracy", "Accuracy (%)")
    plot_metric(history["val_src"], epochs, "Source Validation Accuracy", "Accuracy (%)")
    plot_metric(history["val_tgt"], epochs, "Target Validation Accuracy", "Accuracy (%)")

    print("\n📊 All metric graphs generated successfully!")
    

if __name__ == '__main__':
    main()
