# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import math

from models.feature_extractor import get_backbone
from models.triplet_net import TripletNet, triplet_loss
from datasets import PlantTripletDataset, PlantTestDataset
from utils.metrics import evaluate_embeddings_detailed
from utils.class_utils import load_class_lists, get_class_type_mapping


def main():
    # =============================
    # CONFIG
    # =============================
    data_root = "./AML_project_herbarium_dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    num_epochs = 30

    base_lr = 1e-4
    backbone_lr_ratio = 0.01
    lr_decay_factor = 0.5
    patience = 3
    warmup_epochs = 2                     # <<<<<< WARM-UP EPOCHS
    margin = 0.2

    # =============================
    # TRANSFORMS
    # =============================
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =============================
    # DATASETS
    # =============================
    train_dataset = PlantTripletDataset(data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    test_dataset = PlantTestDataset(data_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    # =============================
    # CLASS TYPE SETUP
    # =============================
    classes_with_pairs, classes_without_pairs = load_class_lists(data_root)
    class_type_mapping = get_class_type_mapping(
        PlantTripletDataset.global_class_to_idx,
        classes_with_pairs,
        classes_without_pairs
    )

    # =============================
    # MODEL
    # =============================
    freeze_mode = "partial"
    feature_extractor, feat_dim = get_backbone(
        freeze_mode=freeze_mode,
        pretrained=True
    )

    model = TripletNet(
        feature_extractor,
        feat_dim=feat_dim,
        embedding_dim=128
    ).to(device)

    # =============================
    # OPTIMIZER
    # =============================
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(),
         'lr': base_lr * backbone_lr_ratio},
        {'params': model.fc.parameters(), 'lr': base_lr},
    ], weight_decay=1e-5)

    # =============================
    # BASELINE PERFORMANCE
    # =============================
    with torch.no_grad():
        baseline = evaluate_embeddings_detailed(
            model, test_loader, device, data_root,
            PlantTripletDataset.global_class_to_idx
        )

    best_top1 = baseline["overall_top1"]
    best_top5 = baseline["overall_top5"]
    best_loss = math.inf

    print("=" * 60)
    print(f"📊 Baseline Performance:")
    print(f"   Top-1: {baseline['overall_top1']:.2f}%")
    print(f"   Top-5: {baseline['overall_top5']:.2f}%")
    print("=" * 60)

    last_improve_epoch = 0

    # =============================
    # TRAINING LOOP
    # =============================
    patience_counter = 0  # <<<<<<<<< count epochs without improvement

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # ===============================================
        # WARM-UP LR ADJUSTMENT
        # ===============================================
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs  # e.g., 0.5 → 1.0

            for g in optimizer.param_groups:
                if g['lr'] >= base_lr:  # FC layer
                    g['lr'] = base_lr * warmup_factor
                else:                   # backbone
                    g['lr'] = (base_lr * backbone_lr_ratio) * warmup_factor

            print(f"\n🔥 Warm-up Epoch {epoch+1}/{warmup_epochs}")
            print(f"   → Warmup Factor: x{warmup_factor:.2f}")

        # ===============================================
        # SHOW CURRENT LEARNING RATES
        # ===============================================
        backbone_lr = optimizer.param_groups[0]['lr']
        fc_lr = optimizer.param_groups[1]['lr']

        print(f"\n📈 LR Status (Epoch {epoch+1}):")
        print(f"   Backbone LR: {backbone_lr:.6f}")
        print(f"   FC Head LR : {fc_lr:.6f}")
        print(f"   Patience Counter: {patience_counter}/{patience}\n")

        # Hard mining only AFTER warm-up
        use_hard = epoch >= max(warmup_epochs, num_epochs // 2)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)

            loss = triplet_loss(
                anchor_emb, pos_emb, neg_emb,
                margin=margin,
                hard_mining=use_hard
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # =============================
        # EVALUATION
        # =============================
        model.eval()
        metrics = evaluate_embeddings_detailed(
            model, test_loader, device, data_root,
            PlantTripletDataset.global_class_to_idx
        )

        top1 = metrics["overall_top1"]
        top5 = metrics["overall_top5"]

        print(f"🎯 Epoch {epoch+1} Results:")
        print(f"   Top-1: {top1:.2f}%")
        print(f"   Top-5: {top5:.2f}%")

        # =============================
        # SAVE BEST MODELS
        # =============================
        improved = False

        if top1 > best_top1:
            best_top1 = top1
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_top1.pth")
            print(f"🏆 NEW BEST Top-1: {best_top1:.2f}%")
            improved = True

        if top5 > best_top5:
            best_top5 = top5
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_top5.pth")
            print(f"🏆 NEW BEST Top-5: {best_top5:.2f}%")
            improved = True

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_loss.pth")
            print(f"📉 NEW BEST Loss: {best_loss:.4f}")
            improved = True

        # =============================
        # PATIENCE + MANUAL LR DECAY
        # =============================
        if not improved:
            patience_counter += 1
            print(f"⚠️ No improvement this epoch (patience {patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\n🔻 Patience reached → Reducing LR by 0.5\n")
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay_factor
            patience_counter = 0  # reset counter

    print("\n🎉 Training finished!")
    print(f"🏆 Best Top-1: {best_top1:.2f}%")
    print(f"🏆 Best Top-5: {best_top5:.2f}%")
    print(f"📉 Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
