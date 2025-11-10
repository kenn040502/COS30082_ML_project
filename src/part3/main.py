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
from utils.metrics import evaluate_embeddings 

def main():
    # ===== IMPROVED CONFIG =====
    data_root = "./AML_project_herbarium_dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🚀 INCREASE BATCH SIZE - CRITICAL FOR HARD NEGATIVE MINING
    batch_size = 8  
    num_workers = 4
    num_epochs = 5  # Increased epochs
    
    # Learning rate adjustments
    base_lr = 1e-4
    backbone_lr_ratio = 0.01 
    
    margin = 0.2  # Reduced margin for better convergence

    # ===== TRANSFORMS =====
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.5),  # Added augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ===== DATASETS =====
    train_dataset = PlantTripletDataset(data_root, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = PlantTestDataset(data_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ===== MODEL =====
    freeze_mode = "full"  
    feature_extractor, feat_dim = get_backbone(freeze_mode=freeze_mode, pretrained=True)
    model = TripletNet(feature_extractor, feat_dim=feat_dim, embedding_dim=128).to(device)

    # ===== OPTIMIZER WITH GRADIENT CLIPPING =====
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': base_lr * backbone_lr_ratio},
        {'params': model.fc.parameters(), 'lr': base_lr}
    ], weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ===== IMPROVED TRACKING =====
    best_top1 = 0.0
    best_loss = math.inf
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"🚀 Training on {len(train_dataset)} triplets, {len(PlantTripletDataset.global_class_to_idx)} classes")
    print(f"📊 Initial Test Performance Baseline...")
    
    # Get baseline performance before training
    with torch.no_grad():
        top1_baseline, top5_baseline = evaluate_embeddings(model, test_loader, device)
    print(f"📈 Baseline - Top-1: {top1_baseline:.2f}% | Top-5: {top5_baseline:.2f}%")

    # ===== TRAINING LOOP =====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
            
            # 🚀 Use both regular and hard mining loss
            if epoch < num_epochs // 2:
                # First half: regular triplet loss
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, 
                                           margin=margin, hard_mining=False)
            else:
                # Second half: hard negative mining
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, 
                                           margin=margin, hard_mining=True)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # ===== EVALUATION =====
        model.eval()
        top1, top5 = evaluate_embeddings(model, test_loader, device)
        print(f"Test Accuracy: Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
        
        # ===== FIXED BEST MODEL TRACKING =====
        save_model = False
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_top1.pth"))
            print(f"🏆 NEW BEST Top-1: {top1:.2f}% - Model saved!")
            save_model = True
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not save_model:  # Don't save twice if both conditions met
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_loss.pth"))
            print(f"📉 NEW BEST Loss: {avg_loss:.4f} - Model saved!")

        # Update learning rate
        scheduler.step()

    # ===== FINAL SAVE =====
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    print(f"✅ Training completed!")
    print(f"🏆 Best Top-1 Accuracy: {best_top1:.2f}%")
    print(f"🏆 Best Loss: {best_loss:.4f}")
    print(f"📈 Improvement from baseline: +{best_top1 - top1_baseline:.2f}%")

if __name__ == "__main__":
    main()