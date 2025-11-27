import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
import json
import os

from improved_baseline1_model import create_improved_model
from improved_baseline1_dataloader import create_improved_dataloaders


class ClassBalancedLoss(nn.Module):
    """
    Weight loss by inverse class frequency.
    Helps with imbalanced dataset where some species have 1-2 samples.
    """
    def __init__(self, class_counts, beta=0.9999, label_smoothing=0.1):
        super().__init__()
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        
        self.register_buffer('weights', torch.FloatTensor(weights))
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, labels):
        return nn.functional.cross_entropy(
            logits, 
            labels, 
            weight=self.weights,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """
    Focal Loss - focuses on hard examples.
    Useful when model is confident on easy samples but struggles on hard ones.
    """
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MixupAugmentation:
    """
    Mixup data augmentation.
    Helps with generalization and regularization.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """Apply mixup to batch"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class ImprovedBaseline1Trainer:
    def __init__(self, model, train_loader, test_loader, device, dataset_info, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dataset_info = dataset_info
        self.config = config
        
        # Mixup augmentation
        self.use_mixup = config.get('mixup_alpha', 0) > 0
        self.mixup = MixupAugmentation(alpha=config['mixup_alpha']) if self.use_mixup else None
        
        # Loss function - Class Balanced or Focal or Standard
        if config.get('use_class_balanced_loss', False):
            # Calculate class counts
            class_counts = np.zeros(dataset_info['num_classes'])
            for _, label in train_loader.dataset.data:
                class_counts[label] += 1
            
            print(f"Using Class-Balanced Loss")
            print(f"   Class counts range: {class_counts.min():.0f} - {class_counts.max():.0f}")
            self.criterion = ClassBalancedLoss(
                class_counts, 
                beta=0.9999,
                label_smoothing=0.1
            ).to(device)
        
        elif config.get('use_focal_loss', False):
            print(f"Using Focal Loss")
            self.criterion = FocalLoss(
                alpha=1.0, 
                gamma=2.0,
                label_smoothing=0.1
            ).to(device)
        
        else:
            print(f"Using Standard Cross-Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Tracking
        self.best_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc_overall': [],
            'val_acc_top5': [],
            'val_acc_with_pairs': [],
            'val_acc_without_pairs': [],
            'learning_rates': [],
            'epoch_times': []  # NEW: Track time per epoch
        }
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = True
        
        print(f"\n{'='*60}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*60}")
        print(f"Mixup: {'Enabled (alpha=' + str(config['mixup_alpha']) + ')' if self.use_mixup else 'Disabled'}")
        print(f"TTA (training): {'Enabled' if config.get('use_tta_during_training', False) else 'Disabled (fast!)'}")
        print(f"TTA (final eval): {'Enabled (' + str(config.get('tta_crops', 5)) + ' crops)' if config.get('use_tta_final', True) else 'Disabled'}")
        print(f"Mixed precision: Enabled")
        print(f"Early stopping: {'Enabled (patience=' + str(config.get('early_stopping_patience', 0)) + ')' if config.get('early_stopping_patience', 0) > 0 else 'Disabled'}")
        print(f"{'='*60}\n")
    
    def train_epoch(self, optimizer, epoch, use_mixup_this_epoch=True):
        """Train one epoch with optional mixup"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()  # Track epoch time
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply mixup with 50% probability
            if self.use_mixup and use_mixup_this_epoch and np.random.random() > 0.5:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
            else:
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            
            # Optimization with gradient scaling
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, epoch_time
    
    def predict_with_tta(self, images, num_crops=5):
        """
        Test-Time Augmentation: apply multiple augmentations and average predictions.
        
        Args:
            images: Batch of images [B, C, H, W]
            num_crops: Number of crops to use (1=center only, 5=center+4corners, 10=all+flips)
        
        Returns:
            Average probabilities across all augmentations
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # 1. Original image (center crop)
            outputs = self.model(images.to(self.device))
            all_predictions.append(torch.softmax(outputs, dim=1))
            
            # 2. Horizontal flip
            flipped = torch.flip(images, dims=[3])
            outputs = self.model(flipped.to(self.device))
            all_predictions.append(torch.softmax(outputs, dim=1))
            
            if num_crops >= 5:
                # 3-6. Four corners
                h, w = images.size(2), images.size(3)
                crop_size = min(h, w)
                
                corners = [
                    (0, 0),                              # Top-left
                    (0, w - crop_size),                  # Top-right
                    (h - crop_size, 0),                  # Bottom-left
                    (h - crop_size, w - crop_size),      # Bottom-right
                ]
                
                for top, left in corners:
                    cropped = images[:, :, top:top+crop_size, left:left+crop_size]
                    outputs = self.model(cropped.to(self.device))
                    all_predictions.append(torch.softmax(outputs, dim=1))
            
            if num_crops >= 10:
                # 7-10. Four corners flipped
                for top, left in corners:
                    cropped = images[:, :, top:top+crop_size, left:left+crop_size]
                    flipped = torch.flip(cropped, dims=[3])
                    outputs = self.model(flipped.to(self.device))
                    all_predictions.append(torch.softmax(outputs, dim=1))
        
        # Average all predictions
        avg_predictions = torch.stack(all_predictions).mean(dim=0)
        return avg_predictions
    
    def evaluate(self, detailed=True, use_tta=None):
        """
        Evaluate with detailed metrics and optional TTA
        
        Args:
            detailed: Whether to print detailed results
            use_tta: Override config TTA setting (None = use config)
        """
        self.model.eval()
        
        # Determine if TTA should be used
        if use_tta is None:
            use_tta = self.config.get('use_tta_during_training', False)
        
        tta_crops = self.config.get('tta_crops', 5)
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        # Per-class tracking
        num_classes = self.dataset_info['num_classes']
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        # With/without pairs tracking
        classes_with = self.dataset_info['classes_with_pairs']
        classes_without = self.dataset_info['classes_without_pairs']
        
        correct_with = 0
        total_with = 0
        correct_without = 0
        total_without = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Evaluating{' (TTA)' if use_tta else ''}")
            for images, labels in pbar:
                labels = labels.to(self.device)
                
                # Use TTA if enabled
                if use_tta:
                    outputs = self.predict_with_tta(images, num_crops=tta_crops)
                else:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    outputs = torch.softmax(outputs, dim=1)
                
                # Top-1
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_top1 += predicted.eq(labels).sum().item()
                
                # Top-5
                _, top5_pred = outputs.topk(5, dim=1)
                for i in range(labels.size(0)):
                    if labels[i] in top5_pred[i]:
                        correct_top5 += 1
                    
                    # Per-class and with/without
                    label = labels[i].item()
                    pred = predicted[i].item()
                    
                    class_correct[label] += (pred == label)
                    class_total[label] += 1
                    
                    if label in classes_with:
                        total_with += 1
                        if pred == label:
                            correct_with += 1
                    elif label in classes_without:
                        total_without += 1
                        if pred == label:
                            correct_without += 1
        
        # Calculate accuracies
        top1_acc = 100. * correct_top1 / total if total > 0 else 0
        top5_acc = 100. * correct_top5 / total if total > 0 else 0
        acc_with = 100. * correct_with / total_with if total_with > 0 else 0
        acc_without = 100. * correct_without / total_without if total_without > 0 else 0
        
        class_accuracies = class_correct / (class_total + 1e-8)
        avg_class_acc = 100. * class_accuracies.mean().item()
        
        results = {
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'avg_class_acc': avg_class_acc,
            'acc_with_pairs': acc_with,
            'acc_without_pairs': acc_without,
        }
        
        if detailed:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            if use_tta:
                print(f"(Using TTA with {tta_crops} crops)")
            print("="*60)
            print(f"Top-1 Accuracy: {top1_acc:.2f}%")
            print(f"Top-5 Accuracy: {top5_acc:.2f}%")
            print(f"Avg Per-Class: {avg_class_acc:.2f}%")
            print(f"\nPerformance by Species Type:")
            print(f"  With pairs: {acc_with:.2f}%")
            print(f"  Without pairs: {acc_without:.2f}%")
            print(f"  Gap: {acc_with - acc_without:.2f}%")
            print("="*60 + "\n")
        
        return results
    
    def train_phase1(self, epochs=15, lr=1e-3, warmup_epochs=5):
        """
        Phase 1: Train classifier head only (backbone frozen).
        """
        print("\n" + "="*60)
        print("PHASE 1: Training Classifier Head")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {self.config['weight_decay']}")
        print(f"Warmup epochs: {warmup_epochs}")
        print()
        
        self.model.freeze_backbone()
        
        # Optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler with warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        
        phase_start_time = time.time()
        
        for epoch in range(epochs):
            # Warmup
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(optimizer, epoch + 1, use_mixup_this_epoch=False)
            
            # Step scheduler after warmup
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Log
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, LR: {current_lr:.6f}, Time: {epoch_time/60:.1f}min")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Evaluate periodically (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                results = self.evaluate(detailed=True)
                
                self.history['val_acc_overall'].append(results['top1_acc'])
                self.history['val_acc_top5'].append(results['top5_acc'])
                self.history['val_acc_with_pairs'].append(results['acc_with_pairs'])
                self.history['val_acc_without_pairs'].append(results['acc_without_pairs'])
                
                if results['top1_acc'] > self.best_accuracy:
                    self.best_accuracy = results['top1_acc']
                    self._save_checkpoint('improved_baseline1_best.pth', epoch, results)
                    print(f"New best: {results['top1_acc']:.2f}%\n")
        
        phase_time = time.time() - phase_start_time
        print(f"\nPhase 1 completed in {phase_time/60:.1f} minutes ({phase_time/3600:.2f} hours)")
    
    def train_phase2(self, epochs=35, backbone_lr=5e-6, head_lr=1e-4):
        """
        Phase 2: Fine-tune entire model with differential learning rates.
        """
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning Entire Model")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Backbone LR: {backbone_lr}")
        print(f"Head LR: {head_lr}")
        print(f"Weight decay: {self.config['weight_decay']}")
        print()
        
        self.model.unfreeze_all()
        
        # Differential learning rates
        optimizer = optim.AdamW([
            {'params': self.model.get_backbone_params(), 'lr': backbone_lr},
            {'params': self.model.get_head_params(), 'lr': head_lr}
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Early stopping
        patience = self.config.get('early_stopping_patience', 0)
        no_improve_count = 0
        
        phase_start_time = time.time()
        
        for epoch in range(epochs):
            # Start using mixup after a few epochs
            use_mixup = epoch >= 5
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(optimizer, epoch + 1, use_mixup_this_epoch=use_mixup)
            
            scheduler.step()
            
            # Log
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_head = optimizer.param_groups[1]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Time: {epoch_time/60:.1f}min")
            print(f"  LR: Backbone={current_lr_backbone:.6f}, Head={current_lr_head:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rates'].append(current_lr_head)
            self.history['epoch_times'].append(epoch_time)
            
            # Evaluate periodically
            eval_frequency = self.config.get('eval_frequency', 5)
            if (epoch + 1) % eval_frequency == 0 or epoch == epochs - 1:
                results = self.evaluate(detailed=True)
                
                self.history['val_acc_overall'].append(results['top1_acc'])
                self.history['val_acc_top5'].append(results['top5_acc'])
                self.history['val_acc_with_pairs'].append(results['acc_with_pairs'])
                self.history['val_acc_without_pairs'].append(results['acc_without_pairs'])
                
                if results['top1_acc'] > self.best_accuracy:
                    self.best_accuracy = results['top1_acc']
                    self._save_checkpoint('improved_baseline1_best.pth', epoch, results)
                    print(f"New best: {results['top1_acc']:.2f}%\n")
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping check
                if patience > 0 and no_improve_count >= patience:
                    print(f"\nEarly stopping triggered after {no_improve_count * eval_frequency} epochs without improvement")
                    break
        
        phase_time = time.time() - phase_start_time
        print(f"\nPhase 2 completed in {phase_time/60:.1f} minutes ({phase_time/3600:.2f} hours)")
    
    def _save_checkpoint(self, filename, epoch, results):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'results': results,
            'history': self.history,
            'dataset_info': self.dataset_info,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        
        # Also save as JSON for easy reading
        results_file = filename.replace('.pth', '_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'results': results,
                'history': {k: [float(v) if isinstance(v, (int, float, np.number)) else v 
                               for v in vals] for k, vals in self.history.items()}
            }, f, indent=2)


def main():
    """Main training function for IMPROVED baseline-1"""
    
    print("="*60)
    print("OPTIMIZED IMPROVED BASELINE-1 TRAINING")
    print("Target: 78-82% accuracy in 5-6 hours")
    print("="*60)
    
    # OPTIMIZED Configuration
    CONFIG = {
        'data_dir': '.',
        
        # Model - Balanced for speed vs accuracy
        'model_name': 'convnext_small',    
        'img_size': 320,                    
        'batch_size': 20,                  
        'num_workers': 4,
        
        # Training - Shorter but effective
        'phase1_epochs': 15,                
        'phase2_epochs': 35,               
        'dropout': 0.5,
        'weight_decay': 0.1,
        
        # Class imbalance
        'use_class_balanced_loss': True,
        'use_focal_loss': False,
        
        # Augmentation
        'use_albumentations': True,
        'use_randaugment': False,
        'mixup_alpha': 0.4,
        
        # TTA Strategy - CRITICAL FOR SPEED
        'use_tta_during_training': False,   # DISABLED during training (5× faster)
        'use_tta_final': True,              # Only enable for final evaluation
        'tta_crops': 5,
        
        # Evaluation
        'eval_frequency': 10,                # Evaluate every N epochs
        'early_stopping_patience': 2,       # Stop if no improvement for 3 evaluations (15 epochs)
        
        # Learning rates
        'phase1_lr': 2e-3,                  
        'phase2_backbone_lr': 5e-6,         
        'phase2_head_lr': 1e-4,            
    }
    
    print("\nConfiguration:")
    print("="*60)
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   WARNING: No GPU detected! Training will be VERY slow.")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader, num_classes, dataset_info = create_improved_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        img_size=CONFIG['img_size'],
        use_albumentations=CONFIG['use_albumentations']
    )
    
    print(f"\nChallenge:")
    print(f"   {len(dataset_info['classes_without_pairs'])} species have NO field training!")
    print(f"   Expected performance gap: ~40% between species with/without pairs")
    
    # Create model
    print("\nCreating model...")
    model = create_improved_model(
        model_name=CONFIG['model_name'],
        num_classes=num_classes,
        pretrained=True,
        dropout=CONFIG['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = ImprovedBaseline1Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        dataset_info=dataset_info,
        config=CONFIG
    )
    
    # Training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    overall_start_time = time.time()
    
    # Phase 1
    trainer.train_phase1(
        epochs=CONFIG['phase1_epochs'],
        lr=CONFIG['phase1_lr'],
        warmup_epochs=5
    )
    
    # Phase 2
    trainer.train_phase2(
        epochs=CONFIG['phase2_epochs'],
        backbone_lr=CONFIG['phase2_backbone_lr'],
        head_lr=CONFIG['phase2_head_lr']
    )
    
    total_training_time = time.time() - overall_start_time
    
    # Final evaluation WITH TTA
    print("\n" + "="*60)
    print("FINAL EVALUATION (with TTA)")
    print("="*60)
    
    checkpoint = torch.load('improved_baseline1_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    trainer.model = model
    
    # Enable TTA for final evaluation
    final_results = trainer.evaluate(detailed=True, use_tta=CONFIG['use_tta_final'])
    
    # Summary
    total_epochs = CONFIG['phase1_epochs'] + CONFIG['phase2_epochs']
    avg_epoch_time = np.mean(trainer.history['epoch_times'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)")
    print(f"Average time per epoch: {avg_epoch_time/60:.1f} minutes")
    print(f"Total epochs: {total_epochs}")
    print(f"Best accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Model saved: improved_baseline1_best.pth")
    print(f"Results saved: improved_baseline1_best_results.json")
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Overall Top-1: {final_results['top1_acc']:.2f}%")
    print(f"Overall Top-5: {final_results['top5_acc']:.2f}%")
    print(f"Avg Per-Class: {final_results['avg_class_acc']:.2f}%")
    print(f"\nBy Species Type:")
    print(f"  WITH field pairs (60 species): {final_results['acc_with_pairs']:.2f}%")
    print(f"  WITHOUT field pairs (40 species): {final_results['acc_without_pairs']:.2f}%")
    print(f"  Performance gap: {final_results['acc_with_pairs'] - final_results['acc_without_pairs']:.2f}%")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Your result: {final_results['top1_acc']:.2f}%")
    print(f"Improvement: +{final_results['top1_acc'] - 65:.1f}%")
    
    if final_results['top1_acc'] >= 78:
        print("\nTARGET ACHIEVED! Excellent work!")
        if final_results['top1_acc'] >= 82:
            print("You exceeded expectations!")
    elif final_results['top1_acc'] >= 75:
        print("\nClose to target! Very good result given the dataset constraints.")
    else:
        print("\nSuggestions to improve:")
        print("   - Try convnext_base (slower but +2-3% accuracy)")
        print("   - Increase img_size to 384 (+1-2% accuracy)")
        print("   - Train phase2 for 60 epochs instead of 40")

if __name__ == "__main__":
    main()