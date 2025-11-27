import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt

from baseline1_model import create_improved_model
from baseline1_dataloader import create_improved_dataloaders


class MixupAugmentation:
    """Mixup data augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class OptimizedTrainer:
    """
    OPTIMIZED trainer - balances performance and stability.
    """
    def __init__(self, model, train_loader, test_loader, device, dataset_info, 
                 use_mixup=True, mixup_alpha=0.25, label_smoothing=0.05):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dataset_info = dataset_info
        
        # Mixup
        self.use_mixup = use_mixup
        self.mixup = MixupAugmentation(alpha=mixup_alpha) if use_mixup else None
        
        # Loss - Using standard CrossEntropyLoss (simpler and more stable)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Tracking
        self.best_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc_overall': [],
            'val_acc_top5': [],
            'val_acc_with_pairs': [],
            'val_acc_without_pairs': [],
            'learning_rates': [],
            'epoch': []
        }
        
        # Mixed precision
        self.scaler = GradScaler()
        
        print(f"OPTIMIZED Trainer initialized (Sweet Spot)")
        print(f"   Mixup: {use_mixup} (alpha={mixup_alpha})")
        print(f"   Label smoothing: {label_smoothing}")
        print(f"   Loss: CrossEntropyLoss (standard, stable)")
    
    def train_epoch(self, optimizer, epoch, use_mixup_this_epoch=True):
        """Train one epoch with balanced approach"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixup
            if self.use_mixup and use_mixup_this_epoch and np.random.random() > 0.3:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
                
                with autocast():
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\n⚠️  NaN loss at batch {batch_idx}! Skipping.")
                continue
            
            # Optimization
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping - using 0.7 for sweet spot balance
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.7)
            
            # Monitor gradients
            grad_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            if grad_norm > 10.0:
                print(f"\n⚠️  Large gradient: {grad_norm:.2f}. Clipped to 0.5")
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'grad': f'{grad_norm:.2f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, detailed=False):
        """Evaluate model with proper species pair tracking"""
        self.model.eval()
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        running_loss = 0.0
        
        # With/without pairs tracking
        classes_with = self.dataset_info['classes_with_pairs']
        classes_without = self.dataset_info['classes_without_pairs']
        
        correct_with = 0
        total_with = 0
        correct_without = 0
        total_without = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # Top-1
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_top1 += predicted.eq(labels).sum().item()
                
                # Top-5
                _, top5_pred = outputs.topk(5, dim=1)
                for i in range(labels.size(0)):
                    if labels[i] in top5_pred[i]:
                        correct_top5 += 1
                    
                    # Track with/without pairs
                    label = labels[i].item()
                    pred = predicted[i].item()
                    
                    if label in classes_with:
                        total_with += 1
                        if pred == label:
                            correct_with += 1
                    elif label in classes_without:
                        total_without += 1
                        if pred == label:
                            correct_without += 1
        
        val_loss = running_loss / len(self.test_loader)
        top1_acc = 100. * correct_top1 / total if total > 0 else 0
        top5_acc = 100. * correct_top5 / total if total > 0 else 0
        acc_with = 100. * correct_with / total_with if total_with > 0 else 0
        acc_without = 100. * correct_without / total_without if total_without > 0 else 0
        
        results = {
            'val_loss': val_loss,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'acc_with_pairs': acc_with,
            'acc_without_pairs': acc_without,
        }
        
        if detailed:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Top-1 Accuracy: {top1_acc:.2f}%")
            print(f"Top-5 Accuracy: {top5_acc:.2f}%")
            print(f"\nPerformance by Species Type:")
            print(f"  With pairs ({total_with} samples): {acc_with:.2f}%")
            print(f"  Without pairs ({total_without} samples): {acc_without:.2f}%")
            print(f"  Gap: {acc_with - acc_without:.2f}%")
            print("="*60 + "\n")
        
        return results
    
    def train_phase1(self, epochs=13, initial_lr=1e-3, warmup_epochs=3):
        """
        Phase 1: BALANCED learning rate (between 5e-4 and 2e-3)
        """
        print("\n" + "="*60)
        print("PHASE 1: Training Classifier Head")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Initial LR: {initial_lr}")
        print(f"Warmup epochs: {warmup_epochs}")
        print()
        
        self.model.freeze_backbone()
        
        # BALANCED learning rate
        optimizer = optim.AdamW(
            self.model.get_head_params(), 
            lr=initial_lr,  # Between conservative and aggressive
            weight_decay=0.01
        )
        
        # Warmup + cosine
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                optimizer, epoch + 1, use_mixup_this_epoch=(epoch >= warmup_epochs)
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate
            results = self.evaluate(detailed=False)
            
            # Log
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {results['val_loss']:.4f}, Acc: {results['top1_acc']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(results['val_loss'])
            self.history['val_acc_overall'].append(results['top1_acc'])
            self.history['val_acc_top5'].append(results['top5_acc'])
            self.history['val_acc_with_pairs'].append(results['acc_with_pairs'])
            self.history['val_acc_without_pairs'].append(results['acc_without_pairs'])
            self.history['learning_rates'].append(current_lr)
            
            # Save best
            if results['top1_acc'] > self.best_accuracy:
                self.best_accuracy = results['top1_acc']
                self._save_checkpoint('baseline1_optimized_best.pth', epoch, results)
                print(f"  ✅ New best: {results['top1_acc']:.2f}%")
            print()
    
    def train_phase2(self, epochs=27, backbone_lr=6e-6, head_lr=1e-4, max_gap=18):
        """
        Phase 2: BALANCED fine-tuning with gap control
        Stops if gap exceeds threshold for too long
        """
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning Entire Model with Gap Control")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Max allowed gap: {max_gap}%")
        print(f"Backbone LR: {backbone_lr}")
        print(f"Head LR: {head_lr}")
        print()
        
        self.model.unfreeze_all()
        
        # BALANCED learning rates
        optimizer = optim.AdamW([
            {'params': self.model.get_backbone_params(), 'lr': backbone_lr, 'weight_decay': 0.15},
            {'params': self.model.get_head_params(), 'lr': head_lr, 'weight_decay': 0.08}
        ])
        
        # Warmup for phase 2
        warmup_epochs = 3
        
        def lr_lambda_with_warmup(epoch):
            if epoch < warmup_epochs:
                return 0.2 + 0.8 * (epoch / warmup_epochs)  # Warm up from 20% to 100%
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_with_warmup)
        
        phase1_epochs = len([e for e in self.history['epoch'] if e <= 13])
        consecutive_high_gap = 0  # Track consecutive epochs with high gap
        
        for epoch in range(epochs):
            # Enable mixup after warmup
            use_mixup = epoch >= 4
            
            # Train
            train_loss, train_acc = self.train_epoch(
                optimizer, phase1_epochs + epoch + 1, use_mixup_this_epoch=use_mixup
            )
            
            scheduler.step()
            
            # Evaluate
            results = self.evaluate(detailed=False)
            
            # Calculate gap
            current_gap = train_acc - results['top1_acc']
            
            # Log
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_head = optimizer.param_groups[1]['lr']
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {results['val_loss']:.4f}, Acc: {results['top1_acc']:.2f}%")
            print(f"  Gap: {current_gap:.2f}%", end="")
            
            # Gap warning
            if current_gap > max_gap:
                consecutive_high_gap += 1
                print(f" ⚠️  HIGH! ({consecutive_high_gap} consecutive)")
            else:
                consecutive_high_gap = 0
                print(f" ✅")
            
            print(f"  LR: Backbone={current_lr_backbone:.7f}, Head={current_lr_head:.7f}")
            
            # Save history
            self.history['epoch'].append(phase1_epochs + epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(results['val_loss'])
            self.history['val_acc_overall'].append(results['top1_acc'])
            self.history['val_acc_top5'].append(results['top5_acc'])
            self.history['val_acc_with_pairs'].append(results['acc_with_pairs'])
            self.history['val_acc_without_pairs'].append(results['acc_without_pairs'])
            self.history['learning_rates'].append(current_lr_head)
            
            # Save best
            if results['top1_acc'] > self.best_accuracy:
                self.best_accuracy = results['top1_acc']
                self._save_checkpoint('baseline1_optimized_best.pth', phase1_epochs + epoch, results)
                print(f"  ✅ New best: {results['top1_acc']:.2f}%")
            
            # Early stopping if gap too high for too long
            if consecutive_high_gap >= 5:
                print(f"\n⚠️  Stopping early: Gap exceeded {max_gap}% for {consecutive_high_gap} consecutive epochs")
                print(f"Best model saved with {self.best_accuracy:.2f}% validation accuracy")
                break
            
            print()
    
    def plot_training_curves(self, save_path='training_curves_optimized.png'):
        """Generate training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.history['epoch']
        
        # Plot 1: Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax.plot(epochs, self.history['val_acc_overall'], 'r-', label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Best accuracy
        best_idx = np.argmax(self.history['val_acc_overall'])
        best_epoch = epochs[best_idx]
        best_acc = self.history['val_acc_overall'][best_idx]
        ax.plot(best_epoch, best_acc, 'r*', markersize=15)
        ax.annotate(f'Best: {best_acc:.2f}%', 
                   xy=(best_epoch, best_acc), 
                   xytext=(best_epoch, best_acc + 5),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Plot 3: Top-1 vs Top-5
        ax = axes[1, 0]
        ax.plot(epochs, self.history['val_acc_overall'], 'g-', label='Top-1 Accuracy', linewidth=2)
        ax.plot(epochs, self.history['val_acc_top5'], 'm-', label='Top-5 Accuracy', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Top-1 vs Top-5 Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: With/Without Pairs
        ax = axes[1, 1]
        if any(x > 0 for x in self.history['val_acc_with_pairs']):
            ax.plot(epochs, self.history['val_acc_with_pairs'], 'c-', label='With Pairs', linewidth=2)
            ax.plot(epochs, self.history['val_acc_without_pairs'], 'orange', label='Without Pairs', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Species With vs Without Field Pairs', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Gap annotation
            if len(self.history['val_acc_with_pairs']) > 0:
                final_with = self.history['val_acc_with_pairs'][-1]
                final_without = self.history['val_acc_without_pairs'][-1]
                gap = final_with - final_without
                ax.text(0.95, 0.05, f'Gap: {gap:.2f}%', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Pair data not available',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Training curves saved to: {save_path}")
        plt.close()
    
    def _save_checkpoint(self, filename, epoch, results):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'results': results,
            'history': self.history,
            'dataset_info': self.dataset_info
        }
        torch.save(checkpoint, filename)
        
        # Save JSON
        results_file = filename.replace('.pth', '_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'results': results,
                'history': {k: [float(v) if isinstance(v, (int, float, np.number)) else v 
                               for v in vals] for k, vals in self.history.items()}
            }, f, indent=2)


def main():
    """Main training"""
    
    print("="*60)
    print("BASELINE-1 TRAINING")
    print("="*60)
    
    CONFIG = {
        'data_dir': '.',
        'batch_size': 32,
        'img_size': 320,
        'num_workers': 4,
        'model_name': 'convnext_small',
        'dropout': 0.2,
        'use_mixup': True,
        'mixup_alpha': 0.25,       
        'label_smoothing': 0.05,
        'use_albumentations': True,
        'phase1_epochs': 13,         
        'phase2_epochs': 27,         
        'phase1_lr': 1e-3,          
        'phase2_backbone_lr': 6e-6, 
        'phase2_head_lr': 1e-4,     
        'max_gap': 16,              
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader, num_classes, dataset_info = create_improved_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        img_size=CONFIG['img_size'],
        use_albumentations=CONFIG['use_albumentations']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_improved_model(
        model_name=CONFIG['model_name'],
        num_classes=num_classes,
        pretrained=True,
        dropout=CONFIG['dropout']
    )
    
    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        dataset_info=dataset_info,
        use_mixup=CONFIG['use_mixup'],
        mixup_alpha=CONFIG['mixup_alpha'],
        label_smoothing=CONFIG['label_smoothing']
    )
    
    # Training
    print("\nStarting OPTIMIZED training...")
    start_time = time.time()
    
    # Phase 1
    trainer.train_phase1(
        epochs=CONFIG['phase1_epochs'],
        initial_lr=CONFIG['phase1_lr'],
        warmup_epochs=3
    )
    
    # Phase 2 with gap control
    trainer.train_phase2(
        epochs=CONFIG['phase2_epochs'],
        backbone_lr=CONFIG['phase2_backbone_lr'],
        head_lr=CONFIG['phase2_head_lr'],
        max_gap=CONFIG['max_gap']
    )
    
    # Generate curves
    trainer.plot_training_curves('training_curves_optimized.png')
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    checkpoint = torch.load('baseline1_optimized_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    trainer.model = model
    
    final_results = trainer.evaluate(detailed=True)
    
    # Summary
    total_time = time.time() - start_time
    total_epochs = len(trainer.history['epoch'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Time per epoch: {(total_time/total_epochs)/60:.1f} minutes")
    print(f"Total epochs trained: {total_epochs}")
    print(f"Best validation accuracy: {trainer.best_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Overall Top-1: {final_results['top1_acc']:.2f}%")
    print(f"Overall Top-5: {final_results['top5_acc']:.2f}%")
    print(f"Species WITH pairs: {final_results['acc_with_pairs']:.2f}%")
    print(f"Species WITHOUT pairs: {final_results['acc_without_pairs']:.2f}%")
    print(f"Performance gap: {final_results['acc_with_pairs'] - final_results['acc_without_pairs']:.2f}%")

if __name__ == "__main__":
    main()