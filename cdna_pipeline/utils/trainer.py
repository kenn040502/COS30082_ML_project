import torch
from utils.metrics import evaluate_model  # unified metric function
import os

class Trainer:
    def __init__(self, feature_extractor, classifier, discriminator, losses, optimizer, device):
        self.F = feature_extractor
        self.C = classifier
        self.D = discriminator
        self.losses = losses
        self.opt = optimizer
        self.device = device
        self.train_accuracies = []
        self.val_accuracies = []
        
    def save_checkpoint(self, epoch, path, stats=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'feature_extractor_state_dict': self.F.state_dict(),
            'classifier_state_dict': self.C.state_dict(),
            'discriminator_state_dict': self.D.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'stats': stats
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.F.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.C.load_state_dict(checkpoint['classifier_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Checkpoint loaded: {path}")
        return checkpoint['epoch'], checkpoint.get('stats', {})

    def train_step(self, train_batch, target_batch, lambda_=1.0):
        """
        Perform one training step (train domain + adaptation domain)
        """
        self.F.train(); self.C.train(); self.D.train()

        imgs_train = train_batch['image'].to(self.device)
        labels_train = train_batch['label'].to(self.device)
        imgs_target = target_batch['image'].to(self.device)

        # === Feature extraction ===
        f_train = self.F(imgs_train)
        f_target = self.F(imgs_target)
        if f_train.dim() == 4:
            f_train = f_train.squeeze()
            f_target = f_target.squeeze()

        # === Classification loss ===
        preds = self.C(f_train)
        loss_cls = self.losses["classification"](preds, labels_train)
        _, predicted = preds.max(1)
        correct = predicted.eq(labels_train).sum().item()
        train_acc = 100.0 * correct / labels_train.size(0)

        # === Domain loss ===
        f_all = torch.cat([f_train, f_target], dim=0)
        d_labels = torch.cat([
            torch.zeros(f_train.size(0), device=self.device),
            torch.ones(f_target.size(0), device=self.device)
        ], dim=0)

        d_preds = self.D(f_all, lambda_).squeeze()
        loss_dom = self.losses["domain"](d_preds, d_labels)

        # === Backprop ===
        total_loss = loss_cls + lambda_ * loss_dom
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return {
            "train_loss": loss_cls.item(),
            "domain_loss": loss_dom.item(),
            "train_acc": train_acc
        }

    def evaluate(self, val_loader, test_loader):
        """
        Evaluate model on validation (herbarium) and test (field) datasets
        """
        val_top1, val_top5 = evaluate_model(self.F, self.C, val_loader, self.device)
        test_top1, test_top5 = evaluate_model(self.F, self.C, test_loader, self.device)
        return {
            "val_top1": val_top1,
            "val_top5": val_top5,
            "test_top1": test_top1,
            "test_top5": test_top5
        }
