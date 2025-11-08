import torch
import os, math, numpy as np
from utils.metrics import evaluate_model
from models.cdan_module import conditional_adversarial_features

class Trainer:
    def __init__(self, feature_extractor, classifier, discriminator, losses, optimizer, device, backbone_type="resnet"):
        self.F = feature_extractor
        self.C = classifier
        self.D = discriminator
        self.losses = losses
        self.opt = optimizer
        self.device = device
        self.backbone_type = backbone_type.lower()

        self.train_accuracies = []
        self.val_accuracies = []

    # ---------- Save / Load ----------
    def save_checkpoint(self, epoch, path, stats=None):
        checkpoint = {
            "epoch": epoch,
            "feature_extractor_state_dict": self.F.state_dict(),
            "classifier_state_dict": self.C.state_dict(),
            "discriminator_state_dict": self.D.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "stats": stats
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.F.load_state_dict(checkpoint["feature_extractor_state_dict"])
        self.C.load_state_dict(checkpoint["classifier_state_dict"])
        self.D.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✅ Checkpoint loaded: {path}")
        return checkpoint["epoch"], checkpoint.get("stats", {})

    # ---------- Lambda schedule ----------
    @staticmethod
    def lambda_schedule(p, max_lambda):
        """Classic DANN lambda schedule (smoothly increases from 0 → max_lambda)."""
        return max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

    # ---------- Training step ----------
    def train_step(self, train_batch, target_batch,
                lambda_=None, epoch=None, batch_idx=None,
                num_batches=None, num_epochs=None, max_lambda=0.1,
                gamma_entropy=0.1): # <--- Added gamma_entropy parameter

        self.F.train(); self.C.train(); self.D.train()

        imgs_train = train_batch["image"].to(self.device)
        labels_train = train_batch["label"].to(self.device)
        imgs_target = target_batch["image"].to(self.device)

        # === Feature extraction ===
        f_train = self.F(imgs_train)
        f_target = self.F(imgs_target)

        # flatten / pool token dims if ViT
        if f_train.dim() > 2:
            f_train = f_train.mean(dim=1)
        if f_target.dim() > 2:
            f_target = f_target.mean(dim=1)

        # === Classification ===
        preds_train = self.C(f_train)
        loss_cls = self.losses["classification"](preds_train, labels_train)
        _, predicted = preds_train.max(1)
        correct = predicted.eq(labels_train).sum().item()
        train_acc = 100.0 * correct / max(1, labels_train.size(0))

        # === Lambda scheduling ===
        if lambda_ is None and epoch is not None and batch_idx is not None:
            p = (epoch * num_batches + batch_idx) / float(num_epochs * num_batches)
            lambda_val = self.lambda_schedule(p, max_lambda)
        else:
            lambda_val = lambda_ if lambda_ is not None else max_lambda

        # === CDAN Domain alignment ===
        preds_target = self.C(f_target)
        soft_train = torch.softmax(preds_train, dim=1)
        soft_target = torch.softmax(preds_target, dim=1)

        f_all = torch.cat([f_train, f_target], dim=0)
        y_all = torch.cat([soft_train, soft_target], dim=0)
        domain_labels = torch.cat([
            torch.zeros(f_train.size(0), device=self.device),
            torch.ones(f_target.size(0), device=self.device)
        ], dim=0)

        # 🔹 Conditional feature combination (outer product)
        feat_cond = conditional_adversarial_features(f_all, y_all, mode="outer")

        # 🔹 Entropy weighting for target samples
        entropy = -torch.sum(soft_target * torch.log(soft_target + 1e-5), dim=1)
        entropy = 1.0 + torch.exp(-entropy)
        weights = entropy / torch.sum(entropy)
        weights = (weights / torch.max(weights)).detach()  # normalize for stability

        # Discriminator forward
        d_out = self.D(feat_cond, lambda_=lambda_val)
        d_out = d_out.squeeze()

        # Split domain preds for entropy weighting
        d_src = d_out[:f_train.size(0)]
        d_tgt = d_out[f_train.size(0):]

        # === Domain loss with entropy conditioning ===
        loss_src = self.losses["domain"](d_src, torch.zeros_like(d_src))
        loss_tgt = (self.losses["domain"](d_tgt, torch.ones_like(d_tgt)) * weights).mean()
        loss_dom = (loss_src + loss_tgt) / 2.0
        
        # === Entropy loss (CDAN+E) ===
        # The goal is to minimize the entropy of target predictions (make them more confident)
        loss_ent = -torch.mean(torch.sum(soft_target * torch.log(soft_target + 1e-5), dim=1)) # <--- Added Entropy Loss

        # === Total loss ===
        total_loss = loss_cls + lambda_val * loss_dom + gamma_entropy * loss_ent # <--- Added Entropy Loss term

        # === Logging metrics ===
        with torch.no_grad():
            d_pred = (torch.sigmoid(d_out) > 0.5).float()
            disc_acc = (d_pred == domain_labels).float().mean().item() * 100.0

        # === Backprop ===
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.F.parameters()) + list(self.C.parameters()) + list(self.D.parameters()), 5.0
        )
        self.opt.step()

        return {
            "train_loss": loss_cls.item(),
            "domain_loss": loss_dom.item(),
            "entropy_loss": loss_ent.item(), # <--- Added entropy_loss to return dict
            "train_acc": train_acc,
            "disc_acc": disc_acc,
            "lambda_val": float(lambda_val)
        }

    # ---------- Evaluation ----------
    def evaluate(self, val_loader, test_loader):
        val_top1, val_top5 = evaluate_model(self.F, self.C, val_loader, self.device)
        test_top1, test_top5 = evaluate_model(self.F, self.C, test_loader, self.device)
        return {
            "val_top1": val_top1,
            "val_top5": val_top5,
            "test_top1": test_top1,
            "test_top5": test_top5
        }
