import torch
import os, math, numpy as np
from utils.metrics import evaluate_model

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
                num_batches=None, num_epochs=None, max_lambda=0.1):

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
        preds = self.C(f_train)
        loss_cls = self.losses["classification"](preds, labels_train)
        _, predicted = preds.max(1)
        correct = predicted.eq(labels_train).sum().item()
        train_acc = 100.0 * correct / max(1, labels_train.size(0))

        # === Lambda scheduling ===
        if lambda_ is None and epoch is not None and batch_idx is not None:
            p = (epoch * num_batches + batch_idx) / float(num_epochs * num_batches)
            lambda_val = self.lambda_schedule(p, max_lambda)
        else:
            lambda_val = lambda_ if lambda_ is not None else max_lambda

        # === Domain alignment (CDAN style for ViT / DINOv2) ===
        if "vit" in self.backbone_type or "dino" in self.backbone_type:
            probs_train = torch.softmax(preds, dim=1)
            probs_target = torch.softmax(self.C(f_target), dim=1).detach()

            feat_cond_train = torch.bmm(probs_train.unsqueeze(2), f_train.unsqueeze(1)).view(f_train.size(0), -1)
            feat_cond_target = torch.bmm(probs_target.unsqueeze(2), f_target.unsqueeze(1)).view(f_target.size(0), -1)
            f_all = torch.cat([feat_cond_train, feat_cond_target], dim=0)
        else:
            f_all = torch.cat([f_train, f_target], dim=0)

        # === Domain labels & prediction ===
        d_labels = torch.cat([
            torch.zeros(f_train.size(0), device=self.device),
            torch.ones(f_target.size(0), device=self.device)
        ], dim=0)

        d_preds = self.D(f_all, lambda_val).squeeze()
        if d_preds.dim() == 0:
            d_preds = d_preds.unsqueeze(0)

        loss_dom = self.losses["domain"](d_preds, d_labels)

        # === 🔹 Entropy Minimization (Target domain) ===
        tgt_logits = self.C(f_target)  # forward target through classifier
        p = torch.softmax(tgt_logits, dim=1)
        entropy = -torch.sum(p * torch.log(p + 1e-5), dim=1)
        entropy_loss = torch.mean(entropy)
        entropy_weight = 0.01  # tune 0.005–0.05

        # === Logging metrics ===
        with torch.no_grad():
            d_prob = torch.sigmoid(d_preds)
            d_pred_labels = (d_prob > 0.5).float()
            disc_acc = (d_pred_labels == d_labels).float().mean().item() * 100.0

        # === Backprop ===
        total_loss = loss_cls + lambda_val * loss_dom + entropy_weight * entropy_loss
        self.opt.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.F.parameters()) + list(self.C.parameters()) + list(self.D.parameters()), 5.0
        )

        self.opt.step()

        return {
            "train_loss": loss_cls.item(),
            "domain_loss": loss_dom.item(),
            "entropy_loss": entropy_loss.item(),
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
