import torch
import os, math, numpy as np
from utils.metrics import evaluate_model


class Trainer:
    def __init__(self, feature_extractor, classifier, discriminator,
                 losses, optimizer, device, backbone_type="resnet"):

        self.F = feature_extractor
        self.C = classifier
        self.D = discriminator
        self.losses = losses
        self.opt = optimizer
        self.device = device
        self.backbone_type = backbone_type.lower()

    # ================================================
    # Save / Load
    # ================================================
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

    # ================================================
    # Classic DANN λ-schedule
    # ================================================
    @staticmethod
    def lambda_schedule(p, max_lambda):
        return max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

    # ================================================
    # DANN Training Step
    # ================================================
    def train_step(self, source_batch, target_batch,
                   epoch=None, batch_idx=None,
                   num_batches=None, num_epochs=None,
                   max_lambda=1.0):

        self.F.train(); self.C.train(); self.D.train()

        # -------------------------
        # Extract source images
        # -------------------------
        imgs_src = source_batch["image"].to(self.device)
        labels_src = source_batch["label"].to(self.device)

        # Target images
        imgs_tgt = target_batch["image"].to(self.device)

        # -------------------------
        # Forward (feature)
        # -------------------------
        f_src = self.F(imgs_src)
        f_tgt = self.F(imgs_tgt)

        if f_src.dim() > 2: f_src = f_src.mean(dim=1)
        if f_tgt.dim() > 2: f_tgt = f_tgt.mean(dim=1)

        # -------------------------
        # Source Classification
        # -------------------------
        preds_src = self.C(f_src)
        loss_cls = self.losses["classification"](preds_src, labels_src)

        _, predicted = preds_src.max(1)
        src_acc = 100.0 * predicted.eq(labels_src).sum().item() / labels_src.size(0)

        # -------------------------
        # Lambda scheduling
        # -------------------------
        if epoch is not None and batch_idx is not None:
            p = (epoch * num_batches + batch_idx) / float(num_epochs * num_batches)
            lambda_val = self.lambda_schedule(p, max_lambda)
        else:
            lambda_val = max_lambda

        # -------------------------
        # DANN Domain Alignment
        # -------------------------
        f_all = torch.cat([f_src, f_tgt], dim=0)
        domain_labels = torch.cat([
            torch.zeros(f_src.size(0), device=self.device),   # source = 0
            torch.ones(f_tgt.size(0), device=self.device)     # target = 1
        ], dim=0)

        d_out = self.D(f_all, lambda_=lambda_val).squeeze()
        loss_dom = self.losses["domain"](d_out, domain_labels)

        # -------------------------
        # Total update
        # -------------------------
        total_loss = loss_cls + lambda_val * loss_dom

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        # -------------------------
        # Discriminator accuracy
        # -------------------------
        with torch.no_grad():
            pred_d = (torch.sigmoid(d_out) > 0.5).float()
            disc_acc = (pred_d == domain_labels).float().mean().item() * 100.0

        return {
            "train_loss": loss_cls.item(),
            "domain_loss": loss_dom.item(),
            "train_acc": src_acc,
            "disc_acc": disc_acc,
            "lambda_val": float(lambda_val)
        }

    # ================================================
    # Evaluation
    # ================================================
    def evaluate(self, val_loader, test_loader):
        val_top1, val_top5 = evaluate_model(self.F, self.C, val_loader, self.device)
        test_top1, test_top5 = evaluate_model(self.F, self.C, test_loader, self.device)
        return {
            "val_top1": val_top1,
            "val_top5": val_top5,
            "test_top1": test_top1,
            "test_top5": test_top5
        }
