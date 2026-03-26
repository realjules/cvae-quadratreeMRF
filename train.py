"""
Segmentation trainer using DHBP (Differentiable Hierarchical Belief Propagation).

Pipeline:
    ContrastiveEncoder (fine-tuned end-to-end) → DHBP → SegmentationLoss

The encoder is NOT frozen — gradients flow through the entire model.
Differential learning rates: 0.1x for pretrained encoder, 1x for DHBP.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import SegmentationLoss


class SegmentationTrainer:
    """Segmentation trainer with encoder + DHBP.

    Key design decisions:
        - Encoder is fine-tuned end-to-end (NOT frozen)
        - Differential LR: encoder at 0.1x, DHBP at 1x
        - Consistent save/load: {encoder_state_dict, dhbp_state_dict, optimizer_state_dict}
        - No duplicate model definitions — uses net/dhbp.py directly
    """

    def __init__(
        self,
        encoder: ContrastiveEncoder,
        n_classes: int = 6,
        learning_rate: float = 1e-3,
        device: str = 'cuda',
        use_bp: bool = True,
        simple_unary: bool = False,
        diagonal_pairwise: bool = False,
        n_levels: int = 3,
    ):
        self.device = device
        self.n_classes = n_classes
        self.use_bp = use_bp
        self.encoder = encoder.to(device)
        self.dhbp = DHBPModule(
            n_classes=n_classes, simple_unary=simple_unary,
            diagonal_pairwise=diagonal_pairwise, n_levels=n_levels,
        ).to(device)
        self.criterion = SegmentationLoss(n_classes=n_classes).to(device)

        if not use_bp:
            # No-BP ablation: encoder → unary head only (no message passing)
            # Reuse unary_1 from DHBP so it's the same architecture
            print("NOTE: BP disabled — using unary head only (ablation mode)")

        # End-to-end optimizer with differential learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': self.encoder.encoder.parameters(), 'lr': learning_rate * 0.1},
            {'params': self.dhbp.parameters(), 'lr': learning_rate},
        ], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,
        )

        self.best_loss = float('inf')

    def train_step(
        self, images: torch.Tensor, labels: torch.Tensor,
    ) -> tuple[float, dict]:
        """Single training step.

        Args:
            images: [B, 3, 256, 256]
            labels: [B, 256, 256] integer class labels

        Returns:
            (loss_value, loss_components_dict)
        """
        self.encoder.train()
        self.dhbp.train()

        images = images.to(self.device)
        labels = labels.to(self.device).long()

        # Extract multi-scale features (encoder IS fine-tuned)
        p1, p2, p3 = self.encoder.encode(images)

        if self.use_bp:
            # Full DHBP: unary + pairwise + belief propagation
            logits = self.dhbp(p1, p2, p3)  # [B, n_classes, 128, 128]
        else:
            # Ablation: unary head only, no message passing
            logits = F.log_softmax(self.dhbp.unary_1(p1), dim=1)  # [B, K, 128, 128]

        # Upsample logits to match label spatial dims
        if logits.shape[2:] != labels.shape[1:]:
            logits = F.interpolate(
                logits, size=labels.shape[1:],
                mode='bilinear', align_corners=False,
            )

        loss, components = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.dhbp.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return loss.item(), components

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Args:
            images: [B, 3, 256, 256]

        Returns:
            [B, H, W] predicted class indices (at input resolution)
        """
        self.encoder.eval()
        self.dhbp.eval()

        images = images.to(self.device)
        p1, p2, p3 = self.encoder.encode(images)

        if self.use_bp:
            logits = self.dhbp(p1, p2, p3)
        else:
            logits = F.log_softmax(self.dhbp.unary_1(p1), dim=1)

        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=images.shape[2:],
            mode='bilinear', align_corners=False,
        )
        return logits.argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate on a dataloader.

        Returns:
            dict with 'accuracy', 'per_class_accuracy', 'mean_accuracy'
        """
        self.encoder.eval()
        self.dhbp.eval()

        total_correct = 0
        total_pixels = 0
        class_correct = torch.zeros(self.n_classes, device=self.device)
        class_total = torch.zeros(self.n_classes, device=self.device)

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device).long()

            preds = self.predict(images)

            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()

            for c in range(self.n_classes):
                mask = labels == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == c).sum()
                    class_total[c] += mask.sum()

        overall_acc = total_correct / total_pixels * 100
        per_class = []
        for c in range(self.n_classes):
            if class_total[c] > 0:
                per_class.append((class_correct[c] / class_total[c] * 100).item())
            else:
                per_class.append(0.0)

        return {
            'accuracy': overall_acc,
            'per_class_accuracy': per_class,
            'mean_accuracy': np.mean(per_class),
        }

    def save(self, path: str):
        """Save full model state."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'dhbp_state_dict': self.dhbp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load full model state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.dhbp.load_state_dict(ckpt['dhbp_state_dict'])


def main():
    """Standalone segmentation training entry point."""
    parser = argparse.ArgumentParser(description='DHBP Segmentation Training')
    parser.add_argument('--encoder_ckpt', required=True,
                        help='Path to contrastive-pretrained encoder checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--labeled_percent', type=int, default=10)
    parser.add_argument('--output_dir', default='./output/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained encoder
    print(f"Loading encoder from {args.encoder_ckpt}")
    encoder = ContrastiveEncoder(pretrained=True)
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    print("Encoder loaded.")

    # Create trainer
    trainer = SegmentationTrainer(
        encoder=encoder, n_classes=6, learning_rate=args.lr, device=device,
    )

    # Create dataloaders
    from complete_training import create_real_dataloaders
    _, labeled_loader, test_loader = create_real_dataloaders(
        batch_size=args.batch_size,
        labeled_percent=args.labeled_percent,
        device=device,
    )

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in tqdm(labeled_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            loss, components = trainer.train_step(images, labels)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluate
        metrics = trainer.evaluate(test_loader)
        acc = metrics['accuracy']
        trainer.scheduler.step(acc)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.2f}%, "
              f"mean_class_acc={metrics['mean_accuracy']:.2f}%")

        # Save best
        if acc > best_acc:
            best_acc = acc
            trainer.save(os.path.join(args.output_dir, 'best_segmentation.pth'))
            print(f"  New best: {acc:.2f}%")

    # Final per-class results
    class_names = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]
    final = trainer.evaluate(test_loader)
    print(f"\nFinal accuracy: {final['accuracy']:.2f}%")
    for name, acc in zip(class_names, final['per_class_accuracy']):
        print(f"  {name}: {acc:.2f}%")


if __name__ == "__main__":
    main()
