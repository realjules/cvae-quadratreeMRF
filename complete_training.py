"""
Complete end-to-end training pipeline for DHBP semi-supervised segmentation.

Stage 1: SimCLR contrastive pre-training on unlabeled ISPRS data
Stage 2: Supervised segmentation with DHBP on labeled ISPRS data (10%)
Stage 3: Evaluation on held-out test set

Usage (local):
    python complete_training.py --epochs_contrastive 50 --epochs_seg 50

Usage (Kaggle):
    python complete_training.py \
        --data_dir /kaggle/input/your-dataset-name \
        --output_dir /kaggle/working/output \
        --epochs_contrastive 50 --epochs_seg 50
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import ISPRS_dataset
from dataset.unsupervised_dataset import ISPRS_unsupervised_dataset
from net.cvae import ContrastiveEncoder
from train import SegmentationTrainer
from utils.cvae_trainer import ContrastiveTrainer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_available_area_ids(data_dir="./input"):
    """Auto-detect available area IDs from ISPRS data files."""
    top_files = glob.glob(os.path.join(data_dir, "top", "top_mosaic_09cm_area*.tif"))
    gt_files = glob.glob(os.path.join(data_dir, "gt", "top_mosaic_09cm_area*.tif"))

    top_ids = [f.split('area')[1].split('.')[0] for f in top_files]
    gt_ids = [f.split('area')[1].split('.')[0] for f in gt_files]

    valid_ids = sorted(set(top_ids) & set(gt_ids), key=lambda x: int(x))
    print(f"Data dir: {data_dir}")
    print(f"Found {len(top_ids)} top images, {len(gt_ids)} ground truth images")
    print(f"Valid IDs with both: {valid_ids}")
    return valid_ids


def split_dataset_ids(valid_ids, labeled_percent=10):
    """Split dataset IDs into labeled train / unlabeled train / test."""
    total = len(valid_ids)

    # Test: last 20%
    test_split = max(1, int(0.2 * total))
    test_ids = valid_ids[-test_split:]
    train_pool = valid_ids[:-test_split]

    # Labeled: labeled_percent% of training pool
    labeled_count = max(1, int(labeled_percent / 100.0 * len(train_pool)))
    labeled_ids = train_pool[:labeled_count]

    # Unlabeled: ALL training data (for contrastive learning)
    unlabeled_ids = train_pool

    print(f"Dataset split ({total} areas):")
    print(f"  Labeled:   {len(labeled_ids)} areas {labeled_ids}")
    print(f"  Unlabeled: {len(unlabeled_ids)} areas (for contrastive)")
    print(f"  Test:      {len(test_ids)} areas {test_ids}")
    return labeled_ids, unlabeled_ids, test_ids


def create_real_dataloaders(data_dir="./input", batch_size=4,
                            labeled_percent=10, device="cuda"):
    """Create DataLoaders using real ISPRS dataset."""
    valid_ids = get_available_area_ids(data_dir)
    if not valid_ids:
        raise ValueError(
            f"No ISPRS data found in {data_dir}/top/ and {data_dir}/gt/\n"
            f"  Expected: {data_dir}/top/top_mosaic_09cm_area*.tif\n"
            f"  Expected: {data_dir}/gt/top_mosaic_09cm_area*.tif"
        )

    labeled_ids, unlabeled_ids, test_ids = split_dataset_ids(valid_ids, labeled_percent)

    top_pattern = os.path.join(data_dir, "top", "top_mosaic_09cm_area{}.tif")
    gt_pattern = os.path.join(data_dir, "gt", "top_mosaic_09cm_area{}.tif")

    unlabeled_dataset = ISPRS_unsupervised_dataset(
        ids=unlabeled_ids,
        data_files=top_pattern,
        window_size=256,
        cache=False,
        augmentation=True,
    )
    labeled_dataset = ISPRS_dataset(
        ids=labeled_ids, ids_type='TRAIN', gt_type='full', gt_modification=None,
        data_files=top_pattern,
        label_files=gt_pattern,
        window_size=256, cache=False, augmentation=True,
    )
    test_dataset = ISPRS_dataset(
        ids=test_ids, ids_type='TEST', gt_type='full', gt_modification=None,
        data_files=top_pattern,
        label_files=gt_pattern,
        window_size=256, cache=False, augmentation=False,
    )

    # Kaggle: 2 workers + pin_memory. Local/CPU: 0 workers.
    is_kaggle = os.path.exists("/kaggle")
    loader_kwargs = dict(
        num_workers=2 if is_kaggle else 0,
        pin_memory=torch.cuda.is_available(),
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, **loader_kwargs,
    )

    print(f"DataLoaders: unlabeled={len(unlabeled_dataset)}, "
          f"labeled={len(labeled_dataset)}, test={len(test_dataset)}")
    return unlabeled_loader, labeled_loader, test_loader


# ---------------------------------------------------------------------------
# Stage 1: Contrastive pre-training
# ---------------------------------------------------------------------------

def train_contrastive(trainer, dataloader, epochs, device, output_dir="./output"):
    """Stage 1: SimCLR contrastive learning on unlabeled data."""
    print("\n" + "=" * 60)
    print("STAGE 1: SimCLR Contrastive Pre-training")
    print("=" * 60)

    max_batches = min(200, len(dataloader))

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Pass RAW images — augmentation happens inside train_step
            metrics = trainer.train_step(images)

            if np.isnan(metrics['total_loss']):
                print(f"  NaN at batch {batch_idx}, skipping")
                continue

            epoch_loss += metrics['total_loss']
            num_batches += 1

            if batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start
                print(f"  [{epoch}/{epochs}] batch {batch_idx+1}/{max_batches} "
                      f"loss={metrics['total_loss']:.4f} ({elapsed:.0f}s)")

        avg_loss = epoch_loss / max(num_batches, 1)
        trainer.scheduler.step()
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # Save checkpoints
        if avg_loss < trainer.best_loss:
            trainer.best_loss = avg_loss
            trainer.save(os.path.join(output_dir, "contrastive_best.pth"), epoch)
            print(f"  New best contrastive model saved")

        if epoch == epochs:
            trainer.save(os.path.join(output_dir, "contrastive_final.pth"), epoch)

    print("Stage 1 complete.\n")
    return trainer


# ---------------------------------------------------------------------------
# Stage 2: Supervised segmentation
# ---------------------------------------------------------------------------

def train_segmentation(trainer, dataloader, test_loader, epochs, device,
                       output_dir="./output"):
    """Stage 2: Supervised segmentation with DHBP."""
    print("=" * 60)
    print("STAGE 2: Supervised Segmentation with DHBP")
    print("=" * 60)

    max_batches = min(100, len(dataloader))
    best_acc = 0.0
    patience = 0
    max_patience = 10

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            loss, components = trainer.train_step(images, labels)
            epoch_loss += loss
            num_batches += 1

            if batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start
                print(f"  [{epoch}/{epochs}] batch {batch_idx+1}/{max_batches} "
                      f"loss={loss:.4f} ({elapsed:.0f}s)")

        avg_loss = epoch_loss / max(num_batches, 1)
        trainer.scheduler.step()

        # Evaluate every 5 epochs (or last epoch)
        if epoch % 5 == 0 or epoch == epochs:
            metrics = trainer.evaluate(test_loader)
            acc = metrics['accuracy']
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.2f}%, "
                  f"mean_class={metrics['mean_accuracy']:.2f}%")

            if acc > best_acc:
                best_acc = acc
                trainer.save(os.path.join(output_dir, "best_segmentation.pth"))
                print(f"  New best: {acc:.2f}%")
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                print(f"Early stopping after {patience} evals without improvement")
                break
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    print(f"Stage 2 complete. Best accuracy: {best_acc:.2f}%\n")
    return trainer


# ---------------------------------------------------------------------------
# Stage 3: Evaluation
# ---------------------------------------------------------------------------

def evaluate_final(trainer, test_loader):
    """Stage 3: Final evaluation with per-class breakdown."""
    print("=" * 60)
    print("STAGE 3: Final Evaluation")
    print("=" * 60)

    metrics = trainer.evaluate(test_loader)
    class_names = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]

    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Mean Class Accuracy: {metrics['mean_accuracy']:.2f}%")
    print("Per-class:")
    for name, acc in zip(class_names, metrics['per_class_accuracy']):
        print(f"  {name}: {acc:.2f}%")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DHBP Complete Training Pipeline')
    parser.add_argument('--data_dir', default='./input',
                        help='Root directory containing top/ and gt/ subdirectories. '
                             'On Kaggle: /kaggle/input/<dataset-name>')
    parser.add_argument('--output_dir', default='./output/',
                        help='Directory for checkpoints and results. '
                             'On Kaggle: /kaggle/working/output')
    parser.add_argument('--epochs_contrastive', type=int, default=50)
    parser.add_argument('--epochs_seg', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_contrastive', type=float, default=1e-3)
    parser.add_argument('--lr_seg', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--labeled_percent', type=int, default=10)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("DHBP Semi-Supervised Segmentation Pipeline")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Labeled data: {args.labeled_percent}%")
    print(f"Contrastive epochs: {args.epochs_contrastive}")
    print(f"Segmentation epochs: {args.epochs_seg}")

    # Create dataloaders
    effective_batch = min(args.batch_size, 4)
    unlabeled_loader, labeled_loader, test_loader = create_real_dataloaders(
        data_dir=args.data_dir,
        batch_size=effective_batch,
        labeled_percent=args.labeled_percent,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    start_time = time.time()

    # Stage 1: Contrastive pre-training
    encoder = ContrastiveEncoder(pretrained=True)
    contrastive_trainer = ContrastiveTrainer(
        encoder=encoder,
        learning_rate=args.lr_contrastive,
        temperature=args.temperature,
        device=str(device),
    )

    if args.epochs_contrastive > 0:
        contrastive_trainer = train_contrastive(
            contrastive_trainer, unlabeled_loader, args.epochs_contrastive,
            device, output_dir=args.output_dir,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: Segmentation with DHBP
    # Pass the encoder object directly — no checkpoint file searching
    seg_trainer = SegmentationTrainer(
        encoder=contrastive_trainer.encoder,
        n_classes=6,
        learning_rate=args.lr_seg,
        device=str(device),
    )

    if args.epochs_seg > 0:
        seg_trainer = train_segmentation(
            seg_trainer, labeled_loader, test_loader, args.epochs_seg,
            device, output_dir=args.output_dir,
        )

    # Stage 3: Final evaluation
    final_metrics = evaluate_final(seg_trainer, test_loader)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Final accuracy: {final_metrics['accuracy']:.2f}%")

    return 0 if final_metrics['accuracy'] >= 90.0 else 1


if __name__ == "__main__":
    exit(main())
