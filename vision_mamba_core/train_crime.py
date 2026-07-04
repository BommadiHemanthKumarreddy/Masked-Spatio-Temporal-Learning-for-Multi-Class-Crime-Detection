# --------------------------------------------------------
# Training Script — Vision Mamba Crime Detector v3
# Pipeline: 2D ViT (per-frame) -> TemporalMamba x2 -> Attention -> Classifier
#
# Changes in v3:
#   - 85% train / 15% test split (no validation — more training data)
#   - MixUp augmentation for small-dataset regularisation
#   - Cosine LR warmup 3 epochs then cosine decay to 1e-6
#   - Higher head LR (5e-4), encoder LR 10x lower
#   - Unfreeze last 6 encoder blocks (was 4)
#   - Gradient clipping 1.0
# --------------------------------------------------------

import argparse
import csv
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from crime_model import CrimeDetectionModel, CRIME_CLASSES, NUM_CLASSES
from crime_dataset import CrimeVideoDataset

try:
    from sklearn.metrics import f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not found. F1-score will not be computed.")


# ────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser('Crime Detection Training v3')

    p.add_argument('--data_dir',    type=str,   required=True)
    p.add_argument('--encoder',     type=str,   default='vit_small_patch16_224')
    p.add_argument('--ckpt',        type=str,   default='',
                   help='VideoMAE checkpoint (leave empty for ImageNet pretrained)')
    p.add_argument('--unfreeze_blocks', type=int, default=6)
    p.add_argument('--num_frames',  type=int,   default=16)
    p.add_argument('--mamba_layers',type=int,   default=2)

    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=5e-4,
                   help='Peak LR for head layers')
    p.add_argument('--encoder_lr_scale', type=float, default=0.1)
    p.add_argument('--weight_decay',type=float, default=0.05)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--warmup_epochs',type=int,  default=3)
    p.add_argument('--clip_grad',   type=float, default=1.0)
    p.add_argument('--mixup_alpha', type=float, default=0.4,
                   help='MixUp alpha (0 = disabled)')

    p.add_argument('--mamba_d_state', type=int, default=16)
    p.add_argument('--mamba_d_conv',  type=int, default=4)
    p.add_argument('--mamba_expand',  type=int, default=2)

    p.add_argument('--output_dir',  type=str,   default='checkpoints')
    p.add_argument('--log_csv',     type=str,   default='training_log.csv')
    p.add_argument('--resume',      type=str,   default='')
    p.add_argument('--test_split',  type=float, default=0.15,
                   help='Fraction of data reserved for final test (default 0.15)')
    return p.parse_args()


# ────────────────────────────────────────────────────────────
# MixUp
# ────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha, device):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)          # always mix at least 50/50
    B   = x.size(0)
    idx = torch.randperm(B, device=device)
    mixed_x  = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ────────────────────────────────────────────────────────────
# LR schedule: linear warmup -> cosine decay
# ────────────────────────────────────────────────────────────

def get_lr_scale(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    import math
    return 1e-2 + 0.5 * (1 - 1e-2) * (1 + math.cos(math.pi * progress))


def apply_lr(optimizer, base_lr, epoch, warmup_epochs, total_epochs):
    scale = get_lr_scale(epoch, warmup_epochs, total_epochs)
    for group in optimizer.param_groups:
        group['lr'] = base_lr * group.get('lr_scale', 1.0) * scale
    return base_lr * scale


# ────────────────────────────────────────────────────────────
# Training loop (no validation — all data goes to training)
# ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    use_mixup = args.mixup_alpha > 0

    for i, (videos, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            videos, y_a, y_b, lam = mixup_data(videos, labels, args.mixup_alpha, device)
            logits = model(videos)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(videos)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        preds  = logits.argmax(dim=1)
        # For accuracy: use original labels (ignore mixup)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if (i + 1) % 20 == 0:
            print(f"  [Epoch {epoch}] Batch [{i+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} | Running Acc: {correct/total:.3f}")

    f1 = 0.0
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return total_loss / len(loader), correct / total, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names=None, verbose=True):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(videos)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    f1 = 0.0
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        if verbose and class_names:
            print(classification_report(all_labels, all_preds,
                                        target_names=class_names, zero_division=0))
    return total_loss / len(loader), correct / total, f1


# ────────────────────────────────────────────────────────────
# CSV logger
# ────────────────────────────────────────────────────────────

def init_csv(path):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'lr', 'train_loss', 'train_acc', 'train_f1', 'elapsed_s'])


def log_csv(path, epoch, lr, loss, acc, f1, elapsed):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(
            [epoch, f'{lr:.2e}', f'{loss:.4f}', f'{acc:.4f}', f'{f1:.4f}', f'{elapsed:.1f}'])


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*64}")
    print(f"  Vision Mamba Crime Detection v3  [{device}]")
    print(f"{'='*64}")
    print(f"  Encoder        : {args.encoder} (2D, ImageNet pretrained)")
    print(f"  Unfreeze blocks: {args.unfreeze_blocks}")
    print(f"  Mamba layers   : {args.mamba_layers}")
    print(f"  Frames/video   : {args.num_frames}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Epochs         : {args.epochs}  (warmup={args.warmup_epochs})")
    print(f"  Peak head LR   : {args.lr}")
    print(f"  Encoder LR     : {args.lr * args.encoder_lr_scale:.2e}")
    print(f"  MixUp alpha    : {args.mixup_alpha}")
    print(f"  Train/Test     : {1-args.test_split:.0%} / {args.test_split:.0%}")
    print(f"{'='*64}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.log_csv)
    init_csv(csv_path)

    # ── Datasets ──────────────────────────────────────────────────────────
    print("Loading dataset ...")
    train_dataset = CrimeVideoDataset(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        use_key_frames=True,
        is_train=True,
    )
    test_dataset = CrimeVideoDataset(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        use_key_frames=False,
        is_train=False,
    )

    # 85/15 split (deterministic)
    total      = len(train_dataset)
    test_size  = int(total * args.test_split)
    train_size = total - test_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(
        train_dataset, [train_size, test_size], generator=generator)
    # test subset uses non-augmented dataset
    from torch.utils.data import Subset
    test_subset = Subset(test_dataset, test_ds.indices)

    print(f"  Train: {train_size} | Test: {test_size}\n")

    # WeightedRandomSampler for class balance
    all_weights   = train_dataset.get_class_weights()
    train_weights = all_weights[train_ds.indices]
    sampler       = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_subset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("Building model ...")
    pretrained_ckpt = args.ckpt.strip() if args.ckpt else None
    model = CrimeDetectionModel(
        encoder_name=args.encoder,
        num_classes=NUM_CLASSES,
        num_frames=args.num_frames,
        pretrained_ckpt=pretrained_ckpt,
        freeze_encoder=True,
        unfreeze_blocks=args.unfreeze_blocks,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_layers=args.mamba_layers,
    ).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_p:,}")
    print(f"  Trainable params : {trainable_p:,}\n")

    # ── Optimizer ─────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(encoder_lr_scale=args.encoder_lr_scale)
    for g in param_groups:
        g['lr'] = args.lr * g['lr_scale']

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_train_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_train_acc = ckpt.get('train_acc', 0.0)
        print(f"Resumed from epoch {start_epoch-1}, best_train_acc={best_train_acc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"{'='*64}")
    print(f"  Training -- {args.epochs} epochs (no validation)")
    print(f"{'='*64}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        current_lr = apply_lr(
            optimizer, args.lr, epoch - 1, args.warmup_epochs, args.epochs)

        loss, acc, f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{args.epochs} ({elapsed:.0f}s) | LR={current_lr:.2e} | "
              f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f}")

        log_csv(csv_path, epoch, current_lr, loss, acc, f1, elapsed)

        # Save best model (by train accuracy)
        ckpt_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc':            acc,
            'args':                 vars(args),
        }

        if acc > best_train_acc:
            best_train_acc = acc
            best_path = os.path.join(args.output_dir, 'best_crime_model.pth')
            torch.save(ckpt_data, best_path)
            print(f"  [BEST] Saved best model (train_acc={acc:.4f}) -> {best_path}")

        torch.save(ckpt_data,
                   os.path.join(args.output_dir, 'latest_crime_model.pth'))

    print(f"\n{'='*64}")
    print(f"  Training complete. Best train acc: {best_train_acc:.4f}")
    print(f"  Log -> {csv_path}")
    print(f"{'='*64}\n")

    # ── Final test evaluation ──────────────────────────────────────────────
    print("Evaluating on held-out test set ...")
    best_path = os.path.join(args.output_dir, 'best_crime_model.pth')
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device,
        class_names=CRIME_CLASSES, verbose=True)

    print(f"\nFinal Test Results:")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  Acc  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  F1   : {test_f1:.4f}")


if __name__ == '__main__':
    main()
