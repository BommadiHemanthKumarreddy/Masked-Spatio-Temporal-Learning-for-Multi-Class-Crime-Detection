# --------------------------------------------------------
# Training Script for Crime Detection Model
# Task 8: CrossEntropyLoss, AdamW, Accuracy, F1-score
# --------------------------------------------------------

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from crime_model import CrimeDetectionModel, CRIME_CLASSES, NUM_CLASSES
from crime_dataset import CrimeVideoDataset

try:
    from sklearn.metrics import f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not found. F1-score will not be computed.")


def get_args():
    parser = argparse.ArgumentParser('Crime Detection Training')
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory with class subfolders')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data for validation')

    # Model
    parser.add_argument('--encoder', type=str, default='vit_small_patch16_224',
                        help='Encoder model name')
    parser.add_argument('--ckpt', type=str, default='vit_s_k710_dl_from_giant.pth',
                        help='Pretrained encoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze encoder weights')
    parser.add_argument('--num_frames', type=int, default=16)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=2)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward
        logits = model(videos)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    f1 = 0.0
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        logits = model(videos)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    f1 = 0.0
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(
            all_labels, all_preds,
            target_names=CRIME_CLASSES,
            zero_division=0,
        )
        print(f"\n{report}")

    return avg_loss, accuracy, f1


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dataset ----
    print(f"\nLoading dataset from: {args.data_dir}")
    full_dataset = CrimeVideoDataset(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        use_key_frames=True,
    )

    # Train/Val/Test split (60% Train, 20% Val, 20% Test)
    val_size = int(len(full_dataset) * 0.20)
    test_size = int(len(full_dataset) * 0.20)
    train_size = len(full_dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    print(f"\nBuilding model: {args.encoder}")
    model = CrimeDetectionModel(
        encoder_name=args.encoder,
        num_classes=NUM_CLASSES,
        num_frames=args.num_frames,
        pretrained_ckpt=args.ckpt,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ---- Training Setup (Task 8) ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ---- Training Loop ----
    best_val_acc = 0.0
    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, 'best_crime_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'args': vars(args),
            }, save_path)
            print(f"  ★ Best model saved (val_acc={val_acc:.4f})")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, os.path.join(args.output_dir, 'latest_crime_model.pth'))

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    # ---- Final Evaluation on Test Set ----
    print("\n" + "=" * 50)
    print("Evaluating Best Model on Test Set (20%)")
    print("=" * 50)
    
    best_ckpt_path = os.path.join(args.output_dir, 'best_crime_model.pth')
    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
        model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")


if __name__ == '__main__':
    main()
