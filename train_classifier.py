import os, sys, glob, re, yaml, torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.crime_dataset import CrimeVideoDataset
from models.crime_classifier import VideoMAE_CrimeClassifier


def evaluate(model, loader, criterion, device, amp_fp16):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for videos, labels, _ in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp_fp16):
                logits, _ = model(videos)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)

    avg_loss = total_loss / max(len(loader), 1)
    acc  = accuracy_score(all_targets, all_preds)
    prec, rec, f1_m, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return avg_loss, acc, prec, rec, f1_m, f1_w, cm


def find_best_pretrain_ckpt(ckpt_dir, requested_epoch):
    target_path = os.path.join(ckpt_dir, f"videomae_pretrain_epoch_{requested_epoch}.pth")
    if os.path.exists(target_path):
        return target_path

    # Fallback: search for available pretraining checkpoints
    pattern = os.path.join(ckpt_dir, "videomae_pretrain_epoch_*.pth")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None

    def extract_epoch(p):
        match = re.search(r"videomae_pretrain_epoch_(\d+)\.pth", p)
        return int(match.group(1)) if match else -1

    ckpts.sort(key=extract_epoch, reverse=True)
    best_fallback = ckpts[0]
    print(f"[Warning] Requested pretrain epoch {requested_epoch} not found at {target_path}.")
    print(f"[Fallback] Found highest pretrain checkpoint: {best_fallback}")
    return best_fallback


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_frames  = cfg["video"]["num_frames"]
    image_size  = cfg["video"]["image_size"]
    patch_size  = cfg["video"]["patch_size"]
    tubelet     = cfg["video"]["tubelet_size"]
    num_classes = cfg["dataset"]["num_classes"]
    batch_size  = cfg["training"]["batch_size"]
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["lr"]
    wd          = cfg["training"]["weight_decay"]
    amp_fp16    = cfg["training"]["amp_fp16"]
    num_workers = cfg["training"].get("num_workers", 0)
    pretrain_ep = cfg["training"].get("pretrain_epoch", 20)

    print(f"Using device: {device}")

    # 1. Dataset & Data Loaders (Train / Validation / Test Separation)
    full_train_ds = CrimeVideoDataset(cfg["dataset"]["root"], is_train=True,
                                      num_frames=num_frames, crop_size=image_size)
    test_ds       = CrimeVideoDataset(cfg["dataset"]["root"], is_train=False,
                                      num_frames=num_frames, crop_size=image_size)

    # 80/20 train/validation split with fixed seed
    val_size   = int(0.20 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(
        full_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True, pin_memory=False)
    train_eval_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

    print(f"Dataset Split -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 2. Model Initialization
    model = VideoMAE_CrimeClassifier(
        img_size=image_size, patch_size=patch_size,
        num_classes=num_classes, embed_dim=768, depth=12, num_heads=12,
        tubelet_size=tubelet, all_frames=num_frames,
        mamba_layers=cfg["mamba"]["num_layers"], freeze_encoder=False
    ).to(device)

    # 3. Load Pretrained VideoMAE Encoder Weights
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    pretrain_ckpt = find_best_pretrain_ckpt(ckpt_dir, pretrain_ep)

    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        state = torch.load(pretrain_ckpt, map_location=device)
        encoder_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[Success] Loaded pretrain encoder from '{pretrain_ckpt}': {len(encoder_state)} keys | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[WARNING] No pretrain checkpoint found! Training classifier from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler    = torch.cuda.amp.GradScaler(enabled=amp_fp16)

    best_val_f1 = 0.0
    print("Starting Stage 2 Fine-Tuning for Crime Classification...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_step_loss = 0.0
        for step, (videos, labels, _) in enumerate(train_ld):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp_fp16):
                logits, _ = model(videos)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_step_loss += loss.item()

        # Compute full metrics for Train, Validation, and Test sets at end of epoch
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1_m, tr_f1_w, _ = evaluate(model, train_eval_ld, criterion, device, amp_fp16)
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1_m, vl_f1_w, _ = evaluate(model, val_ld, criterion, device, amp_fp16)
        te_loss, te_acc, te_prec, te_rec, te_f1_m, te_f1_w, _ = evaluate(model, test_ld, criterion, device, amp_fp16)

        print(f"\n==================== Epoch [{epoch}/{epochs}] Comprehensive Metrics ====================")
        print(f"TRAIN -> Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.2f}% | Prec: {tr_prec:.4f} | Rec: {tr_rec:.4f} | MacroF1: {tr_f1_m:.4f} | WeightedF1: {tr_f1_w:.4f}")
        print(f"VAL   -> Loss: {vl_loss:.4f} | Acc: {vl_acc*100:.2f}% | Prec: {vl_prec:.4f} | Rec: {vl_rec:.4f} | MacroF1: {vl_f1_m:.4f} | WeightedF1: {vl_f1_w:.4f}")
        print(f"TEST  -> Loss: {te_loss:.4f} | Acc: {te_acc*100:.2f}% | Prec: {te_prec:.4f} | Rec: {te_rec:.4f} | MacroF1: {te_f1_m:.4f} | WeightedF1: {te_f1_w:.4f}")
        print("==========================================================================================")

        if vl_f1_m > best_val_f1:
            best_val_f1 = vl_f1_m
            save_path = os.path.join(ckpt_dir, "best_crime_classifier.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ** New best model saved to {save_path} (Val Macro F1: {vl_f1_m:.4f})")

    # 4. Final Summary Evaluation on Best Checkpoint
    print("\n" + "="*70)
    print("Stage 2 Fine-Tuning Complete. Loading Best Saved Checkpoint for Final Report...")
    best_ckpt_path = os.path.join(ckpt_dir, "best_crime_classifier.pth")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    final_loss, final_acc, final_prec, final_rec, final_f1_m, final_f1_w, final_cm = evaluate(model, test_ld, criterion, device, amp_fp16)
    print("="*70)
    print("FINAL UNSEEN TEST EVALUATION SUMMARY:")
    print(f"Test Loss        : {final_loss:.4f}")
    print(f"Test Accuracy    : {final_acc*100:.2f}%")
    print(f"Test Precision   : {final_prec:.4f}")
    print(f"Test Recall      : {final_rec:.4f}")
    print(f"Test Macro F1    : {final_f1_m:.4f}")
    print(f"Test Weighted F1 : {final_f1_w:.4f}")
    print(f"Confusion Matrix :\n{final_cm}")
    print("="*70)


if __name__ == "__main__":
    main()
