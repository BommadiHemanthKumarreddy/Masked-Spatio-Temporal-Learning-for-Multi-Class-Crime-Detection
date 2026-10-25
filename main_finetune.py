import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_finetune import LangBindDataset
from classifier import LVMAEClassifier
from engine_finetune import train_one_epoch, validate


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        required=True
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=""
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./finetune_output_summary"
    )

    parser.add_argument(
        "--model",
        default="mae_vit_base_custom"
    )

    parser.add_argument(
        "--input_size",
        default=1024,
        type=int
    )

    parser.add_argument(
        "--depth",
        default=12,
        type=int
    )

    parser.add_argument(
        "--num_heads",
        default=8,
        type=int
    )

    parser.add_argument(
        "--decoder_embed_dim",
        default=512,
        type=int
    )

    parser.add_argument(
        "--decoder_depth",
        default=8,
        type=int
    )

    parser.add_argument(
        "--decoder_num_heads",
        default=16,
        type=int
    )

    parser.add_argument(
        "--num_patches",
        default=20,
        type=int
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true"
    )

    parser.add_argument(
        "--cos_loss",
        action="store_true"
    )

    return parser.parse_args()


def main():

    args = get_args()

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)
    print(device)

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    train_dataset = LangBindDataset(
        args.train_dir,
        num_patches=args.num_patches
    )

    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_dataset = LangBindDataset(
        args.test_dir,
        num_patches=args.num_patches
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print()
    print(f"Training videos : {len(train_dataset)}")
    print(f"Test videos     : {len(test_dataset)}")
    print(f"Classes         : {len(train_dataset.classes)}")
    print()

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = LVMAEClassifier(
        args,
        num_classes=len(train_dataset.classes)
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )

    start_epoch = 0

    # ------------------------------------------------------------
    # PRETRAINED LV-MAE
    # ------------------------------------------------------------

    if args.resume != "":

        print("=" * 70)
        print("RESUMING")
        print("=" * 70)

        checkpoint = torch.load(
            args.resume,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_epoch = checkpoint["epoch"] + 1

    else:

        model.load_pretrained(
            args.pretrained
        )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("=" * 70)
    print(f"STARTING SUMMARY TOKEN TRAINING FOR {args.epochs} EPOCHS")
    print("=" * 70)

    best_test_acc = 0.0

    for epoch in range(start_epoch, args.epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        test_loss, test_acc = validate(
            model,
            test_loader,
            criterion,
            device
        )

        print()
        print("=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Test Loss  : {test_loss:.4f}")
        print(f"Test Acc   : {test_acc:.2f}%")
        print("=" * 70)

        # Latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            },
            os.path.join(
                args.save_dir,
                "checkpoint_latest.pth"
            )
        )

        # Best checkpoint
        if test_acc > best_test_acc:

            best_test_acc = test_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                },
                os.path.join(
                    args.save_dir,
                    "best_model.pth"
                )
            )

            print(
                f"New best test accuracy: {best_test_acc:.2f}%"
            )

    # ------------------------------------------------------------
    # FINAL MODEL
    # ------------------------------------------------------------

    final_model_path = os.path.join(
        args.save_dir,
        "final_model.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print()
    print("=" * 70)
    print("SUMMARY TOKEN TRAINING FINISHED")
    print("=" * 70)
    print(f"Best Test Accuracy : {best_test_acc:.2f}%")
    print(f"Final Model        : {final_model_path}")
    print(
        f"Best Checkpoint    : "
        f"{os.path.join(args.save_dir, 'best_model.pth')}"
    )


if __name__ == "__main__":
    main()
