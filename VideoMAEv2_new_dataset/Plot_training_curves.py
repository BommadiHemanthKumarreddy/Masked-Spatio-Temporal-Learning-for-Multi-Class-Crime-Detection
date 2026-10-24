"""
Parses VideoMAEv2's log.txt (one JSON object per line, written by run_class_finetuning.py)
and generates the plots you need for your PPT:
  - Epoch vs Loss (train + val on same graph)
  - Epoch vs Accuracy (train + val on same graph, if train_class_acc is logged)

Run on the SAME server where your training happened, pointing at the log.txt
inside your training output_dir (e.g. checkpoints/finetune_v8/log.txt).

Usage:
    python plot_training_curves.py --log_file checkpoints/finetune_v8/log.txt --out_dir plots/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed on a headless server
import matplotlib.pyplot as plt


def load_log(log_file):
    epochs = []
    train_loss, val_loss = [], []
    val_acc1, val_acc5 = [], []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip any non-JSON lines (e.g. stray prints that leaked in)

            if "epoch" not in row:
                continue  # skip the final merged-results line, which has no 'epoch' key

            epochs.append(row["epoch"])
            train_loss.append(row.get("train_loss"))
            val_loss.append(row.get("val_loss"))

            val_acc1.append(row.get("val_acc1"))
            val_acc5.append(row.get("val_acc5"))

    return epochs, train_loss, val_loss, val_acc1, val_acc5


def plot_loss(epochs, train_loss, val_loss, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy(epochs, val_acc1, val_acc5, out_dir):
    # Note: this repo does not log a per-batch train accuracy (mixup makes it
    # not very meaningful), so only validation Top-1/Top-5 are plotted here.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc1, label="Val Top-1 Accuracy", marker="o", markersize=3)
    plt.plot(epochs, val_acc5, label="Val Top-5 Accuracy", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True, help="path to log.txt from your training run")
    ap.add_argument("--out_dir", default="plots", help="where to save the PNG plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    epochs, train_loss, val_loss, val_acc1, val_acc5 = load_log(args.log_file)

    print(f"Parsed {len(epochs)} epoch rows from {args.log_file}")
    if len(epochs) == 0:
        print("No epoch rows found. Print the first few lines of your log.txt to check its format:")
        print("    head -5", args.log_file)
        return

    plot_loss(epochs, train_loss, val_loss, args.out_dir)
    plot_accuracy(epochs, val_acc1, val_acc5, args.out_dir)


if __name__ == "__main__":
    main()
