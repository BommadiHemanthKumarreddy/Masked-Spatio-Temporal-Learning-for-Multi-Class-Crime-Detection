"""
Parses VideoMAEv2's raw prediction file (e.g. checkpoints/finetune_v8/0.txt),
merges multiple crops per video (averaging softmax probabilities), then computes
precision/recall/F1 per class and an overall confusion matrix.

File format (one line per crop):
    <video_name> [13 raw logits] <true_label> <chunk_nb> <split_nb>
The very first line is a running "acc1, acc5" summary and is skipped.

Usage:
    python compute_metrics.py --preds_file checkpoints/finetune_v8/0.txt \
        --classes_file /home/y_rajeshwar/final_year_project_29/data_lists/classes.txt \
        --out_dir plots/
"""
import argparse
import ast
import os
from collections import defaultdict

import numpy as np

try:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  precision_recall_fscore_support)
except ImportError:
    raise SystemExit("Run: pip install scikit-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def parse_preds_file(preds_file):
    """Returns dict: video_name -> {'true': int, 'probs': [list of softmax vectors, one per crop]}"""
    videos = defaultdict(lambda: {"true": None, "probs": []})

    with open(preds_file, "r") as f:
        lines = f.readlines()

    # First line is a running "acc1, acc5" summary -- skip it
    data_lines = lines[1:]

    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        # video name is everything before the first '[', logits are inside [...]
        bracket_start = line.index("[")
        bracket_end = line.index("]")
        video_name = line[:bracket_start].strip()
        logits_str = line[bracket_start:bracket_end + 1]
        logits = ast.literal_eval(logits_str)

        rest = line[bracket_end + 1:].split()
        true_label = int(rest[0])
        # rest[1], rest[2] are chunk_nb, split_nb -- not needed for merging

        videos[video_name]["true"] = true_label
        videos[video_name]["probs"].append(softmax(logits))

    return videos


def merge_and_predict(videos):
    y_true, y_pred = [], []
    for video_name, d in videos.items():
        avg_prob = np.mean(d["probs"], axis=0)
        pred = int(np.argmax(avg_prob))
        y_true.append(d["true"])
        y_pred.append(pred)
    return np.array(y_true), np.array(y_pred)


def load_class_names(classes_file, num_classes):
    if classes_file and os.path.exists(classes_file):
        names = {}
        with open(classes_file, "r") as f:
            for line in f:
                idx, name = line.strip().split(" ", 1)
                names[int(idx)] = name
        return [names.get(i, str(i)) for i in range(num_classes)]
    return [str(i) for i in range(num_classes)]


def plot_confusion_matrix(cm, class_names, out_dir):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_precision_recall_bar(class_names, precision, recall, f1, out_dir):
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.xticks(x, class_names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, F1")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "precision_recall_f1.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_file", required=True)
    ap.add_argument("--classes_file", default=None, help="optional: maps class idx -> name")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    videos = parse_preds_file(args.preds_file)
    print(f"Found {len(videos)} unique videos "
          f"({sum(len(v['probs']) for v in videos.values())} total crop lines)")

    y_true, y_pred = merge_and_predict(videos)
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    class_names = load_class_names(args.classes_file, num_classes)

    overall_acc = (y_true == y_pred).mean() * 100
    print(f"\nMerged Top-1 accuracy (sanity check): {overall_acc:.2f}%")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0)

    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name:20s} precision={precision[i]:.3f} recall={recall[i]:.3f} "
              f"f1={f1[i]:.3f} support={support[i]}")

    print("\nMacro avg  -- precision={:.3f} recall={:.3f} f1={:.3f}".format(
        precision.mean(), recall.mean(), f1.mean()))

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plot_confusion_matrix(cm, class_names, args.out_dir)
    plot_precision_recall_bar(class_names, precision, recall, f1, args.out_dir)


if __name__ == "__main__":
    main()
