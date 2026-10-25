import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from dataset_finetune import LangBindDataset
from classifier import LVMAEClassifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEST_DIR = "/home/y_rajeshwar/final_year_project_29/dataset/2sec/langbind_features_clean_80_20/test"

MODEL_PATH ="./finetune_output_lr5e5/final_model.pth"

NUM_PATCHES = 20
BATCH_SIZE = 16


# ---------------------------------------------------------
# Model arguments
# ---------------------------------------------------------

class Args:
    model = "mae_vit_base_custom"
    input_size = 1024

    depth = 12
    num_heads = 8

    decoder_embed_dim = 512
    decoder_depth = 8
    decoder_num_heads = 16

    num_patches = 20

    norm_pix_loss = False
    cos_loss = False


args = Args()


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)
print(device)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = LangBindDataset(
    TEST_DIR,
    num_patches=NUM_PATCHES
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

class_names = test_dataset.classes

print("Test videos :", len(test_dataset))
print("Classes     :", len(class_names))


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = LVMAEClassifier(
    args,
    num_classes=len(class_names)
)


# ---------------------------------------------------------
# Load FINAL fine-tuned weights
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING FINAL MODEL")
print("=" * 70)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

msg = model.load_state_dict(
    state_dict,
    strict=True
)

print(msg)

model.to(device)
model.eval()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

all_labels = []
all_predictions = []


print("\n" + "=" * 70)
print("EVALUATING")
print("=" * 70)


with torch.no_grad():

    for features, lengths, labels in test_loader:

        features = features.to(device)
        lengths = lengths.to(device)

        outputs = model(
            features,
            lengths
        )

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# ---------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy * 100:.2f}%")


# ---------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------

precision_macro, recall_macro, f1_macro, _ = \
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


precision_weighted, recall_weighted, f1_weighted, _ = \
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


print(f"Macro Precision   : {precision_macro:.4f}")
print(f"Macro Recall      : {recall_macro:.4f}")
print(f"Macro F1-score    : {f1_macro:.4f}")

print()

print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall   : {recall_weighted:.4f}")
print(f"Weighted F1-score : {f1_weighted:.4f}")


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for i, class_name in enumerate(class_names):

    total = cm[i].sum()
    correct = cm[i, i]

    class_acc = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    print(
        f"{class_name:20s}: "
        f"{correct:3d}/{total:3d} "
        f"({class_acc:6.2f}%)"
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)


