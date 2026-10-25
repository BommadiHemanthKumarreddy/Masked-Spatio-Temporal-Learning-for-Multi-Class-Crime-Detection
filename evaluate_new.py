import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

import models_mae


# =========================================================
# Configuration
# =========================================================

TEST_DIR = "/home/y_rajeshwar/final_year_project_29/langbind_features_new_one/test"

MODEL_PATH = (
    "/home/y_rajeshwar/final_year_project_29/v_mea2/"
    "lv-mae-ucfcrime/finetune_output_new/final_model.pth"
)

BATCH_SIZE = 16
NUM_PATCHES = 20
EMBEDDING_DIM = 1024


# =========================================================
# Dataset
# =========================================================

class OneVideoLangBindDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        # -------------------------------------------------
        # Find class folders
        # -------------------------------------------------

        self.classes = sorted([
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        # -------------------------------------------------
        # Collect one embedding for every video
        # -------------------------------------------------

        for cls in self.classes:

            class_dir = os.path.join(
                root_dir,
                cls
            )

            for video_name in sorted(
                os.listdir(class_dir)
            ):

                video_dir = os.path.join(
                    class_dir,
                    video_name
                )

                if not os.path.isdir(video_dir):
                    continue

                embedding_path = os.path.join(
                    video_dir,
                    "langbind_output.pt"
                )

                if os.path.isfile(embedding_path):

                    self.samples.append(
                        (
                            embedding_path,
                            self.class_to_idx[cls]
                        )
                    )

        # -------------------------------------------------
        # Print dataset information
        # -------------------------------------------------

        print("\nDataset structure")
        print("-----------------")
        print("Root       :", root_dir)
        print("Classes    :", self.classes)
        print("Num classes:", len(self.classes))
        print("Videos     :", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        embedding_path, label = self.samples[idx]

        # -------------------------------------------------
        # Load LanguageBind output
        # -------------------------------------------------

        data = torch.load(
            embedding_path,
            map_location="cpu"
        )

        embedding = data["embedding"].float().flatten()

        # -------------------------------------------------
        # Verify embedding dimension
        # -------------------------------------------------

        if embedding.shape[0] != EMBEDDING_DIM:

            raise RuntimeError(
                f"Expected embedding dimension "
                f"{EMBEDDING_DIM}, but got "
                f"{embedding.shape[0]} "
                f"for:\n{embedding_path}"
            )

        # -------------------------------------------------
        # One video = one real embedding
        #
        # Fine-tuned LV-MAE expects:
        #
        #     20 patch positions
        #
        # Therefore:
        #
        #     position 0  -> actual video embedding
        #     positions 1-19 -> zero padding
        # -------------------------------------------------

        features = torch.zeros(
            NUM_PATCHES,
            EMBEDDING_DIM,
            dtype=torch.float32
        )

        # Put actual video embedding in first position
        features[0] = embedding

        # Only first position is valid
        length = torch.tensor(
            1,
            dtype=torch.long
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, length, label


# =========================================================
# LV-MAE Classifier
# =========================================================

class LVMAEClassifier(nn.Module):

    def __init__(self, args, num_classes=13):

        super().__init__()

        self.args = args

        # -------------------------------------------------
        # Build LV-MAE model
        # -------------------------------------------------

        self.mae = models_mae.__dict__[args.model](
            args=args,
            norm_pix_loss=args.norm_pix_loss
        )

        # -------------------------------------------------
        # Remove decoder
        # Decoder is not required for classification
        # -------------------------------------------------

        del self.mae.decoder_blocks
        del self.mae.decoder_embed
        del self.mae.decoder_norm
        del self.mae.decoder_pred
        del self.mae.decoder_pos_embed
        del self.mae.mask_token

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                args.input_size,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.5
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x, length):

        # x:
        # [B, 20, 1024]
        #
        # length:
        # [B]
        #
        # For the one-video-one-embedding setup:
        # length = 1

        B, N, _ = x.shape

        # -------------------------------------------------
        # Create padding mask
        # -------------------------------------------------

        positions = torch.arange(
            N,
            device=x.device
        ).unsqueeze(0)

        padding_mask = (
            positions >= length.unsqueeze(1)
        )

        # -------------------------------------------------
        # CLS token is always valid
        # -------------------------------------------------

        cls_mask = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=x.device
        )

        # -------------------------------------------------
        # Final attention mask
        #
        # [CLS] + [20 video positions]
        #
        # Shape:
        # [B, 21]
        # -------------------------------------------------

        attn_mask = torch.cat(
            [
                cls_mask,
                padding_mask
            ],
            dim=1
        )

        # -------------------------------------------------
        # LV-MAE encoder
        # -------------------------------------------------

        latent = self.mae.forward_encoder(
            x=x,
            attn_mask=attn_mask,
            ids_shuffle=None,
            length=length
        )

        # -------------------------------------------------
        # Extract CLS representation
        # -------------------------------------------------

        cls_feature = latent[:, 0]

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        logits = self.classifier(
            cls_feature
        )

        return logits


# =========================================================
# Model arguments
# =========================================================

class Args:

    model = "mae_vit_base_custom"

    input_size = 1024

    depth = 12
    num_heads = 8

    decoder_embed_dim = 512
    decoder_depth = 8
    decoder_num_heads = 16

    # IMPORTANT:
    # The fine-tuned checkpoint was trained
    # with 20 patches.
    num_patches = 20

    norm_pix_loss = False
    cos_loss = False


args = Args()


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 70)
print("DEVICE")
print("=" * 70)

print(device)


# =========================================================
# Load test dataset
# =========================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = OneVideoLangBindDataset(
    TEST_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

class_names = test_dataset.classes


# =========================================================
# Dataset sanity check
# =========================================================

print("\n" + "=" * 70)
print("DATASET SANITY CHECK")
print("=" * 70)

sample_features, sample_length, sample_label = (
    test_dataset[0]
)

print(
    "Feature shape :",
    sample_features.shape
)

print(
    "Length        :",
    sample_length
)

print(
    "Label         :",
    sample_label
)

print(
    "Expected shape:",
    f"torch.Size([{NUM_PATCHES}, {EMBEDDING_DIM}])"
)

print("=" * 70)


# =========================================================
# Build model
# =========================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = LVMAEClassifier(
    args,
    num_classes=len(class_names)
)


# =========================================================
# Verify model positional embedding
# =========================================================

print("\n" + "=" * 70)
print("MODEL SHAPE CHECK")
print("=" * 70)

print(
    "Model positional embedding:",
    model.mae.pos_embed.shape
)

print(
    "Expected positional embedding:",
    f"torch.Size([1, {NUM_PATCHES + 1}, {EMBEDDING_DIM}])"
)

print("=" * 70)


# =========================================================
# Load final fine-tuned weights
# =========================================================

print("\n" + "=" * 70)
print("LOADING FINAL FINE-TUNED MODEL")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH
)

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"Fine-tuned model not found:\n{MODEL_PATH}"
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

msg = model.load_state_dict(
    state_dict,
    strict=True
)

print(
    "Checkpoint loaded successfully."
)

print(msg)


# =========================================================
# Move model to device
# =========================================================

model.to(device)

model.eval()


# =========================================================
# Evaluation
# =========================================================

all_labels = []
all_predictions = []


print("\n" + "=" * 70)
print("EVALUATING")
print("=" * 70)


with torch.no_grad():

    for batch_idx, (
        features,
        lengths,
        labels
    ) in enumerate(test_loader):

        # -------------------------------------------------
        # Move tensors to GPU
        # -------------------------------------------------

        features = features.to(device)

        lengths = lengths.to(device)

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        outputs = model(
            features,
            lengths
        )

        # -------------------------------------------------
        # Get predicted class
        # -------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            batch_idx + 1
        ) % 5 == 0:

            print(
                f"Processed "
                f"{(batch_idx + 1) * BATCH_SIZE} "
                f"/ {len(test_dataset)} videos"
            )


# =========================================================
# Convert results to NumPy
# =========================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# =========================================================
# Verify number of predictions
# =========================================================

print("\n" + "=" * 70)
print("PREDICTION CHECK")
print("=" * 70)

print(
    "Test videos      :",
    len(test_dataset)
)

print(
    "Ground-truth labels:",
    len(all_labels)
)

print(
    "Predictions       :",
    len(all_predictions)
)

if len(all_labels) != len(test_dataset):

    raise RuntimeError(
        "Number of labels does not match "
        "number of test videos."
    )

if len(all_predictions) != len(test_dataset):

    raise RuntimeError(
        "Number of predictions does not match "
        "number of test videos."
    )

print("Prediction count check: PASSED")


# =========================================================
# Overall accuracy
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# Precision / Recall / F1
# =========================================================

(
    precision_macro,
    recall_macro,
    f1_macro,
    _
) = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


(
    precision_weighted,
    recall_weighted,
    f1_weighted,
    _
) = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print(
    f"Macro Precision   : "
    f"{precision_macro:.4f}"
)

print(
    f"Macro Recall      : "
    f"{recall_macro:.4f}"
)

print(
    f"Macro F1-score    : "
    f"{f1_macro:.4f}"
)

print()

print(
    f"Weighted Precision: "
    f"{precision_weighted:.4f}"
)

print(
    f"Weighted Recall   : "
    f"{recall_weighted:.4f}"
)

print(
    f"Weighted F1-score : "
    f"{f1_weighted:.4f}"
)


# =========================================================
# Classification report
# =========================================================

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


# =========================================================
# Confusion matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# =========================================================
# Per-class accuracy
# =========================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for i, class_name in enumerate(
    class_names
):

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


# =========================================================
# Final
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Final Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro F1-score: "
    f"{f1_macro:.4f}"
)

print("=" * 70)
