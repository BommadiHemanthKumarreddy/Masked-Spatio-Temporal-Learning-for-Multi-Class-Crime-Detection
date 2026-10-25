import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/home/y_rajeshwar/final_year_project_29/human_aware_dataset"
OUT_DIR = "/home/y_rajeshwar/final_year_project_29/human_aware_models"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# Load data
# ============================================================

train_data = torch.load(
    os.path.join(DATA_DIR, "train.pt"),
    map_location="cpu",
)

test_data = torch.load(
    os.path.join(DATA_DIR, "test.pt"),
    map_location="cpu",
)

X = train_data["X"].float()
y = train_data["y"].long()

X_test = test_data["X"].float()
y_test = test_data["y"].long()

classes = train_data["classes"]

print("\nFull training data:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# Stratified train/validation split
# ============================================================

indices = np.arange(len(y))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y.numpy(),
)

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

print("\nSplit:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ============================================================
# Class weights
# ============================================================

num_classes = len(classes)

class_counts = torch.bincount(
    y_train,
    minlength=num_classes,
).float()

class_weights = len(y_train) / (
    num_classes * class_counts.clamp(min=1)
)

class_weights = class_weights.to(device)

print("\nClass weights:")
for i, name in enumerate(classes):
    print(
        f"{i:2d} {name:25s} "
        f"count={int(class_counts[i]):4d} "
        f"weight={class_weights[i].item():.4f}"
    )

# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ============================================================
# Linear classifier
# ============================================================

model = nn.Linear(
    X.shape[1],
    num_classes,
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, X_data, y_data):
    model.eval()

    with torch.no_grad():
        logits = model(X_data.to(device))
        predictions = torch.argmax(logits, dim=1).cpu()

    accuracy = accuracy_score(
        y_data.numpy(),
        predictions.numpy(),
    )

    macro_f1 = f1_score(
        y_data.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, predictions

# ============================================================
# Training
# ============================================================

best_val_f1 = -1.0
best_state = None

print("\n" + "=" * 70)
print("TRAINING HUMAN-AWARE LINEAR BASELINE")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(
            logits,
            batch_y,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        total_samples += len(batch_y)

    train_loss = total_loss / total_samples

    train_acc, train_f1, _ = evaluate(
        model,
        X_train,
        y_train,
    )

    val_acc, val_f1, _ = evaluate(
        model,
        X_val,
        y_val,
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Train F1 {train_f1:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

# ============================================================
# Restore best validation model
# ============================================================

model.load_state_dict(best_state)

print("\nBest validation Macro-F1:", best_val_f1)

# ============================================================
# Final test evaluation
# ============================================================

test_acc, test_f1, test_predictions = evaluate(
    model,
    X_test,
    y_test,
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro F1 : {test_f1:.4f}")

# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test.numpy(),
        test_predictions.numpy(),
        labels=list(range(num_classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)

# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test.numpy(),
    test_predictions.numpy(),
    labels=list(range(num_classes)),
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# Save model
# ============================================================

model_path = os.path.join(
    OUT_DIR,
    "human_aware_linear_baseline.pt",
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "num_classes": num_classes,
        "classes": classes,
        "class_to_idx": train_data["class_to_idx"],
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    },
    model_path,
)

print("\nModel saved:")
print(model_path)

print("\nDONE.")
