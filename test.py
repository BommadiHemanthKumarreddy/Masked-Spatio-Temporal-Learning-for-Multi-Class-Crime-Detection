import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset_finetune import LangBindDataset
from classifier import LVMAEClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--model", default="mae_vit_base_custom")
    parser.add_argument("--input_size", default=1024, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--num_patches", default=256, type=int)

    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--cos_loss", action="store_true")

    return parser.parse_args()


def main():

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LangBindDataset(args.test_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = LVMAEClassifier(
        args,
        num_classes=len(dataset.classes)
    )

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)

            outputs = model(features)

            preds = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print("\nAccuracy")
    print(accuracy_score(y_true, y_pred))

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=dataset.classes
        )
    )

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
