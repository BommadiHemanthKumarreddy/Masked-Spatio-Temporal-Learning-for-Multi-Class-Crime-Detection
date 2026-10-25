import os
import csv
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset_finetune import LangBindDataset
from classifier import LVMAEClassifier


# ================================================================
# ARGUMENTS
# ================================================================

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./attention_results"
    )

    parser.add_argument(
        "--num_patches",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--model",
        default="mae_vit_base_custom"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=12
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--decoder_embed_dim",
        type=int,
        default=512
    )

    parser.add_argument(
        "--decoder_depth",
        type=int,
        default=8
    )

    parser.add_argument(
        "--decoder_num_heads",


        type=int,
        default=16
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


# ================================================================
# MAIN
# ================================================================

def main():

    args = get_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)

    print(device)

    # ============================================================
    # DATASET
    # ============================================================

    dataset = LangBindDataset(
        args.test_dir,
        num_patches=args.num_patches
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("=" * 70)
    print("TEST DATA")
    print("=" * 70)

    print(
        "Number of videos:",
        len(dataset)
    )

    print(
        "Classes:",
        dataset.classes
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = LVMAEClassifier(
        args,
        num_classes=len(dataset.classes)
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    # ------------------------------------------------------------
    # best_model.pth contains:
    #
    # {
    #     "epoch": ...,
    #     "model": ...,
    #     "optimizer": ...,
    #     ...
    # }
    # ------------------------------------------------------------

    if "model" in checkpoint:

        state_dict = checkpoint["model"]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)

    model.eval()

    print("=" * 70)
    print("MODEL LOADED")
    print("=" * 70)

    # ============================================================
    # OUTPUT CSV
    # ============================================================

    csv_path = os.path.join(
        args.output_dir,
        "temporal_attention.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        newline=""
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "video_index",
            "true_class",
            "predicted_class",
            "segment",
            "start_sec",
            "end_sec",
            "attention"
        ]
    )

    # ============================================================
    # PROCESS VIDEOS
    # ============================================================

    with torch.no_grad():

        for video_index, (
            features,
            lengths,
            labels
        ) in enumerate(loader):

            features = features.to(
                device
            )

            lengths = lengths.to(
                device
            )

            labels = labels.to(
                device
            )

            # ----------------------------------------------------
            # MODEL
            # ----------------------------------------------------

            logits = model(
                features,
                lengths
            )

            predictions = logits.argmax(
                dim=1
            )

            # ----------------------------------------------------
            # GET TEMPORAL ATTENTION
            # ----------------------------------------------------

            attention = (
                model
                .temporal_attention
                .last_attention_weights
            )

            if attention is None:

                raise RuntimeError(
                    "Temporal attention weights were not captured."
                )

            # ----------------------------------------------------
            # Shape:
            #
            # (B, heads, 1, T)
            # ----------------------------------------------------

            attention = attention.mean(
                dim=1
            )

            attention = attention[:, 0, :]

            attention = attention.cpu().numpy()

            lengths_np = (
                lengths
                .cpu()
                .numpy()
            )

            label = labels[
                0
            ].item()

            prediction = predictions[
                0
            ].item()

            true_class = dataset.classes[
                label
            ]

            predicted_class = dataset.classes[
                prediction
            ]

            valid_length = int(
                lengths_np[0]
            )

            # ----------------------------------------------------
            # Only valid segments
            # ----------------------------------------------------

            video_attention = (
                attention[0][
                    :valid_length
                ]
            )

            # ----------------------------------------------------
            # Normalize attention
            # ----------------------------------------------------

            attention_sum = (
                video_attention.sum()
            )

            if attention_sum > 0:

                video_attention = (
                    video_attention /
                    attention_sum
                )

            # ----------------------------------------------------
            # SAVE CSV
            # ----------------------------------------------------

            for segment_idx, weight in enumerate(
                video_attention
            ):

                start_sec = (
                    segment_idx * 2
                )

                end_sec = (
                    start_sec + 2
                )

                csv_writer.writerow(
                    [
                        video_index,
                        true_class,
                        predicted_class,
                        segment_idx + 1,
                        start_sec,
                        end_sec,
                        float(weight)
                    ]
                )

            # ====================================================
            # PLOT
            # ====================================================

            plt.figure(
                figsize=(10, 5)
            )

            segments = np.arange(
                1,
                valid_length + 1
            )

            plt.bar(
                segments,
                video_attention
            )

            plt.xlabel(
                "2-second segment"
            )

            plt.ylabel(
                "Temporal attention"
            )

            plt.title(
                f"Video {video_index} | "
                f"True: {true_class} | "
                f"Predicted: {predicted_class}"
            )

            plt.xticks(
                segments
            )

            plt.tight_layout()

            plot_path = os.path.join(
                args.output_dir,
                f"video_{video_index:03d}_attention.png"
            )

            plt.savefig(
                plot_path,
                dpi=150
            )

            plt.close()

            print(
                f"[{video_index + 1}/{len(dataset)}] "
                f"{true_class} -> {predicted_class} | "
                f"segments={valid_length}"
            )

    csv_file.close()

    print()
    print("=" * 70)
    print("ATTENTION EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        "CSV:",
        csv_path
    )

    print(
        "Plots:",
        args.output_dir
    )


if __name__ == "__main__":
    main()
