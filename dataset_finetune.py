import os
import re
import torch
from torch.utils.data import Dataset


class LangBindDataset(Dataset):

    def __init__(self, root_dir, num_patches=20):

        self.samples = []
        self.num_patches = num_patches

        # ----------------------------------------------------------
        # Video folders
        # Example:
        # Abuse013_x264
        # Arrest025_x264
        # ----------------------------------------------------------

        self.video_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # ----------------------------------------------------------
        # Extract unique class names
        # Abuse013_x264 -> Abuse
        # ----------------------------------------------------------

        class_names = sorted({
            re.sub(r"\d+_x264$", "", video)
            for video in self.video_folders
        })

        self.classes = class_names

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        # ----------------------------------------------------------
        # Build dataset
        # ----------------------------------------------------------

        for video in self.video_folders:

            video_path = os.path.join(root_dir, video)

            cls = re.sub(r"\d+_x264$", "", video)

            label = self.class_to_idx[cls]

            segment_paths = []

            for segment in sorted(os.listdir(video_path)):

                pt_path = os.path.join(
                    video_path,
                    segment,
                    "langbind_output.pt"
                )

                if os.path.isfile(pt_path):
                    segment_paths.append(pt_path)

            if len(segment_paths) > 0:

                self.samples.append(
                    (
                        segment_paths,
                        label
                    )
                )

        print("=" * 60)
        print(f"Loaded {len(self.samples)} videos")
        print(f"Number of classes : {len(self.classes)}")
        print(self.class_to_idx)
        print("=" * 60)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        segment_paths, label = self.samples[idx]

        # ----------------------------------------------------------
        # Keep only first num_patches segments
        # ----------------------------------------------------------

        segment_paths = segment_paths[:self.num_patches]

        embeddings = []

        for path in segment_paths:

            data = torch.load(path, map_location="cpu")

            embedding = data["embedding"].float().flatten()

            embeddings.append(embedding)

        features = torch.stack(embeddings)

        length = features.shape[0]

        # ----------------------------------------------------------
        # Pad with zeros if fewer than num_patches
        # ----------------------------------------------------------

        if length < self.num_patches:

            padding = torch.zeros(
                self.num_patches - length,
                features.shape[1],
                dtype=features.dtype
            )

            features = torch.cat(
                [features, padding],
                dim=0
            )

        # ----------------------------------------------------------
        # Truncate if longer than num_patches
        # ----------------------------------------------------------

        elif length > self.num_patches:

            features = features[:self.num_patches]
            length = self.num_patches

        return features, length, label
