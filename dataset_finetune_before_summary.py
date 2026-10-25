import os
import torch
from torch.utils.data import Dataset


class LangBindDataset(Dataset):

    def __init__(self, root_dir, num_patches=20):

        self.samples = []
        self.num_patches = num_patches

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        for cls in self.classes:

            cls_path = os.path.join(root_dir, cls)

            for video in sorted(os.listdir(cls_path)):

                video_path = os.path.join(cls_path, video)

                if not os.path.isdir(video_path):
                    continue

                segment_paths = []

                for segment in sorted(os.listdir(video_path)):

                    pt = os.path.join(
                        video_path,
                        segment,
                        "langbind_output.pt"
                    )

                    if os.path.isfile(pt):
                        segment_paths.append(pt)

                if segment_paths:

                    self.samples.append(
                        (
                            segment_paths[:self.num_patches],
                            self.class_to_idx[cls]
                        )
                    )

        print(f"Loaded {len(self.samples)} videos")
        print(self.class_to_idx)


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        segment_paths, label = self.samples[idx]

        embeddings = []

        for path in segment_paths:

            data = torch.load(path, map_location="cpu")

            embedding = data["embedding"].float().flatten()

            embeddings.append(embedding)

        features = torch.stack(embeddings)

        length = features.shape[0]

        # Pad sequence to num_patches
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

        return features, length, label
