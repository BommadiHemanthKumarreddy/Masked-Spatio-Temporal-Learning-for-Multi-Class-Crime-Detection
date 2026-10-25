import os
import torch
from torch.utils.data import Dataset

class LangBindDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        # Get sorted class names
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # Collect all samples
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)

            for video in os.listdir(cls_path):
                feature_path = os.path.join(
                    cls_path,
                    video,
                    "langbind_output.pt"
                )

                if os.path.exists(feature_path):
                    self.samples.append(
                        (
                            feature_path,
                            self.class_to_idx[cls]
                        )
                    )

        print(f"Loaded {len(self.samples)} samples")
        print(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        data = torch.load(feature_path)
        feature = data["embedding"].float()
        return feature, label
