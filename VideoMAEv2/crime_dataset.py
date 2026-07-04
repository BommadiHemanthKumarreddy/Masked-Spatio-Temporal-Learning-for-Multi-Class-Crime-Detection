# --------------------------------------------------------
# Crime Video Dataset
# Loads videos, applies key frame selection & region masking
# --------------------------------------------------------

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import torch.nn.functional as F

from crime_modules import select_key_frames, region_based_mask
from crime_model import CRIME_CLASSES


class CrimeVideoDataset(Dataset):
    """
    Dataset for multi-class crime detection.

    Expects directory structure:
        root/
            dragging/  video1.mp4, video2.mp4, ...
            hitting/   ...
            ...

    Preprocessing pipeline:
        1. Load video with decord
        2. Key frame selection (motion-based)
        3. Resize + Center crop to 224x224
        4. Normalize with ImageNet stats
        5. (Optional) Region-based masking
    """

    def __init__(
        self,
        root_dir,
        num_frames=16,
        img_size=224,
        use_key_frames=True,
        use_region_mask=False,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.use_key_frames = use_key_frames
        self.use_region_mask = use_region_mask

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Build file list
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CRIME_CLASSES)}

        for cls_name in CRIME_CLASSES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARNING] Class directory not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name],
                    ))

        print(f"[INFO] Loaded {len(self.samples)} videos from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            # 1. Load video
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            total_frames = len(vr)

            # Sample more frames than needed for key frame selection
            if self.use_key_frames and total_frames > self.num_frames:
                # Sample 2x frames, then select top-K
                sample_count = min(total_frames, self.num_frames * 4)
                indices = np.linspace(0, total_frames - 1, sample_count).astype(int)
                
                # Fetch unique frames first and index them to avoid duplicate decord get_batch bugs
                unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
                unique_frames = vr.get_batch(unique_indices).asnumpy()
                all_frames = unique_frames[inverse_indices]
                
                frames = select_key_frames(all_frames, k=self.num_frames)
            else:
                # Uniform sampling
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
                
                # Fetch unique frames first and index them to avoid duplicate decord get_batch bugs
                unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
                unique_frames = vr.get_batch(unique_indices).asnumpy()
                frames = unique_frames[inverse_indices]

        except Exception as e:
            print(f"[ERROR] Failed to load {video_path}: {e}")
            # Return a zero tensor on failure
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        # 2. Convert to tensor: (K, H, W, 3) → (K, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32) / 255.0

        # 3. Resize short side to img_size and center crop
        K, C, H, W = frames.shape
        scale = self.img_size / min(H, W)
        new_h = max(self.img_size, round(H * scale))
        new_w = max(self.img_size, round(W * scale))
        frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)

        top = (new_h - self.img_size) // 2
        left = (new_w - self.img_size) // 2
        frames = frames[:, :, top:top + self.img_size, left:left + self.img_size]

        # 4. Normalize
        frames = (frames - self.mean) / self.std

        # 5. Reshape to model input: (C, T, H, W)
        video_tensor = frames.permute(1, 0, 2, 3)  # (3, K, 224, 224)

        return video_tensor, label

    def get_class_name(self, idx):
        """Get class name from index."""
        return CRIME_CLASSES[idx]
