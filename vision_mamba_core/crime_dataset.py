# --------------------------------------------------------
# Crime Video Dataset
# Loads videos, applies key frame selection and augmentation
#
# Improvements over v1:
#   - Random horizontal flip (consistent across all frames in a clip)
#   - Color jitter (brightness, contrast, saturation) for train split
#   - Random temporal crop: sample different temporal segments each epoch
#   - Random spatial crop (during training) vs center crop (during val)
#   - Returns class weights for WeightedRandomSampler
# --------------------------------------------------------

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from decord import VideoReader, cpu

from crime_modules import select_key_frames, region_based_mask
from crime_model import CRIME_CLASSES


class CrimeVideoDataset(Dataset):
    """
    Dataset for multi-class crime detection with strong augmentations.

    Directory structure:
        root/
            dragging/  video1.mp4, video2.mp4, ...
            hitting/   ...
            ...

    Args:
        root_dir:       Path to dataset root with class subdirectories.
        num_frames:     Number of frames to sample per video clip.
        img_size:       Spatial resolution (height = width = img_size).
        use_key_frames: Use motion-based key frame selection.
        is_train:       If True, apply data augmentation. If False, use center crop only.
        use_region_mask: Apply region-based masking (experimental).
    """

    # ImageNet normalization statistics
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        root_dir,
        num_frames=16,
        img_size=224,
        use_key_frames=True,
        is_train=True,
        use_region_mask=False,
    ):
        self.root_dir       = root_dir
        self.num_frames     = num_frames
        self.img_size       = img_size
        self.use_key_frames = use_key_frames
        self.is_train       = is_train
        self.use_region_mask = use_region_mask

        # Build file list
        self.samples      = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CRIME_CLASSES)}

        for cls_name in CRIME_CLASSES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARNING] Class directory not found: {cls_dir}")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name],
                    ))

        print(f"[INFO] Loaded {len(self.samples)} videos from {root_dir} "
              f"(mode={'train' if is_train else 'val/test'})")

    # ------------------------------------------------------------------ #
    # Class weights for WeightedRandomSampler
    # ------------------------------------------------------------------ #

    def get_class_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.
        Inverse-frequency weighting so each class is sampled equally.
        """
        class_counts = [0] * len(CRIME_CLASSES)
        for _, label in self.samples:
            class_counts[label] += 1

        # Avoid division by zero for missing classes
        weights_per_class = [
            1.0 / count if count > 0 else 0.0
            for count in class_counts
        ]
        sample_weights = [weights_per_class[label] for _, label in self.samples]
        return torch.tensor(sample_weights, dtype=torch.double)

    # ------------------------------------------------------------------ #
    # Augmentation helpers (applied consistently across all frames)
    # ------------------------------------------------------------------ #

    def _random_flip(self, frames):
        """Horizontal flip: applied to all frames identically."""
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])  # flip W dimension
        return frames

    def _color_jitter(self, frames, brightness=0.3, contrast=0.3, saturation=0.2):
        """
        Randomly adjust brightness, contrast, and saturation.
        Operates on (K, 3, H, W) float32 tensors in [0, 1].
        Jitter factors are chosen once per clip and applied to all frames.
        """
        # Brightness
        b = random.uniform(max(0, 1 - brightness), 1 + brightness)
        frames = (frames * b).clamp(0, 1)

        # Contrast: blend with grayscale mean
        gray = frames.mean(dim=1, keepdim=True)
        c = random.uniform(max(0, 1 - contrast), 1 + contrast)
        frames = (c * frames + (1 - c) * gray).clamp(0, 1)

        # Saturation: blend with grayscale
        s = random.uniform(max(0, 1 - saturation), 1 + saturation)
        gray_full = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
        frames = (s * frames + (1 - s) * gray_full).clamp(0, 1)

        return frames

    def _random_crop(self, frames, size):
        """
        Random crop to (size × size). Applied identically across all frames.
        """
        K, C, H, W = frames.shape
        top  = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return frames[:, :, top:top + size, left:left + size]

    def _center_crop(self, frames, size):
        """Center crop to (size × size)."""
        K, C, H, W = frames.shape
        top  = (H - size) // 2
        left = (W - size) // 2
        return frames[:, :, top:top + size, left:left + size]

    # ------------------------------------------------------------------ #
    # Main dataset methods
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < self.num_frames:
                # Video too short: pad by repeating last frame
                indices = list(range(total_frames)) + [total_frames - 1] * (
                    self.num_frames - total_frames)
                indices = np.array(indices)
            elif self.is_train:
                # ---- TRAIN: random temporal crop ----
                # Sample num_frames from a random temporal window
                max_start = total_frames - self.num_frames
                start = random.randint(0, max_start)
                indices = np.arange(start, start + self.num_frames)
            else:
                # ---- VAL/TEST: uniform sampling ----
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

            # Key frame selection (motion-based) on train set with extra headroom
            if self.use_key_frames and self.is_train and total_frames > self.num_frames * 2:
                sample_count = min(total_frames, self.num_frames * 4)
                sample_idx   = np.linspace(0, total_frames - 1, sample_count).astype(int)
                unique_idx, inv = np.unique(sample_idx, return_inverse=True)
                all_frames = vr.get_batch(unique_idx).asnumpy()[inv]
                frames = select_key_frames(all_frames, k=self.num_frames)
            else:
                unique_idx, inv = np.unique(indices, return_inverse=True)
                frames = vr.get_batch(unique_idx).asnumpy()[inv]

        except Exception as e:
            print(f"[ERROR] Failed to load {video_path}: {e}")
            frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        # (K, H, W, 3) -> (K, 3, H, W), float32 in [0, 1]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # ---- Resize short side to img_size + crop ----
        K, C, H, W = frames.shape
        scale = self.img_size / min(H, W)
        new_h = max(self.img_size, round(H * scale))
        new_w = max(self.img_size, round(W * scale))
        frames = F.interpolate(frames, size=(new_h, new_w),
                               mode='bilinear', align_corners=False)

        if self.is_train:
            # Random crop: give a slightly larger resize target
            scale2 = random.uniform(1.0, 1.15)
            rh = max(self.img_size, round(new_h * scale2))
            rw = max(self.img_size, round(new_w * scale2))
            if rh != new_h or rw != new_w:
                frames = F.interpolate(frames, size=(rh, rw),
                                       mode='bilinear', align_corners=False)
            frames = self._random_crop(frames, self.img_size)
        else:
            frames = self._center_crop(frames, self.img_size)

        # ---- Augmentations (training only) ----
        if self.is_train:
            frames = self._random_flip(frames)
            if random.random() < 0.8:           # 80% chance of color jitter
                frames = self._color_jitter(frames)

        # ---- Normalize with ImageNet stats ----
        frames = (frames - self.MEAN) / self.STD

        # (K, 3, H, W) -> (3, K, H, W) for model input
        video_tensor = frames.permute(1, 0, 2, 3).contiguous()
        return video_tensor, label

    def get_class_name(self, idx):
        return CRIME_CLASSES[idx]
