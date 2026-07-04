# --------------------------------------------------------
# Crime Detection Modules
# Key Frame Selection, Region-Based Masking,
# Temporal Mamba, and Temporal Attention
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =======================================================
# TASK 1: Key Frame Selection (Preprocessing)
# =======================================================

def select_key_frames(video_frames, k=16):
    """
    Select the top-K most important frames based on motion.

    Args:
        video_frames: numpy array of shape (T, H, W, C) or (T, C, H, W)
                      or torch.Tensor of the same shapes.
        k: number of frames to select.

    Returns:
        selected_frames: numpy array / tensor of shape (K, H, W, C) or (K, C, H, W)
    """
    is_tensor = isinstance(video_frames, torch.Tensor)
    if is_tensor:
        frames_np = video_frames.cpu().numpy()
    else:
        frames_np = video_frames

    num_frames = frames_np.shape[0]

    # If we have fewer frames than K, return all frames
    if num_frames <= k:
        return video_frames

    # Compute motion scores using frame differencing
    motion_scores = np.zeros(num_frames)
    for t in range(1, num_frames):
        diff = np.abs(frames_np[t].astype(np.float32) - frames_np[t - 1].astype(np.float32))
        motion_scores[t] = diff.sum()

    # First frame gets average motion score so it has a chance of being selected
    motion_scores[0] = np.mean(motion_scores[1:]) if num_frames > 1 else 0

    # Select top-K indices and sort to preserve temporal order
    top_k_indices = np.argsort(motion_scores)[-k:]
    top_k_indices = np.sort(top_k_indices)

    if is_tensor:
        return video_frames[top_k_indices]
    else:
        return frames_np[top_k_indices]


# =======================================================
# TASK 2: Region-Based Masking
# =======================================================

def region_based_mask(frames, patch_size=16, tubelet_size=2, threshold=0.3):
    """
    Generate a region-based mask that preserves high-motion regions
    and masks low-motion regions.

    Args:
        frames: tensor of shape (T, C, H, W) — preprocessed video frames.
        patch_size: spatial patch size (default 16 for ViT).
        tubelet_size: temporal tubelet size (default 2).
        threshold: percentile threshold (0-1). Patches below this
                   percentile of motion are masked (True=masked).

    Returns:
        mask: boolean tensor of shape (num_temporal_patches, num_spatial_patches)
              True = masked (low motion), False = visible (high motion)
    """
    T, C, H, W = frames.shape

    num_t = T // tubelet_size
    num_h = H // patch_size
    num_w = W // patch_size

    # Compute per-pixel temporal motion map
    motion_map = torch.zeros(T, H, W, device=frames.device)
    for t in range(1, T):
        diff = (frames[t] - frames[t - 1]).abs().mean(dim=0)  # (H, W)
        motion_map[t] = diff
    motion_map[0] = motion_map[1] if T > 1 else motion_map[0]

    # Aggregate motion into patch-level scores
    # Reshape to (num_t, tubelet_size, num_h, patch_size, num_w, patch_size)
    motion_patches = motion_map[:num_t * tubelet_size, :num_h * patch_size, :num_w * patch_size]
    motion_patches = motion_patches.reshape(
        num_t, tubelet_size, num_h, patch_size, num_w, patch_size
    )
    # Mean over tubelet and patch spatial dims → (num_t, num_h, num_w)
    patch_scores = motion_patches.mean(dim=(1, 3, 5))

    # Flatten to (num_t, num_h * num_w)
    patch_scores_flat = patch_scores.reshape(num_t, num_h * num_w)

    # Threshold: mask patches with below-threshold motion
    flat_all = patch_scores_flat.reshape(-1)
    cutoff = torch.quantile(flat_all, threshold)
    mask = patch_scores_flat < cutoff  # True = masked (low motion)

    return mask


# =======================================================
# TASK 4: Mamba Temporal Modeling
# =======================================================

try:
    from mamba_ssm import Mamba

    class TemporalMamba(nn.Module):
        """
        Mamba-based temporal modeling using mamba-ssm library.
        Input: (B, T, D)   Output: (B, T, D)
        """
        def __init__(self, dim, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.norm  = nn.LayerNorm(dim)
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            return x + residual

    MAMBA_AVAILABLE = True
    print("[INFO] mamba-ssm library detected. Using optimised Mamba kernel.")

except ImportError:
    try:
        # In vision_mamba_core, mamba.py lives in the same directory
        from mamba import MambaBlock
        _mamba_source = "vision_mamba_core/mamba.py (local)"

    except ImportError:
        try:
            # Fallback: vision_mamba subpackage inside VideoMAEv2
            import sys, os
            _videomae = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VideoMAEv2')
            if os.path.exists(_videomae) and _videomae not in sys.path:
                sys.path.append(_videomae)
            from vision_mamba.mamba import MambaBlock
            _mamba_source = "VideoMAEv2/vision_mamba subpackage"
        except ImportError:
            MambaBlock = None
            _mamba_source = None

    if MambaBlock is not None:
        class TemporalMamba(nn.Module):
            """
            Bidirectional Vision Mamba temporal modeling.
            Uses the pure-PyTorch MambaBlock (bimamba_type='v1').

            Unlike the forward-only GRU fallback, this scans the temporal
            sequence in both directions, letting the model see both past
            (what happened before) and future (what happens next) context.

            Input: (B, T, D)   Output: (B, T, D)
            """
            def __init__(self, dim, d_state=16, d_conv=4, expand=2):
                super().__init__()
                self.norm  = nn.LayerNorm(dim)
                self.mamba = MambaBlock(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba_type="v1",   # bidirectional scanning
                )

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x = self.mamba(x)
                return x + residual

        MAMBA_AVAILABLE = True
        print(f"[INFO] mamba-ssm not found. Loaded MambaBlock from {_mamba_source}.")

    else:
        class TemporalMamba(nn.Module):
            """
            GRU fallback — used only when neither mamba-ssm nor MambaBlock are available.
            Input: (B, T, D)   Output: (B, T, D)
            """
            def __init__(self, dim, d_state=16, d_conv=4, expand=2):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.gru  = nn.GRU(input_size=dim, hidden_size=dim,
                                   num_layers=1, batch_first=True, bidirectional=False)
                self.proj = nn.Linear(dim, dim)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x, _ = self.gru(x)
                x = self.proj(x)
                return x + residual

        MAMBA_AVAILABLE = False
        print("[WARNING] Neither mamba-ssm nor MambaBlock found. Using GRU fallback.")


# =======================================================
# TASK 5: Temporal Attention (Main Novelty)
# =======================================================

class TemporalAttention(nn.Module):
    """
    Learns per-frame importance weights and produces a
    temporally-weighted feature vector.

    Input:  (B, T, D)
    Output: (B, D), attention_weights (B, T)
    """

    def __init__(self, dim):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) temporal feature sequence

        Returns:
            weighted_feature: (B, D) attention-weighted feature
            attention_weights: (B, T) per-frame importance weights
        """
        # Compute attention scores: (B, T, 1)
        scores = self.attention_fc(x)

        # Softmax over time dimension: (B, T, 1)
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum: (B, T, D) * (B, T, 1) → sum → (B, D)
        weighted_feature = (x * attention_weights).sum(dim=1)

        # Return weights squeezed to (B, T) for visualization
        return weighted_feature, attention_weights.squeeze(-1)
