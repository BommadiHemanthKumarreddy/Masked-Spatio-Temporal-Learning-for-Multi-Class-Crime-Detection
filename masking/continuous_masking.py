import torch
import torch.nn as nn
import numpy as np

class ContinuousImportanceMaskGenerator(nn.Module):
    def __init__(self, mask_ratio=0.90, epsilon=1e-3):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.epsilon = epsilon

    def forward(self, importance_map):
        # importance_map: [B, T_p, H_p, W_p] or [B, N]
        B = importance_map.shape[0]
        imp_flat = importance_map.reshape(B, -1) # [B, N]
        B, N = imp_flat.shape

        num_mask = int(self.mask_ratio * N)
        num_vis = N - num_mask

        # Inverse Importance Sampling Weight
        # High importance -> low inverse weight -> lower sampling probability for masking
        inv_weights = (1.0 - imp_flat) + self.epsilon # [B, N]

        bool_masked_pos = torch.zeros((B, N), dtype=torch.bool, device=importance_map.device)

        for b in range(B):
            w = inv_weights[b]
            # Sample exactly num_mask patch indices without replacement
            sampled_mask_indices = torch.multinomial(w, num_samples=num_mask, replacement=False)
            bool_masked_pos[b, sampled_mask_indices] = True

        return bool_masked_pos # [B, N] (True = masked, False = visible)
