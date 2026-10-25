import torch
import torch.nn as nn

class ImportanceFusionModule(nn.Module):
    def __init__(self, alpha=0.70):
        super().__init__()
        self.alpha = alpha

    def forward(self, motion_imp, saliency_imp):
        # motion_imp: [B, T_p, H_p, W_p]
        # saliency_imp: [B, T_p, H_p, W_p]
        fused = self.alpha * motion_imp + (1.0 - self.alpha) * saliency_imp

        # Normalize to [0, 1] per batch item
        B = fused.shape[0]
        f_flat = fused.reshape(B, -1)
        min_v = f_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_v = f_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        fused_norm = (fused - min_v) / (max_v - min_v + 1e-6)
        return fused_norm # [B, T_p, H_p, W_p]
