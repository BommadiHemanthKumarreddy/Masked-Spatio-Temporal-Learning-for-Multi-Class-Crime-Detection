import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyImportanceEstimator(nn.Module):
    def __init__(self, patch_size=(2, 16, 16)):
        super().__init__()
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, frames):
        # frames: [B, C, T, H, W]
        B, C, T, H, W = frames.shape
        device = frames.device

        # Spatial Gradient / Edge Saliency Map per frame
        # Compute spatial gradients dX and dY
        grad_x = torch.abs(frames[:, :, :, :, 1:] - frames[:, :, :, :, :-1])
        grad_y = torch.abs(frames[:, :, :, 1:, :] - frames[:, :, :, :-1, :])
        
        # Pad to keep H, W dimensions
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        sal_map = (grad_x + grad_y).mean(dim=1) # [B, T, H, W]

        # Normalize to [0, 1] per video
        sal_flat = sal_map.reshape(B, -1)
        min_v = sal_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_v = sal_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        sal_norm = (sal_map - min_v) / (max_v - min_v + 1e-6)

        # Average pool over spatio-temporal tubelets
        t_t, p_h, p_w = self.patch_size
        sal_pooled = F.avg_pool3d(sal_norm.unsqueeze(1), kernel_size=(t_t, p_h, p_w), stride=(t_t, p_h, p_w))
        saliency_importance = sal_pooled.squeeze(1) # [B, T_p, H_p, W_p]
        return saliency_importance
