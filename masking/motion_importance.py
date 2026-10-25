import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MotionImportanceEstimator(nn.Module):
    def __init__(self, method="raft", patch_size=(2, 16, 16)):
        super().__init__()
        self.method = method
        self.patch_size = patch_size # (tubelet_size, patch_h, patch_w)
        if self.method == "raft":
            try:
                self.raft = torchvision.models.optical_flow.raft_large(pretrained=True)
                self.raft.eval()
                for p in self.raft.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[Warning] Could not load torchvision RAFT ({e}), falling back to frame_difference")
                self.method = "frame_difference"

    @torch.no_grad()
    def forward(self, frames):
        # frames: [B, C, T, H, W] in range [0, 1] or normalized
        B, C, T, H, W = frames.shape
        device = frames.device

        if self.method == "raft" and hasattr(self, "raft"):
            # Prepare pairs of consecutive frames for RAFT: [B*(T-1), 3, H, W]
            f1 = frames[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            f2 = frames[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

            # Resize if needed to multiples of 8 for RAFT
            H_r = ((H + 7) // 8) * 8
            W_r = ((W + 7) // 8) * 8
            f1_r = F.interpolate(f1, size=(H_r, W_r), mode="bilinear", align_corners=False)
            f2_r = F.interpolate(f2, size=(H_r, W_r), mode="bilinear", align_corners=False)

            # Ensure 3 channels
            if f1_r.shape[1] == 1:
                f1_r = f1_r.repeat(1, 3, 1, 1)
                f2_r = f2_r.repeat(1, 3, 1, 1)

            # Range [0, 255] for RAFT input if required
            if f1_r.max() <= 1.0:
                f1_r = f1_r * 255.0
                f2_r = f2_r * 255.0

            list_of_flows = self.raft(f1_r, f2_r)
            flow_final = list_of_flows[-1] # [B*(T-1), 2, H_r, W_r]

            if H_r != H or W_r != W:
                flow_final = F.interpolate(flow_final, size=(H, W), mode="bilinear", align_corners=False)

            dx = flow_final[:, 0, :, :]
            dy = flow_final[:, 1, :, :]
            mag = torch.sqrt(dx**2 + dy**2 + 1e-6) # [B*(T-1), H, W]
            mag = mag.reshape(B, T-1, H, W)
            # Pad last frame motion by repeating frame T-1
            mag_full = torch.cat([mag, mag[:, -1:]], dim=1) # [B, T, H, W]
        else:
            # Fallback: Frame difference magnitude
            diff = torch.abs(frames[:, :, 1:] - frames[:, :, :-1]).mean(dim=1) # [B, T-1, H, W]
            mag_full = torch.cat([diff, diff[:, -1:]], dim=1) # [B, T, H, W]

        # Normalize to [0, 1] per video
        B_idx = mag_full.reshape(B, -1)
        min_v = B_idx.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_v = B_idx.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        mag_norm = (mag_full - min_v) / (max_v - min_v + 1e-6) # [B, T, H, W]

        # Average pool over spatio-temporal tubelets: (tubelet_t, patch_h, patch_w)
        t_t, p_h, p_w = self.patch_size
        mag_pooled = F.avg_pool3d(mag_norm.unsqueeze(1), kernel_size=(t_t, p_h, p_w), stride=(t_t, p_h, p_w)) # [B, 1, T_p, H_p, W_p]
        motion_importance = mag_pooled.squeeze(1) # [B, T_p, H_p, W_p]
        return motion_importance
