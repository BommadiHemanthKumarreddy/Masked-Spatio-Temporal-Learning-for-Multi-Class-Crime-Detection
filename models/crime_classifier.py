import torch
import torch.nn as nn

from models.mamba_temporal import MambaTemporalModel
from models.temporal_attention import TemporalAttention
from models.modeling_finetune import VisionTransformer

class VideoMAE_CrimeClassifier(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=13,
        embed_dim=768,
        depth=12,
        num_heads=12,
        tubelet_size=2,
        all_frames=16,
        mamba_layers=2,
        freeze_encoder=False
    ):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            tubelet_size=tubelet_size,
            all_frames=all_frames,
            num_classes=0
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.mamba = MambaTemporalModel(d_model=embed_dim, num_layers=mamba_layers)
        self.temporal_attn = TemporalAttention(d_model=embed_dim, hidden_dim=256)

        self.classifier_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.encoder.get_patch_features(x) # [B, N_patches, D]

        T_p = T // 2
        N_spatial = feat.shape[1] // T_p
        
        feat_reshaped = feat.reshape(B, T_p, N_spatial, -1)
        temp_seq = feat_reshaped.mean(dim=2) # [B, T_p, D]

        mamba_seq = self.mamba(temp_seq) # [B, T_p, D]
        summary_vec, attn_weights = self.temporal_attn(mamba_seq) # [B, D], [B, T_p]

        logits = self.classifier_head(summary_vec) # [B, num_classes]
        return logits, attn_weights
