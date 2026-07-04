# --------------------------------------------------------
# Crime Detection Model — Full Integrated Pipeline
# VideoMAE Encoder + Mamba + Temporal Attention + Classifier
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models import create_model

# Register VideoMAE models
import models  # noqa: F401

from crime_modules import TemporalMamba, TemporalAttention


# Crime class labels
CRIME_CLASSES = [
    'dragging', 'hitting', 'snatching', 'kicking', 'pushing',
    'robbery', 'hair_pulling', 'punching', 'slapping', 'abuse',
    'neck_grabbing', 'shooting', 'stabbing',
]
NUM_CLASSES = len(CRIME_CLASSES)  # 13


class CrimeDetectionModel(nn.Module):
    """
    Multi-class crime detection model.

    Pipeline:
        Video Input → VideoMAE Encoder → Reshape (B,T,D)
        → Mamba Temporal Modeling → Temporal Attention → Classifier

    The encoder is frozen by default (lightweight design).
    Only the Mamba layer, Attention layer, and Classifier are trained.
    """

    def __init__(
        self,
        encoder_name='vit_small_patch16_224',
        num_classes=NUM_CLASSES,
        num_frames=16,
        tubelet_size=2,
        pretrained_ckpt=None,
        freeze_encoder=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_temporal_tokens = num_frames // tubelet_size  # e.g., 16//2 = 8

        # ---- 1. VideoMAE Encoder (Task 3) ----
        # Use num_classes=710 to match pretrained checkpoint structure.
        # The encoder's head is NOT used — we have our own classifier.
        self.encoder = create_model(
            encoder_name,
            pretrained=False,
            num_classes=710,
            all_frames=num_frames,
            tubelet_size=tubelet_size,
            use_mean_pooling=False,  # we need all tokens, not pooled
        )
        embed_dim = self.encoder.embed_dim

        # Load pretrained weights if provided
        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

        # Freeze encoder (Task 7: Lightweight Design)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print(f"[INFO] Encoder frozen. embed_dim={embed_dim}")

        # ---- 2. Mamba Temporal Modeling (Task 4) ----
        self.temporal_mamba = TemporalMamba(
            dim=embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # ---- 3. Temporal Attention (Task 5) ----
        self.temporal_attn = TemporalAttention(dim=embed_dim)

        # ---- 4. Classification Head (Task 0) ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

        # Initialize new modules
        self._init_new_layers()

    def _load_pretrained(self, ckpt_path):
        """Load pretrained VideoMAE weights into the encoder."""
        print(f"[INFO] Loading pretrained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Filter out head weights (we have our own classifier)
        new_state_dict = {
            k: v for k, v in new_state_dict.items()
            if not k.startswith('head.')
        }

        msg = self.encoder.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] Loaded pretrained weights. Missing: {msg.missing_keys[:5]}...")

    def _init_new_layers(self):
        """Initialize weights for new (non-pretrained) layers."""
        for module in [self.temporal_mamba, self.temporal_attn, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def extract_temporal_features(self, x):
        """
        Extract features from the encoder and reshape to (B, T, D).

        The encoder produces (B, N, D) where N = num_temporal * num_spatial.
        We average the spatial patches within each temporal step to get (B, T, D).
        """
        B = x.size(0)

        # Run encoder backbone (no head, no pooling)
        x = self.encoder.patch_embed(x)

        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed.expand(B, -1, -1).type_as(x).to(
                x.device).clone().detach()
        x = self.encoder.pos_drop(x)

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)  # (B, N, D)

        # Reshape: N = T_patches * S_patches
        # T_patches = num_frames // tubelet_size
        # S_patches = (H // patch_size) * (W // patch_size)
        B, N, D = x.shape
        T = self.num_temporal_tokens
        S = N // T  # number of spatial patches per temporal step

        # (B, N, D) → (B, T, S, D) → average over spatial → (B, T, D)
        x = x.reshape(B, T, S, D).mean(dim=2)

        return x  # (B, T, D)

    def forward(self, x, return_attention=False):
        """
        Full forward pass.

        Args:
            x: (B, C, T, H, W) video tensor
            return_attention: if True, also return attention weights

        Returns:
            logits: (B, num_classes)
            attention_weights: (B, T) — only if return_attention=True
        """
        # Task 3: Extract temporal features → (B, T, D)
        features = self.extract_temporal_features(x)

        # Task 4: Mamba temporal modeling → (B, T, D)
        features = self.temporal_mamba(features)

        # Task 5: Temporal attention → (B, D)
        attended_features, attn_weights = self.temporal_attn(features)

        # Task 0: Classification → (B, num_classes)
        logits = self.classifier(attended_features)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_trainable_params(self):
        """Return only the trainable parameters (Mamba + Attention + Classifier)."""
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode=True):
        """Override train to keep encoder frozen in eval mode."""
        super().train(mode)
        # Always keep encoder in eval mode if frozen
        if hasattr(self, 'encoder'):
            is_frozen = not any(p.requires_grad for p in self.encoder.parameters())
            if is_frozen:
                self.encoder.eval()
        return self
