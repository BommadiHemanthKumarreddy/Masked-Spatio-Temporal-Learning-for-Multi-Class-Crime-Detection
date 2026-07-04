# --------------------------------------------------------
# Crime Detection Model v3 — Per-Frame 2D ViT + Stacked Mamba
#
# Key fix over v2:
#   The VideoMAE 3D patch-embed CANNOT load 2D ImageNet weights,
#   so the encoder was always random. This version uses standard
#   2D timm ViT which loads ImageNet weights correctly.
#
# Pipeline:
#   (B, 3, T, H, W)
#     -> reshape to (B*T, 3, H, W)
#     -> 2D ViT encoder (ImageNet pretrained, last 6 blocks unfrozen)
#     -> reshape to (B, T, D)
#     -> 2x TemporalMamba (bidirectional SSM)
#     -> TemporalAttention
#     -> MLP head -> (B, 13)
# --------------------------------------------------------

import sys
import os

# NOTE: We deliberately do NOT add VideoMAEv2 to sys.path here.
# VideoMAEv2 registers its own 'vit_small_patch16_224' which expects
# 5D input (B,C,T,H,W). We need the standard 2D timm ViT instead.
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import timm

from crime_modules import TemporalMamba, TemporalAttention

# ── Class labels ──────────────────────────────────────────────────────────
CRIME_CLASSES = [
    'dragging', 'hitting', 'snatching', 'kicking', 'pushing',
    'robbery', 'hair_pulling', 'punching', 'slapping', 'abuse',
    'neck_grabbing', 'shooting', 'stabbing',
]
NUM_CLASSES = len(CRIME_CLASSES)  # 13


class CrimeDetectionModel(nn.Module):
    """
    Crime Detection Model v3.

    Uses a standard 2D timm ViT encoder (ImageNet pretrained) applied
    per-frame, then bidirectional Mamba blocks for temporal reasoning.

    Why 2D ViT instead of VideoMAE 3D:
        VideoMAE's 3D patch embedding uses a different kernel shape
        (3xPxP instead of 1xPxP) so 2D ImageNet weights cannot be
        loaded — the encoder always starts from random weights.
        By processing each frame with a standard 2D ViT, we get
        genuine ImageNet pretrained spatial features for free.
    """

    def __init__(
        self,
        encoder_name='vit_small_patch16_224',
        num_classes=NUM_CLASSES,
        num_frames=16,
        pretrained_ckpt=None,         # path to VideoMAE ckpt (optional)
        freeze_encoder=True,
        unfreeze_blocks=6,            # last N blocks unfrozen
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_layers=2,
        head_dropout=0.5,
        # Legacy kwargs (ignored, kept for CLI compat)
        tubelet_size=2,
        mamba_expand_=None,
    ):
        super().__init__()
        self.num_frames    = num_frames
        self.freeze_encoder = freeze_encoder
        self.unfreeze_blocks = unfreeze_blocks

        # ── 1. 2D ViT Encoder (ImageNet pretrained) ───────────────────────
        # num_classes=0 -> forward returns CLS token feature, no head
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(pretrained_ckpt is None),  # True = ImageNet weights
            num_classes=0,                          # returns CLS token feature
        )
        self.embed_dim = self.encoder.num_features

        if pretrained_ckpt:
            self._load_pretrained(pretrained_ckpt)
            print(f"[INFO] Loaded VideoMAE checkpoint: {pretrained_ckpt}")
        else:
            print(f"[INFO] Using ImageNet-pretrained 2D ViT "
                  f"(embed_dim={self.embed_dim})")

        # ── 2. Selective encoder freezing ────────────────────────────────
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

            # Unfreeze last `unfreeze_blocks` transformer blocks
            blocks = list(self.encoder.blocks)
            total  = len(blocks)
            for i, blk in enumerate(blocks):
                if i >= total - unfreeze_blocks:
                    for p in blk.parameters():
                        p.requires_grad = True
            # Always unfreeze final LayerNorm
            for p in self.encoder.norm.parameters():
                p.requires_grad = True

            frozen   = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
            unfrozen = sum(1 for p in self.encoder.parameters() if p.requires_grad)
            print(f"[INFO] Encoder: {frozen} frozen, {unfrozen} unfrozen "
                  f"(last {unfreeze_blocks}/{total} blocks + norm)")

        # ── 3. Temporal projection (align frame features) ─────────────────
        # Small learned projection before Mamba (stabilises training)
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )

        # ── 4. Stacked TemporalMamba ──────────────────────────────────────
        self.temporal_mamba = nn.Sequential(*[
            TemporalMamba(
                dim=self.embed_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            for _ in range(mamba_layers)
        ])

        # ── 5. Temporal Attention pooling ─────────────────────────────────
        self.temporal_attn = TemporalAttention(dim=self.embed_dim)

        # ── 6. Classification head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim // 2, num_classes),
        )

        self._init_new_layers()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd   = ckpt.get('model', ckpt.get('module', ckpt))
        sd   = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        sd   = {k: v for k, v in sd.items() if not k.startswith('head.')}
        msg  = self.encoder.load_state_dict(sd, strict=False)
        if msg.missing_keys:
            print(f"[INFO] Missing keys (head excluded): {msg.missing_keys[:3]}")

    def _init_new_layers(self):
        for mod in [self.temporal_proj, self.temporal_mamba,
                    self.temporal_attn, self.classifier]:
            for m in mod.modules():
                if isinstance(m, nn.Identity):
                    continue
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Forward ──────────────────────────────────────────────────────────

    def _encoder_forward(self, x):
        """
        Safe forward through the 2D ViT encoder.
        Works with timm 0.4.12 where num_classes=0 returns CLS token directly,
        and with versions where the head may be Identity.
        """
        # Use forward_features if available (returns patch+CLS tokens before head)
        if hasattr(self.encoder, 'forward_features'):
            feats = self.encoder.forward_features(x)  # (B, N+1, D) or (B, D)
            # If 3D (tokens), take CLS token at position 0
            if feats.dim() == 3:
                feats = feats[:, 0]          # CLS token: (B, D)
        else:
            feats = self.encoder(x)          # (B, D) directly

        return feats   # (B, D)

    def encode_frames(self, x):
        """
        Encode all frames independently with the 2D ViT.

        Args:
            x: (B, 3, T, H, W)
        Returns:
            features: (B, T, D)
        """
        B, C, T, H, W = x.shape

        # Merge batch and temporal dims: (B*T, 3, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 2D ViT forward: returns CLS token (B*T, D)
        feats = self._encoder_forward(x)      # (B*T, D)

        # Restore temporal dimension: (B, T, D)
        feats = feats.reshape(B, T, -1)
        return feats

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 3, T, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # 1. Per-frame 2D ViT encoding -> (B, T, D)
        feats = self.encode_frames(x)

        # 2. Temporal projection
        feats = self.temporal_proj(feats)

        # 3. Stacked Mamba temporal modeling -> (B, T, D)
        feats = self.temporal_mamba(feats)

        # 4. Temporal attention pooling -> (B, D)
        attended, attn_weights = self.temporal_attn(feats)

        # 5. Classification -> (B, 13)
        logits = self.classifier(attended)

        if return_attention:
            return logits, attn_weights
        return logits

    # ── Optimizer helpers ─────────────────────────────────────────────────

    def get_param_groups(self, encoder_lr_scale=0.1):
        """Return param groups with separate LR for encoder vs head."""
        enc_params  = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = (list(self.temporal_proj.parameters()) +
                       list(self.temporal_mamba.parameters()) +
                       list(self.temporal_attn.parameters()) +
                       list(self.classifier.parameters()))
        return [
            {'params': enc_params,  'lr_scale': encoder_lr_scale, 'name': 'encoder'},
            {'params': head_params, 'lr_scale': 1.0,              'name': 'head'},
        ]

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            total  = len(list(self.encoder.blocks))
            frozen_upto = total - self.unfreeze_blocks
            for i, blk in enumerate(self.encoder.blocks):
                if i < frozen_upto:
                    blk.eval()
        return self
