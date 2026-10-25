import torch
import torch.nn as nn
import models_mae


class SummaryTokenModule(nn.Module):
    """
    Generates one learned summary token for each temporal segment.

    Input:
        x: (B, T, D)

    Output:
        summary: (B, T, D)
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        # Learnable summary query
        self.summary_query = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):

        B, T, D = x.shape

        # --------------------------------------------------------
        # One learnable query for every temporal segment
        # --------------------------------------------------------

        query = self.summary_query.expand(
            B * T,
            1,
            D
        )

        # Each segment is summarized independently
        segment = x.reshape(
            B * T,
            1,
            D
        )

        summary, _ = self.attention(
            query,
            segment,
            segment,
            need_weights=False
        )

        summary = summary.squeeze(1)

        # Residual connection
        summary = summary + x.reshape(
            B * T,
            D
        )

        summary = self.norm1(summary)

        # Feed-forward refinement
        summary = summary + self.mlp(summary)

        summary = self.norm2(summary)

        summary = summary.reshape(
            B,
            T,
            D
        )

        return summary


class TemporalAttention(nn.Module):
    """
    Temporal attention over segment-level summary tokens.

    Input:
        x: (B, T, D)

    Output:
        video_feature: (B, D)
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

        # Learnable temporal CLS/query
        self.temporal_cls = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        # Stores attention during inference
        self.last_attention_weights = None

    def forward(self, x, padding_mask=None):

        B = x.shape[0]

        # --------------------------------------------------------
        # Temporal CLS token
        # --------------------------------------------------------

        cls = self.temporal_cls.expand(
            B,
            -1,
            -1
        )

        # --------------------------------------------------------
        # Temporal attention
        #
        # Query  = temporal CLS
        # Key    = summary tokens
        # Value  = summary tokens
        # --------------------------------------------------------

        attended, attention_weights = self.attention(
            cls,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        # --------------------------------------------------------
        # Save attention for visualization
        #
        # Shape:
        # (B, num_heads, 1, T)
        # --------------------------------------------------------

        self.last_attention_weights = (
            attention_weights.detach()
        )

        # --------------------------------------------------------
        # Residual + normalization
        # --------------------------------------------------------

        video_feature = self.norm1(
            cls + attended
        )

        # --------------------------------------------------------
        # Feed-forward block
        # --------------------------------------------------------

        video_feature = self.norm2(
            video_feature + self.mlp(video_feature)
        )

        return video_feature[:, 0]


class LVMAEClassifier(nn.Module):

    def __init__(
        self,
        args,
        num_classes=13
    ):

        super().__init__()

        self.args = args

        # ========================================================
        # LV-MAE ENCODER
        # ========================================================

        self.mae = models_mae.__dict__[args.model](
            args=args,
            norm_pix_loss=args.norm_pix_loss
        )

        # --------------------------------------------------------
        # Remove decoder
        # --------------------------------------------------------

        del self.mae.decoder_blocks
        del self.mae.decoder_embed
        del self.mae.decoder_norm
        del self.mae.decoder_pred
        del self.mae.decoder_pos_embed
        del self.mae.mask_token

        # ========================================================
        # SUMMARY TOKEN MODULE
        # ========================================================

        self.summary_tokens = SummaryTokenModule(
            dim=args.input_size,
            num_heads=8,
            dropout=0.1
        )

        # ========================================================
        # TEMPORAL ATTENTION
        # ========================================================

        self.temporal_attention = TemporalAttention(
            dim=args.input_size,
            num_heads=8,
            dropout=0.1
        )

        # ========================================================
        # CLASSIFICATION HEAD
        # ========================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                args.input_size,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.5
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    # ============================================================
    # LOAD PRETRAINED LV-MAE
    # ============================================================

    def load_pretrained(
        self,
        checkpoint_path
    ):

        print("=" * 70)
        print("Loading pretrained LV-MAE checkpoint")
        print("=" * 70)

        print(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        msg = self.mae.load_state_dict(
            checkpoint["model"],
            strict=False
        )

        print("=" * 70)
        print("LV-MAE checkpoint loaded")
        print("=" * 70)

        print(msg)

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x,
        length
    ):

        B, N, _ = x.shape

        # ========================================================
        # PADDING MASK
        # ========================================================

        positions = torch.arange(
            N,
            device=x.device
        ).unsqueeze(0)

        padding_mask = (
            positions >=
            length.unsqueeze(1)
        )

        # ========================================================
        # CLS MASK
        # ========================================================

        cls_mask = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=x.device
        )

        attn_mask = torch.cat(
            [
                cls_mask,
                padding_mask
            ],
            dim=1
        )

        # ========================================================
        # LV-MAE ENCODER
        # ========================================================

        latent = self.mae.forward_encoder(
            x=x,
            attn_mask=attn_mask,
            ids_shuffle=None,
            length=length
        )

        # ========================================================
        # REMOVE LV-MAE CLS
        #
        # latent[:, 0]  = LV-MAE CLS
        # latent[:, 1:] = temporal segment features
        # ========================================================

        segment_features = latent[:, 1:]

        # ========================================================
        # GENERATE SUMMARY TOKENS
        # ========================================================

        summary_tokens = self.summary_tokens(
            segment_features
        )

        # ========================================================
        # TEMPORAL ATTENTION
        # ========================================================

        video_feature = self.temporal_attention(
            summary_tokens,
            padding_mask=padding_mask
        )

        # ========================================================
        # CLASSIFIER
        # ========================================================

        logits = self.classifier(
            video_feature
        )

        return logits
