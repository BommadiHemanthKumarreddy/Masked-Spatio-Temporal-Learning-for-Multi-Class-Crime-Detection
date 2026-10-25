import torch
import torch.nn as nn
import models_mae


class LVMAEClassifier(nn.Module):
    def __init__(self, args, num_classes=13):
        super().__init__()

        self.args = args

        # Build pretrained MAE model
        self.mae = models_mae.__dict__[args.model](
            args=args,
            norm_pix_loss=args.norm_pix_loss
        )

        # Remove decoder (not required for classification)
        del self.mae.decoder_blocks
        del self.mae.decoder_embed
        del self.mae.decoder_norm
        del self.mae.decoder_pred
        del self.mae.decoder_pos_embed
        del self.mae.mask_token

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(args.input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def load_pretrained(self, checkpoint_path):
        """
        Load pretrained LV-MAE encoder weights.
        Decoder weights are ignored because they were removed.
        """

        print("=" * 70)
        print("Loading pretrained checkpoint...")
        print(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        msg = self.mae.load_state_dict(
            checkpoint["model"],
            strict=False
        )

        print("=" * 70)
        print("Checkpoint loaded successfully!")
        print(msg)
        print("=" * 70)

    def forward(self, x, length):
        """
        x      : (B, num_patches, 1024)
        length : (B,) actual number of valid segments
        """

        B, N, _ = x.shape

        # Padding mask for segment tokens
        # False = valid token, True = padding token
        positions = torch.arange(
            N,
            device=x.device
        ).unsqueeze(0)

        padding_mask = positions >= length.unsqueeze(1)

        # CLS token is always valid
        cls_mask = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=x.device
        )

        attn_mask = torch.cat(
            [cls_mask, padding_mask],
            dim=1
        )

        latent = self.mae.forward_encoder(
            x=x,
            attn_mask=attn_mask,
            ids_shuffle=None,
            length=length
        )

        # Use CLS representation for video classification
        cls_feature = latent[:, 0]

        logits = self.classifier(cls_feature)

        return logits
