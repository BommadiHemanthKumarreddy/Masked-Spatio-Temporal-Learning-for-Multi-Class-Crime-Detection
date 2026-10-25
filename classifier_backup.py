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

    def forward(self, x):
        """
        x shape : (B, 1024)
        """

        # Convert to sequence
        x = x.unsqueeze(1)

        B = x.size(0)

        length = torch.ones(
            B,
            dtype=torch.long,
            device=x.device
        )

        latent = self.mae.forward_encoder(
            x=x,
            attn_mask=None,
            ids_shuffle=None,
            length=length
        )

        # CLS token
        cls_feature = latent[:, 0]

        logits = self.classifier(cls_feature)

        return logits
