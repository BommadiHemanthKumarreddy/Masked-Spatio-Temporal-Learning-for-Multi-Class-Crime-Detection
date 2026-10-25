import os, sys, yaml, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.crime_dataset import CrimeVideoDataset
from masking.motion_importance import MotionImportanceEstimator
from masking.saliency_importance import SaliencyImportanceEstimator
from masking.importance_fusion import ImportanceFusionModule
from masking.continuous_masking import ContinuousImportanceMaskGenerator
from models.modeling_pretrain import PretrainVisionTransformer


def patchify(videos, patch_size=16, tubelet_size=2):
    """Extract raw pixel patches from video tensor.
    videos: [B, C, T, H, W]
    returns: [B, N, tubelet_size * patch_size * patch_size * C]
    """
    B, C, T, H, W = videos.shape
    T_p = T // tubelet_size
    H_p = H // patch_size
    W_p = W // patch_size
    x = videos.reshape(B, C, T_p, tubelet_size, H_p, patch_size, W_p, patch_size)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)   # [B, T_p, H_p, W_p, tub, pH, pW, C]
    x = x.reshape(B, T_p * H_p * W_p, tubelet_size * patch_size * patch_size * C)
    return x  # [B, N, 1536]


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_frames   = cfg["video"]["num_frames"]
    image_size   = cfg["video"]["image_size"]
    patch_size   = cfg["video"]["patch_size"]
    tubelet_size = cfg["video"]["tubelet_size"]
    mask_ratio   = cfg["masking"]["mask_ratio"]
    alpha        = cfg["masking"]["alpha"]
    batch_size   = cfg["training"]["batch_size"]
    epochs       = cfg["training"]["epochs"]
    lr           = cfg["training"]["lr"]
    wd           = cfg["training"]["weight_decay"]
    amp_fp16     = cfg["training"]["amp_fp16"]
    num_workers  = cfg["training"].get("num_workers", 0)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = CrimeVideoDataset(
        data_root=cfg["dataset"]["root"], is_train=True,
        num_frames=num_frames, crop_size=image_size
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=False
    )
    print(f"Dataset: {len(dataset)} videos | Steps/epoch: {len(loader)}")

    # ── Masking modules ───────────────────────────────────────────────────
    patch_shape  = (tubelet_size, patch_size, patch_size)
    motion_est   = MotionImportanceEstimator(method=cfg["motion"]["method"],
                                             patch_size=patch_shape).to(device)
    saliency_est = SaliencyImportanceEstimator(patch_size=patch_shape).to(device)
    fusion_mod   = ImportanceFusionModule(alpha=alpha).to(device)
    mask_gen     = ContinuousImportanceMaskGenerator(mask_ratio=mask_ratio)

    # ── Model ─────────────────────────────────────────────────────────────
    decoder_num_classes = patch_size * patch_size * 3 * tubelet_size  # 1536

    model = PretrainVisionTransformer(
        img_size=image_size, patch_size=patch_size,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=decoder_num_classes,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        tubelet_size=tubelet_size, all_frames=num_frames
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler    = torch.cuda.amp.GradScaler(enabled=amp_fp16)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Starting Stage 1 Self-Supervised Pretraining...")
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, (videos, _, _) in enumerate(loader):
            videos = videos.to(device)  # [B, 3, T, H, W]

            # Step 1: compute per-patch importance and mask
            with torch.no_grad():
                m_imp  = motion_est(videos)
                s_imp  = saliency_est(videos)
                fused  = fusion_mod(m_imp, s_imp)   # [B, T_p, H_p, W_p]
                mask   = mask_gen(fused)             # [B, N] bool; True=masked

            # Step 2: pixel-space targets for masked positions [B, N_mask, 1536]
            with torch.no_grad():
                all_patches = patchify(videos, patch_size=patch_size,
                                       tubelet_size=tubelet_size)  # [B, N, 1536]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp_fp16):
                outputs = model(videos, mask)
                B = videos.size(0)
                N_mask = outputs.shape[1]
                masked_targets = all_patches[mask].reshape(B, N_mask, -1).to(outputs.dtype)
                loss = ((outputs - masked_targets) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (step + 1) % 5 == 0 or step == len(loader) - 1:
                print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"--> Epoch [{epoch}/{epochs}] Avg Loss: {avg:.4f}")
        ckpt_path = os.path.join(ckpt_dir, f"videomae_pretrain_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

    print("Pretraining completed successfully.")


if __name__ == "__main__":
    main()
