# --------------------------------------------------------
# Inference Script for Crime Detection Model
# Task 9: Load model, process video, predict crime class
# --------------------------------------------------------

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu

from crime_modules import select_key_frames
from crime_model import CrimeDetectionModel, CRIME_CLASSES, NUM_CLASSES


def get_args():
    parser = argparse.ArgumentParser('Crime Detection Inference')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--model_ckpt', type=str, default='checkpoints/best_crime_model.pth',
                        help='Path to trained crime model checkpoint')
    parser.add_argument('--encoder', type=str, default='vit_small_patch16_224',
                        help='Encoder model name')
    parser.add_argument('--encoder_ckpt', type=str, default='vit_s_k710_dl_from_giant.pth',
                        help='Pretrained encoder checkpoint')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--use_key_frames', action='store_true', default=True,
                        help='Use motion-based key frame selection')
    return parser.parse_args()


def preprocess_video(video_path, num_frames=16, use_key_frames=True):
    """Load and preprocess a video for inference."""
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    total_frames = len(vr)

    # Sample frames
    if use_key_frames and total_frames > num_frames:
        sample_count = min(total_frames, num_frames * 4)
        indices = np.linspace(0, total_frames - 1, sample_count).astype(int)
        
        # Fetch unique frames first and index them to avoid duplicate decord get_batch bugs
        unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
        unique_frames = vr.get_batch(unique_indices).asnumpy()
        all_frames = unique_frames[inverse_indices]
        
        frames = select_key_frames(all_frames, k=num_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        # Fetch unique frames first and index them to avoid duplicate decord get_batch bugs
        unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
        unique_frames = vr.get_batch(unique_indices).asnumpy()
        frames = unique_frames[inverse_indices]

    # Convert to tensor: (K, H, W, 3) → (K, 3, H, W)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32) / 255.0

    # Resize and center crop to 224x224
    K, C, H, W = frames.shape
    scale = 224.0 / min(H, W)
    new_h = max(224, round(H * scale))
    new_w = max(224, round(W * scale))
    frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)

    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    frames = frames[:, :, top:top + 224, left:left + 224]

    # Normalize (ImageNet defaults)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # Reshape to (1, C, T, H, W)
    video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, K, 224, 224)

    return video_tensor


def save_frame_visualizations(frames_tensor, attn_weights, output_prefix='crime_output'):
    """
    Save key frame grid and attention-weighted visualization.

    Args:
        frames_tensor: (K, 3, H, W) preprocessed frames (normalized)
        attn_weights: (T,) attention weights per temporal step
        output_prefix: filename prefix for output images
    """
    from torchvision.utils import save_image, make_grid
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Denormalize frames for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    vis_frames = frames_tensor.cpu() * std + mean
    vis_frames = vis_frames.clamp(0, 1)

    # 1. Save plain key frame grid
    grid = make_grid(vis_frames, nrow=4, padding=4, pad_value=1.0)
    save_image(grid, f'{output_prefix}_keyframes.png')
    print(f"  Saved: {output_prefix}_keyframes.png")

    # 2. Save attention-weighted frame visualization
    num_frames = vis_frames.shape[0]
    num_temporal = len(attn_weights)

    # Map attention weights to frames (each temporal token covers tubelet_size frames)
    tubelet = num_frames // num_temporal
    frame_weights = []
    for t in range(num_temporal):
        for _ in range(tubelet):
            frame_weights.append(attn_weights[t])
    frame_weights = np.array(frame_weights[:num_frames])

    # Normalize weights for display
    w_min, w_max = frame_weights.min(), frame_weights.max()
    if w_max > w_min:
        norm_weights = (frame_weights - w_min) / (w_max - w_min)
    else:
        norm_weights = np.ones_like(frame_weights)

    # Create matplotlib figure with attention bars
    nrow, ncol = 2, min(8, num_frames)
    if num_frames > 8:
        nrow = (num_frames + ncol - 1) // ncol

    fig = plt.figure(figsize=(ncol * 3, nrow * 3.5))
    gs = gridspec.GridSpec(nrow, ncol, hspace=0.4, wspace=0.1)

    for i in range(num_frames):
        r, c = i // ncol, i % ncol
        ax = fig.add_subplot(gs[r, c])

        # Convert tensor frame to numpy for matplotlib
        frame_np = vis_frames[i].permute(1, 2, 0).numpy()
        ax.imshow(frame_np)

        # Color border by attention weight (red=high, blue=low)
        color = plt.cm.RdYlGn(norm_weights[i])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        ax.set_title(f"F{i+1} | Attn: {frame_weights[i]:.3f}",
                     fontsize=9, fontweight='bold',
                     color='darkred' if norm_weights[i] > 0.7 else 'black')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Temporal Attention Weights per Frame\n(Green border = high attention, Red = low)",
                 fontsize=13, fontweight='bold')
    plt.savefig(f'{output_prefix}_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_attention.png")


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Build Model ----
    print("Building model...")
    model = CrimeDetectionModel(
        encoder_name=args.encoder,
        num_classes=NUM_CLASSES,
        num_frames=args.num_frames,
        pretrained_ckpt=args.encoder_ckpt,
        freeze_encoder=True,
    ).to(device)

    # Load trained crime model weights (if available)
    if args.model_ckpt and args.model_ckpt.strip():
        print(f"Loading trained weights from: {args.model_ckpt}")
        checkpoint = torch.load(args.model_ckpt, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("[INFO] No trained crime model loaded — using encoder weights only.")

    model.eval()

    # ---- Preprocess Video ----
    print(f"Processing video: {args.video}")
    video_tensor = preprocess_video(
        args.video,
        num_frames=args.num_frames,
        use_key_frames=args.use_key_frames,
    ).to(device)

    # Also get raw frames for visualization
    vr = VideoReader(args.video, num_threads=1, ctx=cpu(0))
    total_frames_count = len(vr)
    if args.use_key_frames and total_frames_count > args.num_frames:
        sample_count = min(total_frames_count, args.num_frames * 4)
        indices = np.linspace(0, total_frames_count - 1, sample_count).astype(int)
        all_raw = vr.get_batch(indices).asnumpy()
        raw_frames = select_key_frames(all_raw, k=args.num_frames)
    else:
        indices = np.linspace(0, total_frames_count - 1, args.num_frames).astype(int)
        raw_frames = vr.get_batch(indices).asnumpy()

    # ---- Inference ----
    print("Running inference...")
    with torch.no_grad():
        logits, attn_weights = model(video_tensor, return_attention=True)
        probs = F.softmax(logits, dim=1)[0]

    # ---- Results ----
    top5_prob, top5_idx = torch.topk(probs, min(5, NUM_CLASSES))

    print("\n" + "=" * 50)
    print("  CRIME DETECTION RESULTS")
    print("=" * 50)
    for i in range(top5_prob.size(0)):
        cls_name = CRIME_CLASSES[top5_idx[i]]
        confidence = top5_prob[i].item() * 100
        print(f"  {cls_name:<20s} : {confidence:6.2f}%")
    print("=" * 50)

    # ---- Attention Weights ----
    attn = attn_weights[0].cpu().numpy()
    print("\n  TEMPORAL ATTENTION WEIGHTS (per frame):")
    print("  " + "-" * 40)
    for t in range(len(attn)):
        bar = "█" * int(attn[t] * 100)
        print(f"  Frame {t+1:2d}: {attn[t]:.4f} {bar}")
    print("  " + "-" * 40)

    peak_frame = np.argmax(attn) + 1
    print(f"\n  → Most attended frame: Frame {peak_frame} "
          f"(weight={attn[peak_frame-1]:.4f})")

    # ---- Save Visualizations ----
    print("\nSaving frame visualizations...")
    # Prepare frames tensor for visualization (K, 3, H, W)
    vis_tensor = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float() / 255.0
    vis_tensor = F.interpolate(vis_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    # Normalize for saving
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    vis_normalized = (vis_tensor - mean) / std

    save_frame_visualizations(vis_normalized, attn, output_prefix='crime_output')
    print("\nDone!")


if __name__ == '__main__':
    main()
