import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_masking_visualization(
    frame,
    motion_map,
    saliency_map,
    importance_map,
    mask_pos,
    save_path="mask_vis.png"
):
    # frame: [3, H, W] in [0, 1]
    # motion_map: [H_p, W_p]
    # saliency_map: [H_p, W_p]
    # importance_map: [H_p, W_p]
    # mask_pos: [N_p] boolean array
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    # Frame
    frame_img = frame.detach().cpu().numpy().transpose(1, 2, 0)
    frame_img = (frame_img - frame_img.min()) / (frame_img.max() - frame_img.min() + 1e-6)
    axes[0].imshow(frame_img)
    axes[0].set_title("Original Frame")
    axes[0].axis('off')

    # Motion Map
    axes[1].imshow(motion_map.detach().cpu().numpy(), cmap='jet')
    axes[1].set_title("RAFT Motion Map")
    axes[1].axis('off')

    # Saliency Map
    axes[2].imshow(saliency_map.detach().cpu().numpy(), cmap='magma')
    axes[2].set_title("U²-Net Saliency Map")
    axes[2].axis('off')

    # Importance Map
    axes[3].imshow(importance_map.detach().cpu().numpy(), cmap='viridis')
    axes[3].set_title("Fused Importance (α=0.7)")
    axes[3].axis('off')

    # Mask Map
    m_shape = motion_map.shape
    mask_grid = mask_pos.reshape(-1, m_shape[0], m_shape[1])[0].detach().cpu().numpy()
    axes[4].imshow(mask_grid, cmap='gray')
    axes[4].set_title("Continuous Mask (90%)")
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {save_path}")
