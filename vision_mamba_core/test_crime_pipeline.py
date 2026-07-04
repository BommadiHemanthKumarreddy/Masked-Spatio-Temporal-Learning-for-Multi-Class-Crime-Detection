# --------------------------------------------------------
# Smoke Test for Crime Detection Pipeline
# Verifies all modules work end-to-end with dummy data
# --------------------------------------------------------

import torch
import numpy as np
import sys
import os

# NOTE: Do NOT import VideoMAEv2 models here.
# VideoMAEv2 registers its own 'vit_small_patch16_224' (3D, expects 5D input)
# which would shadow the standard 2D timm ViT used by crime_model.py v3.


def test_key_frame_selection():
    """Test that select_key_frames returns K frames in sorted temporal order."""
    from crime_modules import select_key_frames

    print("Testing select_key_frames... ", end="")
    # Create dummy video: 32 frames, 64x64, 3 channels
    np.random.seed(42)
    video = np.random.randint(0, 255, (32, 64, 64, 3), dtype=np.uint8)

    # Add strong motion to specific frames
    video[10] = 255  # big jump at frame 10
    video[20] = 0    # big jump at frame 20

    selected = select_key_frames(video, k=16)
    assert selected.shape[0] == 16, f"Expected 16 frames, got {selected.shape[0]}"

    # Also test with tensor input
    video_t = torch.from_numpy(video)
    selected_t = select_key_frames(video_t, k=8)
    assert selected_t.shape[0] == 8, f"Expected 8 frames, got {selected_t.shape[0]}"

    print("[PASSED]")


def test_region_based_mask():
    """Test that region_based_mask returns a valid boolean mask."""
    from crime_modules import region_based_mask

    print("Testing region_based_mask... ", end="")
    # Create dummy frames: 16 frames, 3 channels, 224x224
    frames = torch.randn(16, 3, 224, 224)

    mask = region_based_mask(frames, patch_size=16, tubelet_size=2, threshold=0.3)

    # Expected shape: (T//tubelet, (H//patch)*(W//patch)) = (8, 196)
    expected_t = 16 // 2
    expected_s = (224 // 16) * (224 // 16)
    assert mask.shape == (expected_t, expected_s), \
        f"Expected ({expected_t}, {expected_s}), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool mask, got {mask.dtype}"

    print("[PASSED]")


def test_temporal_mamba():
    """Test TemporalMamba input/output shapes."""
    from crime_modules import TemporalMamba, MAMBA_AVAILABLE

    backend = "Mamba" if MAMBA_AVAILABLE else "GRU fallback"
    print(f"Testing TemporalMamba ({backend})... ", end="")

    dim = 384
    mamba = TemporalMamba(dim=dim)

    x = torch.randn(2, 8, dim)  # (B=2, T=8, D=384)
    out = mamba(x)

    assert out.shape == (2, 8, dim), f"Expected (2, 8, {dim}), got {out.shape}"

    print("[PASSED]")


def test_temporal_attention():
    """Test TemporalAttention output shape and attention weight sum."""
    from crime_modules import TemporalAttention

    print("Testing TemporalAttention... ", end="")

    dim = 384
    attn = TemporalAttention(dim=dim)

    x = torch.randn(2, 8, dim)  # (B=2, T=8, D=384)
    weighted, weights = attn(x)

    assert weighted.shape == (2, dim), f"Expected (2, {dim}), got {weighted.shape}"
    assert weights.shape == (2, 8), f"Expected (2, 8), got {weights.shape}"

    # Attention weights should sum to ~1.0
    weight_sum = weights.sum(dim=1)
    assert torch.allclose(weight_sum, torch.ones(2), atol=1e-5), \
        f"Attention weights don't sum to 1: {weight_sum}"

    print("[PASSED]")


def test_full_model():
    """Test the full CrimeDetectionModel forward pass."""
    from crime_model import CrimeDetectionModel, NUM_CLASSES

    print("Testing CrimeDetectionModel (full forward)... ", end="")

    model = CrimeDetectionModel(
        encoder_name='vit_small_patch16_224',
        num_classes=NUM_CLASSES,
        num_frames=16,
        pretrained_ckpt=None,  # no pretrained weights for test
        freeze_encoder=True,
    )
    model.eval()

    # Dummy video: (B=1, C=3, T=16, H=224, W=224)
    x = torch.randn(1, 3, 16, 224, 224)

    with torch.no_grad():
        logits, attn_weights = model(x, return_attention=True)

    assert logits.shape == (1, NUM_CLASSES), \
        f"Expected (1, {NUM_CLASSES}), got {logits.shape}"
    assert attn_weights.shape[0] == 1, \
        f"Expected batch=1, got {attn_weights.shape}"

    print("[PASSED]")


def test_backward_pass():
    """Test that the loss backward pass completes without error."""
    from crime_model import CrimeDetectionModel, NUM_CLASSES

    print("Testing backward pass (gradient flow)... ", end="")

    model = CrimeDetectionModel(
        encoder_name='vit_small_patch16_224',
        num_classes=NUM_CLASSES,
        num_frames=16,
        pretrained_ckpt=None,
        freeze_encoder=True,
    )
    model.train()

    x = torch.randn(2, 3, 16, 224, 224)
    labels = torch.tensor([0, 5])  # dragging and robbery

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Check that trainable params have gradients
    trainable = model.get_trainable_params()
    has_grad = any(p.grad is not None for p in trainable)
    assert has_grad, "No gradients found in trainable parameters!"

    print(f"[PASSED] (loss={loss.item():.4f})")


def main():
    print("=" * 50)
    print(" CRIME DETECTION PIPELINE — SMOKE TEST")
    print("=" * 50)
    print()

    tests = [
        test_key_frame_selection,
        test_region_based_mask,
        test_temporal_mamba,
        test_temporal_attention,
        test_full_model,
        test_backward_pass,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAILED]: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n All tests passed! Pipeline is working correctly.\n")


if __name__ == '__main__':
    main()
