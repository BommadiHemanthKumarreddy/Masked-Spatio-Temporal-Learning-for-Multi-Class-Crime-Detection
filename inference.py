import os
import sys
import yaml
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.crime_dataset import CrimeVideoDataset
from masking.motion_importance import MotionImportanceEstimator
from masking.saliency_importance import SaliencyImportanceEstimator
from masking.importance_fusion import ImportanceFusionModule
from masking.continuous_masking import ContinuousImportanceMaskGenerator
from masking.visualization import save_masking_visualization
from models.crime_classifier import VideoMAE_CrimeClassifier

def main():
    config_path = os.path.join(os.path.dirname(__file__), "configs/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset sample
    test_dataset = CrimeVideoDataset(cfg["dataset"]["root"], is_train=False, num_frames=16)
    video_tensor, label, video_path = test_dataset[0]
    video_tensor = video_tensor.unsqueeze(0).to(device) # [1, 3, T, H, W]

    # Initialize Masking Modules
    motion_est = MotionImportanceEstimator(method=cfg["motion"]["method"]).to(device)
    saliency_est = SaliencyImportanceEstimator().to(device)
    fusion_mod = ImportanceFusionModule(alpha=cfg["masking"]["alpha"]).to(device)
    mask_gen = ContinuousImportanceMaskGenerator(mask_ratio=cfg["masking"]["mask_ratio"]).to(device)

    # Compute Importance Maps & Mask for Frame 0
    with torch.no_grad():
        m_imp = motion_est(video_tensor) # [1, T_p, H_p, W_p]
        s_imp = saliency_est(video_tensor) # [1, T_p, H_p, W_p]
        fused_imp = fusion_mod(m_imp, s_imp)
        mask = mask_gen(fused_imp)

    # Model & Classification Prediction
    model = VideoMAE_CrimeClassifier(
        img_size=cfg["video"]["image_size"],
        patch_size=cfg["video"]["patch_size"],
        num_classes=cfg["dataset"]["num_classes"],
        embed_dim=768,
        depth=12,
        num_heads=12,
        tubelet_size=cfg["video"]["tubelet_size"],
        all_frames=cfg["video"]["num_frames"]
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(video_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    class_names = cfg["dataset"]["class_names"]
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item() * 100.0

    print("==================== INFERENCE RESULT ====================")
    print(f"Video File: {os.path.basename(video_path)}")
    print(f"Predicted Crime Class: {pred_class}")
    print(f"Confidence Score: {confidence:.2f}%")
    print(f"Ground Truth Class: {class_names[label]}")

    print("\nTop Temporal Segments (Attention Weights):")
    attn = attn_weights[0].cpu().numpy()
    for t_idx, a_val in enumerate(attn):
        print(f"  Segment {t_idx+1} -> Weight: {a_val:.4f}")

    # Generate Visualization Figure
    vis_path = os.path.join(os.path.dirname(__file__), "inference_visualization.png")
    save_masking_visualization(
        frame=video_tensor[0, :, 0],
        motion_map=m_imp[0, 0],
        saliency_map=s_imp[0, 0],
        importance_map=fused_imp[0, 0],
        mask_pos=mask[0],
        save_path=vis_path
    )

if __name__ == "__main__":
    main()
