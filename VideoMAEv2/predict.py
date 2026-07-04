import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu
from timm.models import create_model
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# Register models
import models  # noqa: F401

def get_args():
    parser = argparse.ArgumentParser('VideoMAE Direct Prediction')
    parser.add_argument('--video', default='test_video.mp4', type=str, help='path to video')
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--ckpt', default='vit_b_k710_dl_from_giant.pth', type=str, help='checkpoint path')
    parser.add_argument('--label_map', default='misc/label_map_k710.txt', type=str, help='label map path')
    return parser.parse_args()

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading model...")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif 'module' in checkpoint:
        checkpoint = checkpoint['module']
    
    # If keys start with 'module.', remove the prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Video
    print(f"Opening video: {args.video}")
    vr = VideoReader(args.video, num_threads=1, ctx=cpu(0))
    
    # Sample 16 frames uniformly
    v_len = len(vr)
    indices = np.linspace(0, v_len - 1, 16).astype(int)
    frames = vr.get_batch(indices).asnumpy() # [16, H, W, 3]
    
    # Preprocessing
    # 1. Convert to Tensor [16, H, W, 3] -> [16, 3, H, W]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    
    # 2. Resize short side to 224 and Center Crop
    T, C, H, W = frames.shape
    scale = 224.0 / min(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Center Crop 224x224
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    frames = frames[:, :, top:top+224, left:left+224]
    
    # 3. Normalization (ImageNet defaults used by VideoMAE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    # 4. Prepare for model [B, C, T, H, W]
    input_data = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # 5. Save Sampled Frames for Visualization
    print("Saving sampled frames to 'sampled_frames.png'...")
    # frames is [16, 3, 224, 224], normalize it back for saving
    vis_frames = frames * std + mean
    grid = make_grid(vis_frames, nrow=4)
    save_image(grid, 'sampled_frames.png')

    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_data)
        probs = F.softmax(outputs, dim=1)[0]
    
    # 4. Results
    labels = load_labels(args.label_map)
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    print("\n" + "="*30)
    print(" TOP 5 PREDICTIONS ")
    print("="*30)
    for i in range(top5_prob.size(0)):
        print(f"{labels[top5_catid[i]]:<20} : {top5_prob[i].item():.4f}")
    print("="*30)

if __name__ == '__main__':
    main()
