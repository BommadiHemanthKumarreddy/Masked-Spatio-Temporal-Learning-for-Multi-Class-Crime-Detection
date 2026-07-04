import torch
import torch.nn as nn
import torch.optim as optim
from video_crime_detector import VideoCrimeDetector

def test_video_detector():
    print("=" * 60)
    print("Initializing Video Crime Detector (Spatio-Temporal Vim) Demo...")
    print("=" * 60)
    
    # Configure a lightweight architecture for fast CPU execution
    # Spatial depth = 2 (2D frames), Temporal depth = 1 (time dynamics)
    model = VideoCrimeDetector(
        num_classes=2,        # e.g., 0: Normal Activity, 1: Crime/Anomaly Detected
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=64,
        spatial_depth=2,
        temporal_depth=1,
        d_state=8
    )
    
    print("\n--- Model Configuration ---")
    print(f"Frame Feature Extractor Depth: {len(model.spatial_extractor.layers)} layers")
    print(f"Temporal Modeler Layers:       {len(model.temporal_layers)} layers")
    print(f"Embedding Dimension:           {model.spatial_extractor.embed_dim}")
    print(f"Target Classification Classes: {model.head.out_features} (e.g. Normal vs. Crime)")
    print(f"Total Model Parameters:        {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Generate mock batch: 2 video clips, each containing 4 frames
    # Shape: (Batch=2, Frames=4, Channels=3, Height=224, Width=224)
    print("\n1. Generating dummy video batch...")
    x = torch.randn(2, 4, 3, 224, 224)
    y = torch.tensor([0, 1])  # Target labels: clip 0 is Normal (0), clip 1 is Crime (1)
    
    print(f"   Input video tensor shape:  {list(x.shape)} (Batch, Frames, Channels, Height, Width)")
    print(f"   Target labels tensor shape: {list(y.shape)}")
    
    # Forward Pass
    print("\n2. Executing forward pass...")
    model.train()
    logits = model(x)
    print(f"   Logits shape: {list(logits.shape)}")
    
    # Loss computation
    print("\n3. Computing Cross Entropy loss...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(f"   Loss value:   {loss.item():.4f}")
    
    # Backward Pass
    print("\n4. Executing backward pass...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    print("   Backward pass completed successfully.")
    
    # Step optimizer
    print("\n5. Updating weights via optimizer...")
    optimizer.step()
    print("   Weight update completed successfully.")
    
    print("\n" + "=" * 60)
    print("VIDEO DETECTOR VERIFICATION SUCCESSFUL!")
    print("Spatio-Temporal Video Crime Detector is operational and ready to train!")
    print("=" * 60)

if __name__ == "__main__":
    test_video_detector()
