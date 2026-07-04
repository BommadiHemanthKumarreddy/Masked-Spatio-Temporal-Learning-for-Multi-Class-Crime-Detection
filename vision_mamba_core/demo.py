import torch
import torch.nn as nn
import torch.optim as optim
from vision_mamba import VisionMamba

def run_demo():
    print("=" * 60)
    print("Initializing Vision Mamba (Vim) Demo...")
    print("=" * 60)
    
    # Define a lightweight model configuration for fast verification on CPU
    # Standard settings use depth=24, but depth=2 is sufficient to verify correctness of all components
    model = VisionMamba(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=64,     # lightweight embedding size
        depth=2,          # lightweight depth
        d_state=8,        # lightweight state dimension
        d_conv=4,
        expand=2,
        bimamba_type="v1",# Bidirectional scanning enabled
        drop_rate=0.1,
        drop_path_rate=0.1
    )
    
    print("\n--- Model Architecture Summary ---")
    print(f"Embedding Dimension: {model.embed_dim}")
    print(f"Number of Mamba Blocks: {len(model.layers)}")
    print(f"Classification Classes: {model.num_classes}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 1. Create mock batch (Batch size = 2, Channels = 3, Height = 224, Width = 224)
    print("\n1. Generating dummy input batch and target labels...")
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([3, 7]) # Dummy classification labels
    print(f"   Input shape:  {list(x.shape)}")
    print(f"   Target shape: {list(y.shape)}")
    
    # 2. Forward Pass
    print("\n2. Executing forward pass...")
    model.train()
    logits = model(x)
    print(f"   Logits shape: {list(logits.shape)}")
    
    # 3. Loss Computation
    print("\n3. Computing Cross Entropy loss...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(f"   Loss value:   {loss.item():.4f}")
    
    # 4. Backward Pass
    print("\n4. Executing backward pass (gradient calculation)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    print("   Backward pass completed successfully.")
    
    # 5. Optimization step
    print("\n5. Updating weights using AdamW optimizer...")
    optimizer.step()
    print("   Weight update completed successfully.")
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUCCESSFUL!")
    print("Vision Mamba (Vim) implemented successfully and runs on Windows!")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
