import torch
import torch.nn as nn
from vision_mamba import VisionMamba
from mamba import MambaBlock

class VideoCrimeDetector(nn.Module):
    """
    Spatio-Temporal Video Crime Detector.
    Uses 2D Vision Mamba (Vim) to extract spatial frame-level features,
    and a temporal Mamba Block to model the progression of actions/events
    over time (e.g. detecting violence, theft, or anomalies in surveillance video).
    """
    def __init__(self, num_classes=2, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=192, spatial_depth=12, temporal_depth=1, d_state=16):
        """
        Args:
            num_classes: Number of target crime/anomaly classes (e.g., 2 for Normal vs. Crime).
            img_size: Height/Width of input video frames.
            patch_size: Patch size for spatial partition.
            in_chans: Number of image channels (typically 3).
            embed_dim: Feature embedding size.
            spatial_depth: Number of Vision Mamba blocks for feature extraction per frame.
            temporal_depth: Number of temporal Mamba blocks.
            d_state: SSM state dimension.
        """
        super().__init__()
        # 1. Spatial Feature Extractor (2D Vision Mamba)
        # num_classes=0 configures the model to output the raw class embedding (cls_token)
        # rather than class logits.
        self.spatial_extractor = VisionMamba(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depth=spatial_depth,
            d_state=d_state
        )
        
        # 2. Temporal Sequence Modeler (stack of temporal Mamba Blocks)
        # Processes sequences of frame embeddings across the video timeline.
        self.temporal_layers = nn.ModuleList([
            MambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                bimamba_type="v1"  # Bidirectional scanning models both past and future frame contexts
            )
            for _ in range(temporal_depth)
        ])
        
        # 3. Final Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, video):
        """
        Args:
            video: Tensor of shape (B, T, C, H, W)
                   B: Batch size
                   T: Number of video frames (sequence length)
                   C: Channels (e.g. 3)
                   H: Height (e.g. 224)
                   W: Width (e.g. 224)
        Returns:
            logits: Predicted class logits of shape (B, num_classes)
        """
        B, T, C, H, W = video.shape
        
        # Flatten batch and time dimensions to extract features for all frames in parallel
        # Shape: (B * T, C, H, W)
        flat_frames = video.view(B * T, C, H, W)
        
        # Extract features per frame
        # Shape: (B * T, embed_dim)
        frame_features = self.spatial_extractor(flat_frames)
        
        # Reshape back to sequence of frames per video
        # Shape: (B, T, embed_dim)
        video_sequence = frame_features.view(B, T, -1)
        
        # Run through temporal Mamba sequence modelers
        for layer in self.temporal_layers:
            video_sequence = layer(video_sequence)
            
        # Aggregate features across the timeline (temporal average pooling)
        # Shape: (B, embed_dim)
        video_representation = video_sequence.mean(dim=1)
        
        # Project representation to class logits
        # Shape: (B, num_classes)
        logits = self.head(video_representation)
        
        return logits
