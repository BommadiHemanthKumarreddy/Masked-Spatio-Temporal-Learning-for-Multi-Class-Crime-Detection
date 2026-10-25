import os
import torch

from languagebind_2sec import LanguageBind, to_device, transform_dict

# Load LanguageBind
video_model_name = "LanguageBind_Video_Huge_V1.5_FT"

clip_type = {"video": video_model_name}

device = "cuda"

model = LanguageBind(
    clip_type=clip_type,
    cache_dir="./cache_dir"
)

model = model.to(device)
model.eval()

modality_transform = {
    c: transform_dict[c](model.modality_config[c])
    for c in clip_type.keys()
}

clip_path = "/tmp/test_clip.mp4"

inputs = {
    "video": to_device(
        modality_transform["video"]([clip_path]),
        device
    )
}

with torch.no_grad():
    embedding = model(inputs)["video"]

print("Embedding shape:", embedding.shape)

torch.save(
    {"embedding": embedding.cpu()},
    "test_embedding.pt"
)

print("Saved test_embedding.pt")
