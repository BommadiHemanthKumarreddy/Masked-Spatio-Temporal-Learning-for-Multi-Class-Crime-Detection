import torch
from types import SimpleNamespace
from classifier import LVMAEClassifier

args = SimpleNamespace(
    model="mae_vit_base_custom",
    input_size=1024,
    depth=12,
    num_heads=8,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    num_patches=256,
    norm_pix_loss=False,
    cos_loss=False,
)

model = LVMAEClassifier(args)

x = torch.randn(4,1024)

y = model(x)

print(y.shape)
