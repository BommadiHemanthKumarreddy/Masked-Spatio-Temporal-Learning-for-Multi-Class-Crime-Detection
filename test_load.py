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

model.load_pretrained(
    "./logs/prod_combined_b4_e50_modelmae_vit_base_custom_in1024_mask0.7_wd0.05_warmup40_d12_heads8_dec_embed_dim512_dec_d8_dec_heads16_distributed_False_cosloss_False_ms_False/checkpoint-49.pth"
)

print("\nModel loaded successfully!")

x = __import__("torch").randn(4, 1024)

y = model(x)

print("Output Shape:", y.shape)
