_base_ = ["../_base_/default_runtime_backbone.py"]


batch_size = 1
# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m1",
        in_channels=6,
        num_classes=4,
        patch_embed_depth=2,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=32,
        enc_depths=(2, 4, 2),
        enc_channels=(96, 192, 384),
        enc_groups=(12, 24, 48),
        enc_neighbours=(16, 16, 16),
        dec_depths=(1, 1, 1),
        dec_channels=(48, 96, 192),
        dec_groups=(6, 12, 24),
        dec_neighbours=(16, 16, 16),
        grid_sizes=(0.2, 0.4, 0.8),
        attn_qkv_bias=True,
        pe_multiplier=True,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.15,
        enable_checkpoint=False,
        unpool_backend="interp",  # map / interp
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]



