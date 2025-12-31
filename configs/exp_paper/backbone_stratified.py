_base_ = ["../_base_/default_runtime_backbone.py"]



# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="ST-v1m2",
        in_channels=6,
        num_classes=4,
        channels=(48, 96, 192, 384, 384),
        num_heads=(6, 12, 24, 24),
        depths=(3, 9, 3, 3),
        window_size=(0.2, 0.4, 0.8, 1.6),
        quant_size=(0.01, 0.02, 0.04, 0.08),
        mlp_expend_ratio=4.0,
        down_ratio=0.25,
        down_num_sample=16,
        kp_ball_radius=2.5 * 0.02,
        kp_max_neighbor=34,
        kp_grid_size=0.02,
        kp_sigma=1.0,
        drop_path_rate=0.2,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        qkv_bias=True,
        stem=True,
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



