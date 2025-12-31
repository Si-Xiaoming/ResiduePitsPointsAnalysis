_base_ = ["../_base_/default_runtime_backbone.py"]

# misc custom settings


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=6,
        num_classes=4,
        base_grid_size=0.02,
        depths=[2, 4, 9, 4, 4],
        channels=[80, 160, 320, 640, 640],
        num_heads=[10, 10, 20, 40, 40],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        drop_path_rate=0.3,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=3,
        upsample="linear_attn",
        knn_down=True,
        cRSE="XYZ_RGB_NORM",
        fp16_mode=1,
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


