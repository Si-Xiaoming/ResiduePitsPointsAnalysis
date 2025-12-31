_base_ = ["../_base_/default_runtime_backbone.py"]



# weight = "/datasets/exp-0801/model/supervised/model_best_supervised.pth"

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OACNNs",
        in_channels=6,
        num_classes=4,
        embed_channels=64,
        enc_channels=[64, 64, 128, 256],
        groups=[4, 4, 8, 16],
        enc_depth=[3, 3, 9, 8],
        dec_channels=[256, 256, 256, 256],
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
        dec_depth=[2, 2, 2, 2],
        enc_num_ref=[16, 16, 16, 16],
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)
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


