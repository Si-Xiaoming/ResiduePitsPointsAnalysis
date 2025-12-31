_base_ = ["../_base_/default_runtime_backbone.py"]



# weight = "/datasets/exp-0801/model/supervised/model_best_supervised.pth"

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="MinkUNet34C", in_channels=6, out_channels=4),
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


