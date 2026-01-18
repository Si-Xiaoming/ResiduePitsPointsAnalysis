_base_ = ["../_base_/default_runtime_backbone.py"]

epoch = 300
enable_amp = False
batch_size = 2
# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1
            #, weight=class_weights
            ),
        # dict(type="Poly1CrossEntropyLoss", 
        #  loss_weight=1.0, 
        #  ignore_index=-1, 
        #  epsilon=1.0,
        #  weight=class_weights),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)
# scheduler settings

# optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=[0.006, 0.0006], # , 0.0006
#     pct_start=0.05,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=1000.0,
# )
# param_dicts = [dict(keyword="block", lr=0.0006)]


optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]
