_base_ = ["../_base_/default_runtime_backbone.py"]


batch_size = 1
# model settings

model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m1", 
        in_channels=6, # 确保你的数据加载器确实输出了6个特征 (XYZ + RGB 或 XYZ + Intensity + Return + Time)
        num_classes=4, 
        
        # === [修复 1]: Embedding Group 整除性 ===
        # 错误: 64 / 6 = 10.66 (除不尽，报错)
        # 修正: 64 / 8 = 8 (整除)
        patch_embed_depth=2,
        patch_embed_channels=64, 
        patch_embed_groups=8,     # 修正为 8
        patch_embed_neighbours=32,


        enc_depths=(3, 3, 12, 3), 
        enc_channels=(96, 192, 384, 512), 
        
        # === [修复 3]: Vector Attention Group 整除性 ===
        # 错误: 512 / 96 = 5.33 (除不尽)
        # 修正: 512 / 64 = 8 (整除，且保持每个Group 8个通道)
        enc_groups=(12, 24, 48, 64), 
        enc_neighbours=(32, 32, 32, 32), 

        # === Decoder 设置 (对称设计) ===
        dec_depths=(1, 1, 1, 1),
        dec_channels=(96, 192, 384, 512), 
        dec_groups=(12, 24, 48, 64), # 必须与 enc_groups 对应修正
        dec_neighbours=(32, 32, 32, 32), 

        # Grid Sizes (保持你的设置，搭配 stride=4 使用)
        grid_sizes=(0.25, 1.0, 4.0, 16.0), 
        
        attn_qkv_bias=True,
        pe_multiplier=False, 
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3, 
        enable_checkpoint=True, # 强烈建议开启，因为你用了 Stride=4 和 Deep Layer，显存压力极大
        unpool_backend="interp", 
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR", 
    max_lr=[0.006, 0.0006], # , 0.0006
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]



