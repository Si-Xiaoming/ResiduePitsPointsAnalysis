_base_ = ["../_base_/default_runtime.py"]

# ==============================================================================
# Global Settings
# ==============================================================================
weight = "/home/shsi/outputs/residue/residue-segment/subset-4-11/model/model_best.pth"  # 模型路径
save_path = "/home/shsi/outputs/residue/residue-segment/subset-4-11/lazout"                        # 结果保存路径
laz_file = "/home/shsi/outputs/residue/residue-segment/subset-4-11/navarra/02.laz"       # 待推理的大文件
grid_size = 0.1                                            # 体素大小 (需与训练保持一致)

# ==============================================================================
# Spatial Tiling Settings (关键参数)
# ==============================================================================
# block_size: 空间切块的大小（单位：米）。建议 30-100m，取决于显存。
# stride: 滑动窗口的步长。stride = block_size * (1 - overlap_ratio)
# 如果 overlap_ratio = 0.1 (10%重叠)，且 block_size = 50，则 stride = 45
block_size = 50.0
stride = 45.0  

# ==============================================================================
# Model Settings
# ==============================================================================
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

# ==============================================================================
# Data Settings
# ==============================================================================
data = dict(
    num_classes=4,
    ignore_index=-1,
    names=["ground", "vegetation", "building", "others"],
    
    # Train is strictly not needed for pure inference, but kept for compatibility
    train=dict(
        type="PredictDataset", 
        split="train", 
        data_root="", 
        test_mode=False
    ),

    # 关键修改部分
    test=dict(
        type="PredictDataset",          # 对应我们重写的 Dataset 类名
        test_file=laz_file,             # 传入文件路径
        block_size=block_size,          # 传入分块大小
        stride=stride,                  # 传入步长
        test_mode=True,
        test_cfg=dict(
            has_label=True,             # 如果LAZ里有真值用于评估，设为True
        ),
        transforms=[
            dict(type="NormalizeColor"), # 归一化颜色 (0-255 -> 0-1)
            # GridSample 负责将切好的 block 进行体素化
            # mode="test" 会生成 index 映射，这对我们很重要
            dict(
                type="GridSample", 
                grid_size=grid_size, 
                hash_type="fnv", 
                mode="test", 
                return_grid_coord=True
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                # 必须包含 "index"，这是 SpatialPredictDataset 传出来的全局索引
                keys=("coord", "grid_coord", "segment", "index"), 
                feat_keys=("coord", "color")
            )
        ]
    )
)

# ==============================================================================
# Environment & Tester Settings
# ==============================================================================
batch_size = 1       # 每次处理一个 Block
num_workers = 4      # 数据加载线程数
enable_amp = True    # 开启混合精度加速
empty_cache = True   # 每个 Block 后清理缓存，防止碎片化

# Tester 必须匹配我们在 test_laz.py 中注册的类名
test = dict(
    type="SemSegTesterLaz", 
    verbose=True
)