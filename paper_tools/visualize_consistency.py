import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from collections import OrderedDict
from pointcept.models import build_model
# [修改] 增加导入 collate_fn
from pointcept.datasets import build_dataset, collate_fn 
from pointcept.models.utils.structure import Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pointcept 依赖
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.models.utils.structure import Point
import pointops

def get_jet_color(value):
    """
    将 0-1 的误差值映射为 Jet 颜色 (蓝 -> 红)
    用于误差热力图: 蓝色(低误差) -> 红色(高误差)
    """
    cmap = plt.get_cmap('jet')
    # value shape: (N,)
    rgba = cmap(value)
    return (rgba[:, :3] * 255).astype(np.uint8)

def get_pca_color(feat, brightness=1.25, center=True):
    """
    使用 PCA 将高维特征投影到 RGB 空间进行可视化
    参照 misc/pca.py 的实现
    """
    # PCA 降维 (N, C) -> (N, 3)
    # q=3 因为我们需要 RGB 3个通道
    try:
        u, s, v = torch.pca_lowrank(feat, center=center, q=3, niter=5)
    except Exception as e:
        # Fallback if feat dimension is too small
        print(f"Warning: PCA failed ({e}), using random projection")
        v = torch.randn(feat.shape[1], 3, device=feat.device)
        v = F.normalize(v, dim=0)

    projection = feat @ v
    
    # 归一化到 [0, 1] 区间以便作为颜色显示
    min_val = projection.min(dim=0, keepdim=True)[0]
    max_val = projection.max(dim=0, keepdim=True)[0]
    
    # 避免除以零
    div = max_val - min_val
    div[div < 1e-6] = 1.0 
    
    color = (projection - min_val) / div
    
    # 亮度增强
    color = color * brightness
    color = color.clamp(0.0, 1.0)
    
    return (color.cpu().numpy() * 255).astype(np.uint8)

def write_ply_color(save_path, points, colors):
    """
    保存带颜色的点云为 PLY
    points: (N, 3)
    colors: (N, 3) uint8
    """
    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(points.shape[0]):
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

def load_checkpoint(model, filename):
    """
    手动加载权重，处理 DDP 前缀 'module.'
    设置 strict=False 以忽略 Checkpoint 中多余的键 (如 density_head)
    """
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        # 这里的 weights_only=False 是为了保持兼容性，如果你确信文件安全，可以改为 True
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # [修改] strict=False 允许加载包含多余键（如 density_head）的权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("=> Loaded successfully.")
        
        # 打印匹配信息以便调试
        if len(msg.missing_keys) > 0:
            print(f"   Missing keys: {len(msg.missing_keys)} (Expected if partial load)")
            # print(msg.missing_keys) # 如果需要详细信息取消注释
        if len(msg.unexpected_keys) > 0:
            print(f"   Unexpected keys: {len(msg.unexpected_keys)} (Likely auxiliary heads from training)")
            # print(msg.unexpected_keys) # 如果需要详细信息取消注释
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="path/to/your/config.py", help="path to config file")
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument("--checkpoint", default="path/to/your/model_best.pth", help="path to checkpoint")
    parser.add_argument("--save-path", default="vis_results", help="path to save ply files")
    parser.add_argument("--drop-rate", type=float, default=0.8, help="Student view drop rate (e.g. 0.8 means keep 20%)")
    args = parser.parse_args()

    # 1. 配置加载
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg.batch_size = 1
    cfg.num_worker = 0
    
    # 2. 模型构建
    print(f"=> Building model: {cfg.model.type}")
    model = build_model(cfg.model)
    model = model.cuda()
    model.eval()

    # 加载权重
    load_checkpoint(model, args.checkpoint)

    # 3. 数据集构建 (使用 val 或 test)
    print(f"=> Building dataset: {cfg.data.val.type}")
    # 优先尝试读取 val 配置，否则读取 test
    dataset_cfg = cfg.data.val if hasattr(cfg.data, 'val') else cfg.data.test
    dataset = build_dataset(dataset_cfg) 
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("=> Start generating visualizations (Consistency Error + PCA Features)...")

    with torch.no_grad():
        for idx, data_dict in enumerate(loader):
            # 将数据移至 GPU
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
            
            # --- 步骤 A: 准备稠密输入 (Teacher) ---
            grid_coord = data_dict.get("grid_coord", None)
            if grid_coord is None:
                # 兜底策略：如果没有 grid_coord，则传入 grid_size 让 Point 内部计算
                # 注意：这里的 0.1 必须与 config 中的 grid_size 一致
                grid_size = 0.1
            else:
                grid_size = 0.1

            dense_point = Point(
                coord=data_dict["coord"],
                feat=data_dict["feat"] if "feat" in data_dict else data_dict["coord"], 
                offset=data_dict["offset"],
                # [关键修改] 传入 grid_coord 或 grid_size
                grid_coord=data_dict["grid_coord"],
                grid_size=grid_size
            )
            
            # Teacher Forward
            dense_out = model.teacher.backbone(dense_point)
            if True: #hasattr(model.teacher, 'up_cast'):
                dense_out = model.teacher.up_cast(dense_out)
            feat_dense = model.teacher.mask_head(dense_out.feat)
            
            # --- 步骤 B: 准备稀疏输入 (Student) ---
            N = dense_point.coord.shape[0]
            keep_rate = 1.0 - args.drop_rate
            num_keep = int(N * keep_rate)
            if num_keep < 10: num_keep = 10
            
            # 随机 Drop
            perm = torch.randperm(N, device=dense_point.coord.device)
            keep_indices = perm[:num_keep]
            
            sparse_grid_coord = None
            if "grid_coord" in dense_point.keys():
                sparse_grid_coord = dense_point.grid_coord[keep_indices]

            sparse_point = Point(
                coord=dense_point.coord[keep_indices],
                feat=dense_point.feat[keep_indices],
                offset=torch.tensor([num_keep], device=dense_point.coord.device).int(),
                # 传入切片后的 grid_coord
                grid_coord=sparse_grid_coord,
                grid_size=dense_point.get("grid_size")
            )
            
            # Student Forward
            sparse_out = model.student.backbone(sparse_point)
            if True: #hasattr(model.student, 'up_cast'):
                sparse_out = model.student.up_cast(sparse_out)
            feat_sparse = model.student.mask_head(sparse_out.feat)
            
            # --- 步骤 C: 计算一致性误差 ---
            # 1. 寻找对应关系 (Sparse -> Dense)
            knn_idx, _ = pointops.knn_query(1, dense_point.coord, dense_point.offset, 
                                            sparse_point.coord, sparse_point.offset)
            target_idx = knn_idx.long().squeeze()
            
            # 2. 取出对应的 Dense 特征
            feat_dense_matched = feat_dense[target_idx]
            
            # 3. 计算 Cosine 误差
            feat_sparse_norm = F.normalize(feat_sparse, dim=-1)
            feat_dense_norm = F.normalize(feat_dense_matched, dim=-1)
            cos_sim = (feat_sparse_norm * feat_dense_norm).sum(dim=-1)
            error = 1.0 - cos_sim
            error = torch.clamp(error, 0.0, 1.0)
            
            # --- 步骤 D: 可视化生成 ---
            # 1. 误差热力图颜色 (Sparse点云)
            error_np = error.cpu().numpy()
            sparse_coord_np = sparse_point.coord.cpu().numpy()
            error_colors = get_jet_color(error_np)
            
            # 2. Dense 特征 PCA 颜色 (Dense点云)
            # 计算全量点云的特征分布
            pca_color_dense = get_pca_color(feat_dense, brightness=1.2, center=True)
            dense_coord_np = dense_point.coord.cpu().numpy()
            
            # 3. Sparse 特征 PCA 颜色 (Sparse点云)
            # 独立计算 Student 的特征分布，看其流形结构是否保留
            pca_color_sparse = get_pca_color(feat_sparse, brightness=1.2, center=True)
            
            # --- 步骤 E: 保存文件 ---
            scene_name = data_dict["name"][0] if "name" in data_dict else f"scene_{idx:04d}"
            
            # 保存路径
            file_error  = os.path.join(args.save_path, f"{scene_name}_error.ply")
            file_feat_d = os.path.join(args.save_path, f"{scene_name}_feat_dense.ply")
            file_feat_s = os.path.join(args.save_path, f"{scene_name}_feat_sparse.ply")
            
            write_ply_color(file_error, sparse_coord_np, error_colors)
            write_ply_color(file_feat_d, dense_coord_np, pca_color_dense)
            write_ply_color(file_feat_s, sparse_coord_np, pca_color_sparse)
            
            print(f"[{idx}] {scene_name}: Saved Error/DenseFeat/SparseFeat | Mean Error: {error_np.mean():.4f}")
            
            if idx >= 10:
                print("Processed 10 scenes, stopping.")
                break

if __name__ == "__main__":
    main()