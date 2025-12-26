import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pointcept 依赖
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.utils.checkpoint import CheckpointLoader
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
import pointops

def get_jet_color(value):
    """
    将 0-1 的误差值映射为 Jet 颜色 (蓝 -> 红)
    0 (一致性高/误差小) -> 蓝色
    1 (一致性低/误差大) -> 红色
    """
    cmap = plt.get_cmap('jet')
    # value shape: (N,)
    rgba = cmap(value)
    return (rgba[:, :3] * 255).astype(np.uint8)

def write_ply_color(save_path, points, colors):
    """
    保存带颜色的点云为 PLY
    points: (N, 3)
    colors: (N, 3)
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
    
    # 强制设置 batch_size 为 1 以便可视化
    cfg.batch_size = 1
    cfg.num_worker = 0
    
    # 2. 模型构建
    print(f"=> Building model: {cfg.model.type}")
    model = build_model(cfg.model)
    model = model.cuda()
    model.eval()

    # 加载权重
    CheckpointLoader.load_checkpoint(model, args.checkpoint)

    # 3. 数据集构建
    # 注意：这里我们使用 val 或 test 集
    print(f"=> Building dataset: {cfg.data.train.type}")
    # 为了方便，直接复用 train 的 dataset定义，但指向 processed/test (需确保config里路径对)
    # 或者如果 config 里有 test dataset 定义更好
    dataset = build_dataset(cfg.data.test) 
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=dataset.collate_fn
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("=> Start generating consistency heatmaps...")

    with torch.no_grad():
        for idx, data_dict in enumerate(loader):
            # 将数据移至 GPU
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
            
            # --- 步骤 A: 准备稠密输入 (Teacher) ---
            dense_point = Point(
                coord=data_dict["coord"],
                feat=data_dict["feat"] if "feat" in data_dict else data_dict["coord"], # 假设 feat=coord+color
                offset=data_dict["offset"]
            )
            
            # 提取稠密特征
            # 注意：这里我们取 Backbone 输出 + Mask Head (投影后特征)
            dense_out = model.backbone(dense_point)
            if hasattr(model, 'up_cast'):
                dense_out = model.up_cast(dense_out)
            # 使用 Mask Head 将特征投影到语义空间
            feat_dense = model.mask_head(dense_out.feat)
            
            # --- 步骤 B: 准备稀疏输入 (Student) ---
            # 手动模拟 Drop
            N = dense_point.coord.shape[0]
            # 保留比例
            keep_rate = 1.0 - args.drop_rate
            num_keep = int(N * keep_rate)
            if num_keep < 10: num_keep = 10
            
            # 随机采样
            perm = torch.randperm(N, device=dense_point.coord.device)
            keep_indices = perm[:num_keep]
            
            sparse_point = Point(
                coord=dense_point.coord[keep_indices],
                feat=dense_point.feat[keep_indices],
                offset=torch.tensor([num_keep], device=dense_point.coord.device).int()
            )
            
            # 提取稀疏特征
            sparse_out = model.backbone(sparse_point)
            if hasattr(model, 'up_cast'):
                sparse_out = model.up_cast(sparse_out)
            feat_sparse = model.mask_head(sparse_out.feat)
            
            # --- 步骤 C: 计算特征一致性误差 ---
            # 1. 找到 Sparse 点对应的 Dense 点索引 (KNN k=1)
            # Query: Sparse, Support: Dense
            knn_idx, _ = pointops.knn_query(1, dense_point.coord, dense_point.offset, 
                                            sparse_point.coord, sparse_point.offset)
            target_idx = knn_idx.long().squeeze()
            
            # 2. 取出对应的 Dense 特征
            feat_dense_matched = feat_dense[target_idx]
            
            # 3. 归一化特征
            feat_sparse_norm = F.normalize(feat_sparse, dim=-1)
            feat_dense_norm = F.normalize(feat_dense_matched, dim=-1)
            
            # 4. 计算余弦相似度 (Cosine Similarity)
            # Range: [-1, 1], 1 is best
            cos_sim = (feat_sparse_norm * feat_dense_norm).sum(dim=-1)
            
            # 5. 转换为误差 (Consistency Error)
            # Error = 1 - CosSim. 范围 [0, 2]. 0 表示完全一致.
            # 为了可视化好看，我们截断到 [0, 1] 区间，因为负相似度也是极差的
            error = 1.0 - cos_sim
            error = torch.clamp(error, 0.0, 1.0)
            
            # --- 步骤 D: 生成热力图并保存 ---
            # 转为 numpy
            error_np = error.cpu().numpy()
            sparse_coord_np = sparse_point.coord.cpu().numpy()
            
            # 获取颜色 (Jet: Blue=Low Error, Red=High Error)
            colors = get_jet_color(error_np)
            
            # 保存文件名
            scene_name = data_dict["name"][0] if "name" in data_dict else f"scene_{idx:04d}"
            save_file = os.path.join(args.save_path, f"{scene_name}_error_heatmap.ply")
            
            write_ply_color(save_file, sparse_coord_np, colors)
            
            print(f"[{idx}] Saved {save_file} | Mean Error: {error_np.mean():.4f}")
            
            # 仅处理前 10 个场景用于展示，避免跑太多
            if idx >= 10:
                break

if __name__ == "__main__":
    main()