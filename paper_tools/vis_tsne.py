import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Pointcept 依赖
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.utils.checkpoint import CheckpointLoader
from pointcept.models.utils.structure import Point
import pointops

def compute_local_curvature(coord, offset, k=16):
    """
    计算局部几何粗糙度/曲率 (利用 KNN 方差)
    """
    # Self-query to find neighbors
    idx, dist = pointops.knn_query(k, coord.float(), offset.int(), coord.float(), offset.int())
    
    # Gather neighbor coordinates: (N, k, 3)
    neighbor_coords = coord[idx.long()]
    
    # Center neighbors around the query point
    centered = neighbor_coords - coord.unsqueeze(1)
    
    # Variance sum (Trace of covariance roughly) -> Roughness
    # (N, k, 3) -> var dim=1 -> (N, 3) -> sum dim=-1 -> (N,)
    curvature = torch.var(centered, dim=1).sum(dim=-1)
    
    return curvature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/sonata/sonata_v1m1_resolution_navarra.py", help="path to config file")
    parser.add_argument("--checkpoint", default="exp/sonata_resolution/model_best.pth", help="path to checkpoint")
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument("--save-path", default="visualization/tsne_plots", help="path to save png")
    parser.add_argument("--drop-rate", type=float, default=0.9, help="Extremely high drop rate to prove robustness")
    parser.add_argument("--num-points-vis", type=int, default=2000, help="Total points to plot in t-SNE to avoid clutter")
    args = parser.parse_args()

    # 1. Setup
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg.batch_size = 1
    cfg.num_worker = 0
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 2. Build Model
    print(f"=> Building model...")
    model = build_model(cfg.model).cuda()
    model.eval()
    CheckpointLoader.load_checkpoint(model, args.checkpoint)

    # 3. Build Dataset
    # 强制读取 test 目录的数据 (假设数据在 data_root/processed/test)
    # 也可以直接复用 cfg.data.val 或 cfg.data.test
    print(f"=> Building dataset...")
    if hasattr(cfg.data, 'test'):
        dataset = build_dataset(cfg.data.test)
    else:
        dataset = build_dataset(cfg.data.val) # Fallback to val if test not defined
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    # 容器：存储特征和标签
    # Labels: 0=Dense_Flat, 1=Dense_Edge, 2=Sparse_Flat, 3=Sparse_Edge
    collected_feats = []
    collected_labels = []
    
    print("=> Extracting features...")
    
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            # Move to GPU
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
            
            # --- A. Prepare Dense (Teacher) ---
            dense_point = Point(
                coord=data_dict["coord"],
                feat=data_dict["feat"] if "feat" in data_dict else data_dict["coord"],
                offset=data_dict["offset"]
            )
            
            # Extract Dense Features
            dense_out = model.backbone(dense_point)
            if hasattr(model, 'up_cast'): dense_out = model.up_cast(dense_out)
            # [关键] 取 Projection Head 之后的特征，那是对比学习发生的地方
            dense_feats_all = model.mask_head(dense_out.feat) 
            dense_feats_all = F.normalize(dense_feats_all, dim=-1)

            # --- B. Calculate Curvature (Distinguish Edge vs Flat) ---
            curvature = compute_local_curvature(dense_point.coord, dense_point.offset)
            # Define thresholds (Top 10% as Edge, Bottom 50% as Flat)
            k_val = int(curvature.shape[0] * 0.1)
            _, top_k_idx = torch.topk(curvature, k_val) # Edges/Canals
            _, bot_k_idx = torch.topk(curvature, k_val, largest=False) # Flat ground
            
            edge_mask = torch.zeros_like(curvature, dtype=torch.bool)
            flat_mask = torch.zeros_like(curvature, dtype=torch.bool)
            edge_mask[top_k_idx] = True
            flat_mask[bot_k_idx] = True

            # --- C. Prepare Sparse (Student) ---
            N = dense_point.coord.shape[0]
            num_keep = int(N * (1 - args.drop_rate))
            if num_keep < 10: num_keep = 10
            
            # Random Drop
            perm = torch.randperm(N, device=dense_point.coord.device)
            keep_indices = perm[:num_keep]
            
            sparse_point = Point(
                coord=dense_point.coord[keep_indices],
                feat=dense_point.feat[keep_indices],
                offset=torch.tensor([num_keep], device=dense_point.coord.device).int()
            )
            
            # Extract Sparse Features
            sparse_out = model.backbone(sparse_point)
            if hasattr(model, 'up_cast'): sparse_out = model.up_cast(sparse_out)
            sparse_feats_all = model.mask_head(sparse_out.feat)
            sparse_feats_all = F.normalize(sparse_feats_all, dim=-1)

            # --- D. Match & Collect ---
            # Match Sparse back to Dense to know which point is which
            idx, _ = pointops.knn_query(1, dense_point.coord, dense_point.offset, 
                                        sparse_point.coord, sparse_point.offset)
            target_idx = idx.long().squeeze() # Indices in Dense cloud corresponding to Sparse points
            
            # Filter: We only care about points that survived the drop AND are either Edge or Flat
            # Check if the surviving points were Edge or Flat in the original cloud
            is_edge = edge_mask[target_idx]
            is_flat = flat_mask[target_idx]
            
            # Select samples
            # 1. Edge Points
            valid_edge_indices = torch.where(is_edge)[0]
            if len(valid_edge_indices) > 50: # Limit samples per scene
                valid_edge_indices = valid_edge_indices[:50]
                
            for idx in valid_edge_indices:
                # Dense representation of this point
                collected_feats.append(dense_feats_all[target_idx[idx]].cpu().numpy())
                collected_labels.append(0) # 0 = Dense Edge (Reference)
                
                # Sparse representation of the SAME point
                collected_feats.append(sparse_feats_all[idx].cpu().numpy())
                collected_labels.append(1) # 1 = Sparse Edge (Should align with 0)

            # 2. Flat Points
            valid_flat_indices = torch.where(is_flat)[0]
            if len(valid_flat_indices) > 50:
                valid_flat_indices = valid_flat_indices[:50]

            for idx in valid_flat_indices:
                collected_feats.append(dense_feats_all[target_idx[idx]].cpu().numpy())
                collected_labels.append(2) # 2 = Dense Flat
                
                collected_feats.append(sparse_feats_all[idx].cpu().numpy())
                collected_labels.append(3) # 3 = Sparse Flat

            print(f"Collected points from scene {i}, Total points: {len(collected_feats)}")
            
            if len(collected_feats) >= args.num_points_vis:
                break

    # --- E. Run t-SNE ---
    print(f"=> Running t-SNE on {len(collected_feats)} features...")
    X = np.array(collected_feats)
    y = np.array(collected_labels)
    
    # Init with PCA for stability
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)

    # --- F. Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Define styles
    # Dense Edge: Red Circle
    # Sparse Edge: Red Cross (Should be near Red Circle)
    # Dense Flat: Blue Circle
    # Sparse Flat: Blue Cross (Should be near Blue Circle)
    
    styles = [
        {'mask': y==0, 'label': 'Dense Edge (Reference)', 'color': '#d62728', 'marker': 'o', 'alpha': 0.4, 's': 50}, # Red
        {'mask': y==1, 'label': 'Sparse Edge (Ours)',      'color': '#d62728', 'marker': 'x', 'alpha': 1.0, 's': 60, 'linewidth': 2},
        {'mask': y==2, 'label': 'Dense Flat (Reference)', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.4, 's': 50}, # Blue
        {'mask': y==3, 'label': 'Sparse Flat (Ours)',      'color': '#1f77b4', 'marker': 'x', 'alpha': 1.0, 's': 60, 'linewidth': 2},
    ]

    for style in styles:
        plt.scatter(
            X_embedded[style['mask'], 0], 
            X_embedded[style['mask'], 1], 
            label=style['label'],
            c=style['color'],
            marker=style['marker'],
            alpha=style['alpha'],
            s=style['s'],
            linewidths=style.get('linewidth', 1)
        )

    # Draw lines connecting Dense and Sparse pairs to visualize alignment explicitly
    # Since we added them in pairs (i, i+1), we can just iterate
    # Only draw lines for a subset to avoid mess
    print("=> Drawing connection lines...")
    num_pairs = len(X) // 2
    for i in range(0, min(num_pairs, 200)): # Draw first 200 lines
        dense_idx = i * 2
        sparse_idx = i * 2 + 1
        
        # Color based on class
        line_color = '#d62728' if y[dense_idx] == 0 else '#1f77b4'
        
        plt.plot(
            [X_embedded[dense_idx, 0], X_embedded[sparse_idx, 0]],
            [X_embedded[dense_idx, 1], X_embedded[sparse_idx, 1]],
            color=line_color,
            alpha=0.1, # Very faint lines
            linewidth=1
        )

    plt.title(f"Feature Alignment: Dense vs Sparse (Drop Rate {args.drop_rate})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_file = os.path.join(args.save_path, f"tsne_alignment_drop{args.drop_rate}.png")
    plt.savefig(save_file, dpi=300)
    print(f"=> Saved t-SNE plot to {save_file}")

if __name__ == "__main__":
    main()