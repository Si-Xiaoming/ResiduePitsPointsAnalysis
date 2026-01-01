import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Pointcept Dependencies
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn 
from pointcept.models.utils.structure import Point
import pointops

def compute_local_curvature(coord, offset, k=16):
    """
    Compute local geometric roughness/curvature using KNN variance.
    Input: coord (N, 3), offset (B)
    Output: curvature (N,)
    """
    # Self-query to find neighbors
    # Note: Ensure coords are contiguous for cuda ops
    coord = coord.float().contiguous()
    offset = offset.int().contiguous()
    
    idx, dist = pointops.knn_query(k, coord, offset, coord, offset)
    
    # Gather neighbor coordinates: (N, k, 3)
    neighbor_coords = coord[idx.long()]
    
    # Center neighbors around the query point
    centered = neighbor_coords - coord.unsqueeze(1)
    
    # Variance sum (approximate roughness)
    # (N, k, 3) -> var dim=1 -> (N, 3) -> sum dim=-1 -> (N,)
    curvature = torch.var(centered, dim=1).sum(dim=-1)
    
    return curvature

def load_checkpoint(model, filename):
    """Manual checkpoint loader to handle DDP prefixes and partial loading."""
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("=> Loaded successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/sonata/sonata_v1m1_resolution_navarra.py", help="path to config file")
    parser.add_argument("--checkpoint", default="exp/sonata_resolution/model_best.pth", help="path to checkpoint")
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument("--save-path", default="visualization/tsne_plots", help="path to save png")
    parser.add_argument("--drop-rate", type=float, default=0.9, help="Student view drop rate")
    parser.add_argument("--num-points-vis", type=int, default=2000, help="Total points to plot in t-SNE")
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
    print(f"=> Building model: {cfg.model.type}")
    model = build_model(cfg.model).cuda()
    model.eval()
    
    load_checkpoint(model, args.checkpoint)

    # 3. Build Dataset
    print(f"=> Building dataset...")
    # Prefer 'test' split, fallback to 'val'
    dataset_cfg = cfg.data.test if hasattr(cfg.data, 'test') else cfg.data.val
    dataset = build_dataset(dataset_cfg)
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    # Containers for t-SNE
    # Labels: 0=Dense_Flat, 1=Dense_Edge, 2=Sparse_Flat, 3=Sparse_Edge
    collected_feats = []
    collected_labels = []
    
    GRID_SIZE = 0.1 # Ensure this matches your config/training
    
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
                offset=data_dict["offset"],
                grid_coord=data_dict.get("grid_coord", None),
                grid_size=GRID_SIZE
            )
            
            # Teacher Forward (Output is SORTED/RE-ORDERED)
            dense_out = model.teacher.backbone(dense_point)
            if hasattr(model, 'up_cast'): 
                dense_out = model.up_cast(dense_out)
            
            # Get Teacher Features (N, C)
            feat_dense_all = model.teacher.mask_head(dense_out.feat)
            feat_dense_all = F.normalize(feat_dense_all, dim=-1)

            # --- B. Calculate Curvature on OUTPUT coordinates ---
            # We calculate on output coordinates to ensure 1-to-1 mapping with features
            curvature = compute_local_curvature(dense_out.coord, dense_out.offset)
            
            # Define thresholds (Top 10% as Edge, Bottom 50% as Flat)
            k_val = int(curvature.shape[0] * 0.1)
            if k_val < 1: k_val = 1
            
            _, top_k_idx = torch.topk(curvature, k_val) # Edge indices (in sorted list)
            _, bot_k_idx = torch.topk(curvature, k_val, largest=False) # Flat indices (in sorted list)
            
            edge_mask = torch.zeros_like(curvature, dtype=torch.bool)
            flat_mask = torch.zeros_like(curvature, dtype=torch.bool)
            edge_mask[top_k_idx] = True
            flat_mask[bot_k_idx] = True

            # --- C. Prepare Sparse (Student) ---
            N_input = dense_point.coord.shape[0]
            num_keep = int(N_input * (1 - args.drop_rate))
            if num_keep < 10: num_keep = 10
            
            # Random Drop on Input
            perm = torch.randperm(N_input, device=dense_point.coord.device)
            keep_indices = perm[:num_keep]
            
            sparse_grid_coord = None
            if "grid_coord" in dense_point.keys():
                sparse_grid_coord = dense_point.grid_coord[keep_indices]

            sparse_point = Point(
                coord=dense_point.coord[keep_indices],
                feat=dense_point.feat[keep_indices],
                offset=torch.tensor([num_keep], device=dense_point.coord.device).int(),
                grid_coord=sparse_grid_coord,
                grid_size=GRID_SIZE
            )
            
            # Student Forward (Output is also SORTED/RE-ORDERED)
            sparse_out = model.student.backbone(sparse_point)
            if hasattr(model, 'up_cast'): 
                sparse_out = model.up_cast(sparse_out)
            
            feat_sparse = model.student.mask_head(sparse_out.feat)
            feat_sparse = F.normalize(feat_sparse, dim=-1)

            # --- D. Robust Matching (KNN) ---
            # Match Sparse Output -> Dense Output to find correspondence
            knn_idx, _ = pointops.knn_query(1, dense_out.coord, dense_out.offset, 
                                            sparse_out.coord, sparse_out.offset)
            target_idx = knn_idx.long().squeeze()
            
            # --- E. Collection Loop ---
            # Iterate through Sparse points, find their Dense match, check if interesting (Edge/Flat)
            # target_idx[s_idx] gives the index in 'dense_out'
            
            # Boolean masks for Sparse points based on their matched Dense parent
            sparse_is_edge = edge_mask[target_idx]
            sparse_is_flat = flat_mask[target_idx]
            
            # Collect Edges
            valid_edge_s_indices = torch.where(sparse_is_edge)[0]
            if len(valid_edge_s_indices) > 50: 
                valid_edge_s_indices = valid_edge_s_indices[:50]
                
            for s_idx in valid_edge_s_indices:
                d_idx = target_idx[s_idx]
                
                # Pair: Dense (Reference) -> Label 0
                collected_feats.append(feat_dense_all[d_idx].cpu().numpy())
                collected_labels.append(0)
                
                # Pair: Sparse (Ours) -> Label 1
                collected_feats.append(feat_sparse[s_idx].cpu().numpy())
                collected_labels.append(1)

            # Collect Flats
            valid_flat_s_indices = torch.where(sparse_is_flat)[0]
            if len(valid_flat_s_indices) > 50: 
                valid_flat_s_indices = valid_flat_s_indices[:50]
                
            for s_idx in valid_flat_s_indices:
                d_idx = target_idx[s_idx]
                
                # Pair: Dense (Reference) -> Label 2
                collected_feats.append(feat_dense_all[d_idx].cpu().numpy())
                collected_labels.append(2)
                
                # Pair: Sparse (Ours) -> Label 3
                collected_feats.append(feat_sparse[s_idx].cpu().numpy())
                collected_labels.append(3)

            print(f"[{i}] Collected total {len(collected_feats)} points so far...")
            
            if len(collected_feats) >= args.num_points_vis:
                break

    if len(collected_feats) == 0:
        print("Error: No points collected. Check drop rate or thresholds.")
        return

    # --- F. Run t-SNE ---
    print(f"=> Running t-SNE on {len(collected_feats)} features...")
    X = np.array(collected_feats)
    y = np.array(collected_labels)
    
    # Init with PCA for stability
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)

    # --- G. Plotting ---
    plt.figure(figsize=(10, 8))
    
    styles = [
        {'mask': y==0, 'label': 'Dense Edge (Reference)', 'color': '#d62728', 'marker': 'o', 'alpha': 0.4, 's': 50}, 
        {'mask': y==1, 'label': 'Sparse Edge (Student)',  'color': '#d62728', 'marker': 'x', 'alpha': 1.0, 's': 60, 'linewidth': 2},
        {'mask': y==2, 'label': 'Dense Flat (Reference)', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.4, 's': 50}, 
        {'mask': y==3, 'label': 'Sparse Flat (Student)',  'color': '#1f77b4', 'marker': 'x', 'alpha': 1.0, 's': 60, 'linewidth': 2},
    ]

    for style in styles:
        if style['mask'].sum() > 0:
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

    # Draw lines connecting matched pairs
    print("=> Drawing connection lines...")
    num_pairs = len(X) // 2
    
    # Limit lines to avoid clutter
    draw_limit = 200 
    for i in range(0, min(num_pairs, draw_limit)):
        dense_idx = i * 2
        sparse_idx = i * 2 + 1
        
        # Color based on class (Red for Edge, Blue for Flat)
        line_color = '#d62728' if y[dense_idx] == 0 else '#1f77b4'
        
        plt.plot(
            [X_embedded[dense_idx, 0], X_embedded[sparse_idx, 0]],
            [X_embedded[dense_idx, 1], X_embedded[sparse_idx, 1]],
            color=line_color,
            alpha=0.15,
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