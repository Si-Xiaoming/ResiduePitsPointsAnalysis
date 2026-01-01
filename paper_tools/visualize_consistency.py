import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pointcept dependencies
from pointcept.utils.config import Config, DictAction
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn 
from pointcept.models.utils.structure import Point
import pointops # We need pointops for KNN

def get_jet_color(value):
    cmap = plt.get_cmap('jet')
    rgba = cmap(value)
    return (rgba[:, :3] * 255).astype(np.uint8)

def get_pca_color(feat, brightness=1.25, center=True):
    try:
        u, s, v = torch.pca_lowrank(feat, center=center, q=3, niter=5)
    except Exception as e:
        # print(f"Warning: PCA failed ({e}), using random projection")
        v = torch.randn(feat.shape[1], 3, device=feat.device)
        v = F.normalize(v, dim=0)

    projection = feat @ v
    min_val = projection.min(dim=0, keepdim=True)[0]
    max_val = projection.max(dim=0, keepdim=True)[0]
    div = max_val - min_val
    div[div < 1e-6] = 1.0 
    color = (projection - min_val) / div
    color = color * brightness
    color = color.clamp(0.0, 1.0)
    return (color.cpu().numpy() * 255).astype(np.uint8)

def write_ply_color(save_path, points, colors):
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
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("=> Loaded successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="/workspace/codes/ResiduePitsPointsAnalysis/configs/on_sbatch/vis_consist.py")
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--checkpoint", default="/workspace/datasets/checkpoints/residue/epoch_10.pth")
    parser.add_argument("--save-path", default="vis_results")
    parser.add_argument("--drop-rate", type=float, default=0.8)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg.batch_size = 1
    cfg.num_worker = 0
    
    print(f"=> Building model: {cfg.model.type}")
    model = build_model(cfg.model)
    model = model.cuda()
    model.eval()

    load_checkpoint(model, args.checkpoint)

    print(f"=> Building dataset: {cfg.data.val.type}")
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

    GRID_SIZE = 0.1 
    print("=> Start processing...")

    with torch.no_grad():
        for idx, data_dict in enumerate(loader):
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
            
            # --- 1. Teacher (Dense) ---
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
            
            # Use the sorted features from the backbone
            feat_dense_all = model.teacher.mask_head(dense_out.feat)
            
            # --- 2. Student (Sparse) ---
            N = dense_point.coord.shape[0]
            keep_rate = 1.0 - args.drop_rate
            num_keep = int(N * keep_rate)
            if num_keep < 10: num_keep = 10
            
            perm = torch.randperm(N, device=dense_point.coord.device)
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
            
            # --- 3. Robust Matching (KNN) ---
            # IMPORTANT: We must match sparse_out.coord (Sorted Student) 
            # to dense_out.coord (Sorted Teacher)
            # dense_out.coord is source (N), sparse_out.coord is query (M)
            knn_idx, _ = pointops.knn_query(1, dense_out.coord, dense_out.offset, 
                                            sparse_out.coord, sparse_out.offset)
            target_idx = knn_idx.long().squeeze()
            
            # Retrieve the matched features from the SORTED Teacher features
            feat_dense_matched = feat_dense_all[target_idx]
            
            # --- 4. Verify Shapes ---
            # Now both should be size (num_keep_sorted, C)
            if feat_sparse.shape[0] != feat_dense_matched.shape[0]:
                print(f"ERROR: Shape mismatch! Sparse: {feat_sparse.shape}, Matched: {feat_dense_matched.shape}")
                continue

            # --- 5. Calculation ---
            feat_sparse_norm = F.normalize(feat_sparse, dim=-1)
            feat_dense_norm = F.normalize(feat_dense_matched, dim=-1)
            cos_sim = (feat_sparse_norm * feat_dense_norm).sum(dim=-1)
            error = 1.0 - cos_sim
            error = torch.clamp(error, 0.0, 1.0)
            
            # --- 6. Visualization ---
            error_np = error.cpu().numpy()
            # Use the coords from sparse_out/dense_out because they match the feature order
            sparse_coord_np = sparse_out.coord.cpu().numpy()
            dense_coord_np = dense_out.coord.cpu().numpy()
            
            error_colors = get_jet_color(error_np)
            pca_color_dense = get_pca_color(feat_dense_all, brightness=1.2, center=True)
            pca_color_sparse = get_pca_color(feat_sparse, brightness=1.2, center=True)
            
            scene_name = data_dict["name"][0] if "name" in data_dict else f"scene_{idx:04d}"
            
            file_error  = os.path.join(args.save_path, f"{scene_name}_error.ply")
            file_feat_d = os.path.join(args.save_path, f"{scene_name}_feat_dense.ply")
            file_feat_s = os.path.join(args.save_path, f"{scene_name}_feat_sparse.ply")
            
            write_ply_color(file_error, sparse_coord_np, error_colors)
            write_ply_color(file_feat_d, dense_coord_np, pca_color_dense)
            write_ply_color(file_feat_s, sparse_coord_np, pca_color_sparse)
            
            print(f"[{idx}] {scene_name}: Saved with Mean Error: {error_np.mean():.4f}")
            
            if idx >= 10:
                break

if __name__ == "__main__":
    main()