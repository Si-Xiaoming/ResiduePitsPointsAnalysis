import os
import numpy as np

# 定义 processed 下的 train 目录路径
dataset_path = "/home/shsi/datasets/Point_Cloud/residue/unused_pert100_gs01/processed/val"

def analyze_segments(base_path):
    print(f"{'Data Folder':<20} | {'Unique Labels'}")
    print("-" * 50)
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"Error: 路径 {base_path} 不存在。")
        return

    # 遍历 train 目录下的所有子文件夹 (例如 train_data_01, train_data_02 等)
    folders = sorted(os.listdir(base_path))
    
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name)
        
        # 确保是目录
        if os.path.isdir(folder_path):
            segment_file = os.path.join(folder_path, "segment.npy")
            
            if os.path.exists(segment_file):
                try:
                    # 加载 segment.npy 文件
                    segment_data = np.load(segment_file)
                    # 获取非重复标签
                    unique_labels = np.unique(segment_data)
                    
                    print(f"{folder_name:<20} | {unique_labels.tolist()}")
                except Exception as e:
                    print(f"{folder_name:<20} | 读取失败: {e}")
            else:
                print(f"{folder_name:<20} | 未找到 segment.npy")

if __name__ == "__main__":
    analyze_segments(dataset_path)