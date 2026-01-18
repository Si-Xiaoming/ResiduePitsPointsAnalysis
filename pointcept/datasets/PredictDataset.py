import numpy as np
import torch
import os
import pdal
import math
from .defaults import DefaultDataset
from .builder import DATASETS

@DATASETS.register_module()
class PredictDataset(DefaultDataset):
    def __init__(self,
                 split="test",
                 data_root="",
                 transforms=None,
                 test_mode=True,
                 test_cfg=None,
                 cache=False,
                 ignore_index=-1,
                 loop=1):
        super(PredictDataset, self).__init__(split, data_root, transforms, test_mode,
                                             test_cfg, cache, ignore_index, loop)
        self.test_file = self.test_cfg.test_file
        # 分块参数
        self.block_size = self.test_cfg.get("block_size", 50.0) # 默认50米一块
        self.stride = self.test_cfg.get("stride", 25.0)         # 默认步长25米 (50%重叠)
        
        # 存储元数据
        self.srs = ''
        self.header_offsets = [0, 0, 0]
        self.header_scales = [1, 1, 1]
        
        # 数据容器 (全场数据常驻内存)
        self.pos = None
        self.color = None
        self.segment = None
        
        # 分块索引列表
        self.block_indices = [] 
        
        self.prepare_data()

    def prepare_data(self):
        print(f"Loading full data from {self.test_file} ...")
        
        # 1. 读取完整点云 (仅读取 XYZ 和 颜色，不进行下采样)
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=self.test_file, nosrs=True)
        pipeline |= pdal.Filter.stats(dimensions="Red,Green,Blue")
        pipeline.execute()
        
        arrays = pipeline.arrays[0]
        metadata = pipeline.metadata['metadata']
        
        # 保存头文件信息
        las_meta = metadata['readers.las']
        self.srs = las_meta.get('comp_spatialreference', '')
        self.header_scales = [las_meta['scale_x'], las_meta['scale_y'], las_meta['scale_z']]
        self.header_offsets = [las_meta['offset_x'], las_meta['offset_y'], las_meta['offset_z']]
        
        # 坐标中心化 (保留原始偏移量以便还原)
        minx, miny, minz = las_meta['minx'], las_meta['miny'], las_meta['minz']
        self.data_shift = np.array([minx, miny, minz])
        
        self.pos = np.stack([
            arrays['X'] - minx,
            arrays['Y'] - miny,
            arrays['Z'] - minz
        ], axis=1).astype(np.float32)

        # 颜色处理
        stats = metadata['filters.stats']['statistic']
        def get_stat(name):
            for s in stats:
                if name in s['name']: return s
            return None
        
        r_stat = get_stat("Red")
        g_stat = get_stat("Green")
        b_stat = get_stat("Blue")
        
        if r_stat and g_stat and b_stat:
            r = self.normalize_attributes(arrays['Red'], 0.0, r_stat['maximum'], r_stat['average'], r_stat['stddev'])
            g = self.normalize_attributes(arrays['Green'], 0.0, g_stat['maximum'], g_stat['average'], g_stat['stddev'])
            b = self.normalize_attributes(arrays['Blue'], 0.0, b_stat['maximum'], b_stat['average'], b_stat['stddev'])
            self.color = np.concatenate([r, g, b], axis=-1).astype(np.float32)
        else:
            self.color = np.ones_like(self.pos) * 0.5

        if 'Classification' in arrays.dtype.names:
            self.segment = arrays['Classification'].astype(np.int32).reshape(-1)
        else:
            self.segment = np.zeros(self.pos.shape[0], dtype=np.int32) - 1

        # 2. 生成空间分块 (Spatial Partitioning)
        print(f"Partitioning scene with block_size={self.block_size}m, stride={self.stride}m ...")
        self.block_indices = self._split_scene(self.pos, self.block_size, self.stride)
        print(f"Generated {len(self.block_indices)} blocks.")

    def _split_scene(self, points, block_size, stride):
        """生成分块的索引列表"""
        coord_min = points.min(0)
        coord_max = points.max(0)
        
        # 计算 X 和 Y 维度的网格数量
        grid_x = math.ceil((coord_max[0] - coord_min[0]) / stride)
        grid_y = math.ceil((coord_max[1] - coord_min[1]) / stride)
        
        block_list = []
        
        for i in range(grid_x):
            for j in range(grid_y):
                # 当前 Block 的边界
                # 注意：我们使用 center 方式或者 min-max 方式均可，这里用 min-max
                s_x = coord_min[0] + i * stride
                s_y = coord_min[1] + j * stride
                e_x = s_x + block_size
                e_y = s_y + block_size
                
                # 找出在当前 Block 范围内的点索引
                # 使用 numpy 的逻辑操作，速度很快
                mask = (points[:, 0] >= s_x) & (points[:, 0] < e_x) & \
                       (points[:, 1] >= s_y) & (points[:, 1] < e_y)
                
                indices = np.where(mask)[0]
                
                # 只有非空的块才加入列表
                if len(indices) > 100: # 过滤掉点数太少的边缘块
                    block_list.append(indices)
                    
        return block_list

    @staticmethod
    def normalize_attributes(array, min_v, max_v, mean_v, std_v):
        min_v = np.clip(mean_v - 2 * std_v, min_v, max_v)
        max_v = np.clip(mean_v + 2 * std_v, min_v, max_v)
        # 避免除以零
        div = max_v - min_v
        if div == 0: div = 1.0
        val = (array.astype(np.float32) - min_v) / div
        val = np.clip(val, 0.0, 1.0)
        return np.expand_dims(val, 1)

    def get_data(self, idx):
        # idx 对应的是 self.block_indices 中的第几个块
        block_idx = self.block_indices[idx]
        
        data_dict = {}
        # 截取对应块的数据
        data_dict["coord"] = self.pos[block_idx]
        data_dict["color"] = self.color[block_idx]
        data_dict["segment"] = self.segment[block_idx]
        
        # 关键：保留全局索引 (Global Index)
        # 这样在推理完这个小块后，我们知道这些预测结果属于原始大点云的哪些点
        data_dict["index"] = block_idx.astype(np.int64) 
        
        data_dict["name"] = os.path.basename(self.test_file)
        return data_dict

    def __len__(self):
        # 数据集长度等于块的数量
        return len(self.block_indices)