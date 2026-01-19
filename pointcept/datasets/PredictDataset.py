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
                 loop=1, # [新功能] 支持 TTA 循环次数
                 test_file=None,
                 block_size=50.0,
                 stride=25.0):
        
        # 1. 强制 test_mode=False，这样 transform 会被当作训练模式执行 (支持随机增强)
        super(PredictDataset, self).__init__(split=split, 
                                             data_root=data_root, 
                                             transform=transforms, 
                                             test_mode=False, 
                                             test_cfg=test_cfg, 
                                             cache=cache, 
                                             ignore_index=ignore_index, 
                                             loop=loop)

        self.test_file = test_file if test_file is not None else self.test_cfg.get('test_file')
        self.block_size = block_size if block_size is not None else self.test_cfg.get('block_size', 50.0)
        self.stride = stride if stride is not None else self.test_cfg.get('stride', 25.0)
        self.loop = loop # 记录循环次数

        if self.test_file is None:
            raise ValueError("PredictDataset requires 'test_file' argument.")

        # [自动修复] 强制修正 transform 参数，确保 GridSample 和 Collect 配合 TTA 工作
        if hasattr(self, 'transform') and hasattr(self.transform, 'transforms'):
            for t in self.transform.transforms:
                t_name = t.__class__.__name__
                if t_name == 'GridSample':
                    print(f"[Auto-Fix] Forcing GridSample to mode='train', return_inverse=True")
                    t.mode = 'train'
                    t.return_inverse = True
                if t_name == 'Collect':
                    keys = list(t.keys) if isinstance(t.keys, (list, tuple)) else [t.keys]
                    if "inverse" not in keys:
                        print(f"[Auto-Fix] Adding 'inverse' to Collect keys")
                        keys.append("inverse")
                        t.keys = keys

        self.srs = ''
        self.header_offsets = [0, 0, 0]
        self.header_scales = [1, 1, 1]
        self.pos = None
        self.color = None
        self.segment = None
        self.block_indices = [] 
        
        self.prepare_data()

    def prepare_data(self):
        print(f"Loading full data from {self.test_file} ...")
        
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=self.test_file, nosrs=True)
        pipeline |= pdal.Filter.stats(dimensions="Red,Green,Blue")
        pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")
        pipeline |= pdal.Filter.assign(value=[
            f"Classification = 0 WHERE Classification == 2",
            f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
            f"Classification = 2 WHERE Classification == 6",
            f"Classification = 3 WHERE Classification == 8"
        ])
        
        pipeline.execute()
        
        arrays = pipeline.arrays[0]
        metadata = pipeline.metadata['metadata']
        
        las_meta = metadata['readers.las']
        self.srs = las_meta.get('comp_spatialreference', '')
        self.header_scales = [las_meta['scale_x'], las_meta['scale_y'], las_meta['scale_z']]
        self.header_offsets = [las_meta['offset_x'], las_meta['offset_y'], las_meta['offset_z']]
        
        minx, miny, minz = las_meta['minx'], las_meta['miny'], las_meta['minz']
        self.data_shift = np.array([minx, miny, minz])
        
        self.pos = np.stack([
            arrays['X'] - minx,
            arrays['Y'] - miny,
            arrays['Z'] - minz
        ], axis=1).astype(np.float32)

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

        print(f"Partitioning scene with block_size={self.block_size}m, stride={self.stride}m ...")
        self.block_indices = self._split_scene(self.pos, self.block_size, self.stride)
        print(f"Generated {len(self.block_indices)} blocks. TTA Loop: {self.loop}x. Total samples: {len(self.block_indices) * self.loop}")

    def _split_scene(self, points, block_size, stride):
        coord_min = points.min(0)
        coord_max = points.max(0)
        grid_x = math.ceil((coord_max[0] - coord_min[0]) / stride)
        grid_y = math.ceil((coord_max[1] - coord_min[1]) / stride)
        
        block_list = []
        for i in range(grid_x):
            for j in range(grid_y):
                s_x = coord_min[0] + i * stride
                s_y = coord_min[1] + j * stride
                e_x = s_x + block_size
                e_y = s_y + block_size
                mask = (points[:, 0] >= s_x) & (points[:, 0] < e_x) & \
                       (points[:, 1] >= s_y) & (points[:, 1] < e_y)
                indices = np.where(mask)[0]
                if len(indices) > 100: 
                    block_list.append(indices)
        return block_list

    @staticmethod
    def normalize_attributes(array, min_v, max_v, mean_v, std_v):
        min_v = np.clip(mean_v - 2 * std_v, min_v, max_v)
        max_v = np.clip(mean_v + 2 * std_v, min_v, max_v)
        div = max_v - min_v
        if div == 0: div = 1.0
        val = (array.astype(np.float32) - min_v) / div
        val = np.clip(val, 0.0, 1.0)
        return np.expand_dims(val, 1)

    def get_data(self, idx):
        # [核心逻辑] 支持循环采样：不同的 idx 可能指向同一个 block，
        # 但后续的 transforms (如 RandomFlip) 每次调用都会产生不同的随机效果
        block_real_idx = idx % len(self.block_indices)
        block_idx_arr = self.block_indices[block_real_idx]
        
        data_dict = {}
        data_dict["coord"] = self.pos[block_idx_arr]
        data_dict["color"] = self.color[block_idx_arr]
        data_dict["segment"] = self.segment[block_idx_arr]
        data_dict["index"] = block_idx_arr.astype(np.int64) 
        data_dict["name"] = os.path.basename(self.test_file)
        return data_dict

    def __len__(self):
        # 数据集总长度 = 块数量 * 循环次数
        return len(self.block_indices) * self.loop