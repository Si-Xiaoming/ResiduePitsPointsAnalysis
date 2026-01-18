import os
import time
import numpy as np
import torch
import pdal
from collections import OrderedDict

import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import intersection_and_union, make_dirs
from .defaults import create_ddp_model
from .test import TESTERS

@TESTERS.register_module()
class SemSegTesterLaz:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False, load_strict=True):
        self.cfg = cfg
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        
        # 1. Build Model
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model

        # 2. Build Dataset & Loader
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.dataset = build_dataset(cfg.data.test)
            # 这里的 batch_size 可以大于1，表示同时推理多个空间块
            self.test_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=cfg.batch_size_test_per_gpu, 
                shuffle=False,
                num_workers=cfg.num_worker_test,
                pin_memory=True,
                collate_fn=collate_fn, 
            )
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"=> Loading weight from: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location="cpu")
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    key = key[7:]
                weight[key] = value
            model.load_state_dict(weight, strict=True)
        return create_ddp_model(model.cuda(), broadcast_buffers=False)

    def test(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Spatial Tiling Inference >>>>>>>>>>>>>>>>")
        self.model.eval()
        
        save_path = os.path.join(self.cfg.save_path, "result_laz")
        make_dirs(save_path)

        # 获取全场点数，用于初始化全局结果容器
        # 由于我们只有一个文件，直接从 dataset 拿全场信息
        total_points = self.dataset.pos.shape[0]
        num_classes = self.cfg.data.num_classes
        
        self.logger.info(f"Total Scene Points: {total_points}")
        
        # 全局 Logits 累加器 (FP16节省内存，如果内存够大可用FP32)
        # 这里的 index 对应原始点云的行号
        global_logits = torch.zeros((total_points, num_classes), dtype=torch.float16, device='cpu')
        # 计数器，用于处理重叠区域的平均
        global_counts = torch.zeros((total_points), dtype=torch.int8, device='cpu')

        start_time = time.time()
        
        # 遍历所有空间块 (Blocks)
        for idx, input_dict in enumerate(self.test_loader):
            # input_dict 包含了一个或多个 Block 的数据
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            
            with torch.no_grad():
                output_dict = self.model(input_dict)
                pred_part = output_dict["seg_logits"] # (N_batch_points, C)
                
            # 将预测结果填回全局容器
            # input_dict["index"] 是我们在 Dataset 里塞进去的全局索引
            # 注意：GridSample 可能会打乱块内的顺序，或者丢弃点（如果用了 voxelize）
            # 但只要 transforms 里有 GridSample，它就会处理 coordinate 和 feature
            # 关键：我们必须确保 input_dict["index"] 能够对应上 pred_part 的每一行
            
            # 如果你在 transforms 里用了 GridSample(mode='test')，
            # 那么 input_dict["index"] 是被 GridSample 处理过后的索引，
            # 它直接指向原始点云的 ID，无需额外映射。
            
            indices = input_dict["index"].detach().cpu()
            preds = pred_part.detach().cpu().half()
            
            # 累加 Logits (Voting)
            # 注意处理索引越界或形状不匹配（理论上不应发生）
            global_logits[indices] += preds
            global_counts[indices] += 1
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed block batch {idx+1}/{len(self.test_loader)}")

        # --- 后处理与保存 ---
        self.logger.info("Merging blocks and saving...")
        
        # 避免除以0 (虽然有 overlap 应该都有值，但以防万一)
        global_counts[global_counts == 0] = 1
        # 实际上不需要除以 counts，argmax 结果是一样的，除非要算概率
        
        pred_labels = global_logits.argmax(dim=1).numpy().astype(np.int32)
        
        # 计算全场精度
        segment = self.dataset.segment
        if (segment != -1).any():
            intersection, union, target = intersection_and_union(
                pred_labels, segment, num_classes, ignore_index=-1
            )
            iou_class = intersection / (union + 1e-10)
            mIoU = np.mean(iou_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)
            self.logger.info(f"Final Whole Scene Results: mIoU={mIoU:.4f}, OA={allAcc:.4f}")

        # 保存为 LAZ
        out_name = os.path.basename(self.cfg.data.test.test_file)
        out_file = os.path.join(save_path, out_name)
        self.save_laz(out_file, self.dataset, pred_labels)
        
        self.logger.info(f"Total time: {time.time() - start_time:.2f}s")

    def save_laz(self, filename, dataset, pred_labels):
        self.logger.info(f"Writing LAZ to {filename}...")
        
        raw_pos = dataset.pos
        data_shift = dataset.data_shift
        
        x = (raw_pos[:, 0] + data_shift[0]).astype(np.float64)
        y = (raw_pos[:, 1] + data_shift[1]).astype(np.float64)
        z = (raw_pos[:, 2] + data_shift[2]).astype(np.float64)
        
        if dataset.color is not None:
            red = (dataset.color[:, 0] * 65535).astype(np.uint16)
            green = (dataset.color[:, 1] * 65535).astype(np.uint16)
            blue = (dataset.color[:, 2] * 65535).astype(np.uint16)
        else:
            red = green = blue = np.zeros_like(x, dtype=np.uint16)

        classification = pred_labels.astype(np.uint8)

        dtype = [
            ('X', np.float64), ('Y', np.float64), ('Z', np.float64),
            ('Red', np.uint16), ('Green', np.uint16), ('Blue', np.uint16),
            ('Classification', np.uint8)
        ]
        data = np.zeros(x.shape[0], dtype=dtype)
        data['X'] = x
        data['Y'] = y
        data['Z'] = z
        data['Red'] = red
        data['Green'] = green
        data['Blue'] = blue
        data['Classification'] = classification

        pipeline = pdal.Pipeline([data])
        
        writer_opts = {
            "filename": filename,
            "scale_x": dataset.header_scales[0],
            "scale_y": dataset.header_scales[1],
            "scale_z": dataset.header_scales[2],
            "offset_x": dataset.header_offsets[0],
            "offset_y": dataset.header_offsets[1],
            "offset_z": dataset.header_offsets[2],
        }
        if dataset.srs:
            writer_opts["a_srs"] = dataset.srs

        pipeline |= pdal.Writer.las(**writer_opts)
        pipeline.execute()