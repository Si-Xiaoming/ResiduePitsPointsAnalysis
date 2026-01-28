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
        
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model

        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.dataset = build_dataset(cfg.data.test)
            
            num_workers = cfg.get('num_worker_test', cfg.get('num_workers', 4))
            batch_size = cfg.get('batch_size_test_per_gpu', cfg.get('batch_size', 1))

            self.test_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
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
        else:
            self.logger.warning(f"=> No checkpoint found at '{self.cfg.weight}'. Using random init.")
            
        return create_ddp_model(model.cuda(), broadcast_buffers=False)

    def test(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start TTA Inference >>>>>>>>>>>>>>>>")
        self.model.eval()
        
        save_path = os.path.join(self.cfg.save_path, "result_laz")
        make_dirs(save_path)

        total_points = self.dataset.pos.shape[0]
        num_classes = self.cfg.data.num_classes
        
        self.logger.info(f"Total Scene Points: {total_points}")
        
        # 全局 Logits 累加器
        global_logits = torch.zeros((total_points, num_classes), dtype=torch.float16, device='cpu')
        global_counts = torch.zeros((total_points), dtype=torch.int16, device='cpu')

        start_time = time.time()
        
        for idx, input_dict in enumerate(self.test_loader):
            # 移除 segment 防止报错
            if "segment" in input_dict:
                del input_dict["segment"]

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            
            with torch.no_grad():
                output_dict = self.model(input_dict)
                pred_part = output_dict["seg_logits"] 
                
            if "inverse" in input_dict:
                inverse = input_dict["inverse"]
                pred_expanded = pred_part[inverse]
            else:
                pred_expanded = pred_part

            indices = input_dict["index"].detach().cpu()
            preds = pred_expanded.detach().cpu().half()
            
            if indices.shape[0] != preds.shape[0]:
                self.logger.warning(f"Shape mismatch: indices {indices.shape} vs preds {preds.shape}. Truncating.")
                min_len = min(indices.shape[0], preds.shape[0])
                indices = indices[:min_len]
                preds = preds[:min_len]

            # [投票核心]：多次循环的预测结果会累加到 global_logits 的同一位置
            global_logits[indices] += preds
            global_counts[indices] += 1
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed batch {idx+1}/{len(self.test_loader)}")
                
            if self.cfg.get('empty_cache', False):
                torch.cuda.empty_cache()

        self.logger.info("Merging TTA results and saving...")
        
        pred_labels = global_logits.argmax(dim=1).numpy().astype(np.int32)
        
        # 计算精度
        segment = self.dataset.segment
        valid_mask = (segment >= 0) & (segment < num_classes)
        if not valid_mask.any():
            self.logger.warning("Warning: No valid GT labels found. Skipping metrics.")
        else:
            eval_segment = np.full_like(segment, -1)
            eval_segment[valid_mask] = segment[valid_mask]
            
            if (eval_segment != -1).any():
                intersection, union, target = intersection_and_union(
                    pred_labels, eval_segment, num_classes, ignore_index=-1
                )
                iou_class = intersection / (union + 1e-10)
                mIoU = np.mean(iou_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)
                
                self.logger.info(f"Final TTA Results:")
                self.logger.info(f"  mIoU: {mIoU:.4f}")
                self.logger.info(f"  OA:   {allAcc:.4f}")
                self.logger.info(f"  IoU per class: {iou_class}")

        
        if hasattr(self.cfg.data.test, 'test_file'):
            src_filename = self.cfg.data.test.test_file
        else:
            src_filename = "output.laz"
            
        out_name = os.path.basename(src_filename).replace(".laz", "_pred.laz")
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

        pipeline = pdal.Pipeline(arrays=[data])
        
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