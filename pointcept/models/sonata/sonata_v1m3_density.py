"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.models.sonata.sonata_v1m2_uni_teacher_head import Sonata


import torch
import torch_scatter
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.models.modules import Point


class GenericDensityAugmentor(nn.Module):
    def __init__(
            self,
            num_density_views=2,  # 减少密度视图数量，从3减到2
            min_ratio=0.3,
            max_ratio=3.0,
            prob_anisotropic=0.3
    ):
        super().__init__()
        self.num_views = num_density_views
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob_anisotropic = prob_anisotropic

    def forward(self, point):
        """
        生成不同相对密度的点云视图
        不依赖绝对密度计算，通过相对比例缩放原始点数量实现
        """
        batch = offset2batch(point.offset)  # [N]
        num_points_per_batch = torch.bincount(batch)  # 每个批次的原始点数量
        unique_batches = torch.unique(batch)  # 唯一批次索引
        density_views = []

        for _ in range(self.num_views):
            # 为每个批次随机生成密度比例（相对于原始密度）
            ratios = torch.rand(len(unique_batches), device=point.coord.device)
            ratios = ratios * (self.max_ratio - self.min_ratio) + self.min_ratio

            # 各向异性采样（可选）
            if self.training and torch.rand(1) < self.prob_anisotropic:
                sampled_indices = self._anisotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)
            else:
                sampled_indices = self._isotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)

            # 构建新密度视图 - 只保留必要的字段
            dense_view = Point({
                "feat": point.feat[sampled_indices],
                "coord": point.coord[sampled_indices],
                "origin_coord": point.origin_coord[sampled_indices],
                "offset": batch2offset(batch[sampled_indices]),
                "grid_size": point.grid_size
            })
            density_views.append(dense_view)

        return density_views

    def _isotropic_sample(self, point, batch, unique_batches, num_points_per_batch, ratios):
        """各向同性采样：均匀降低/增加所有方向的点密度"""
        sampled_points = []

        for i, b in enumerate(unique_batches):
            # 获取当前 batch 的所有点索引
            mask = batch == b
            indices_in_batch = torch.where(mask)[0]

            # 根据比例计算采样数量（至少保留50个点）
            num_sample = max(50, int(ratios[i] * num_points_per_batch[b]))

            # 随机采样
            if num_sample >= len(indices_in_batch):
                selected_indices = indices_in_batch
            else:
                # 使用更高效的随机采样方法
                rand_indices = torch.randperm(len(indices_in_batch), device=point.coord.device, dtype=torch.int64)[:num_sample]
                selected_indices = indices_in_batch[rand_indices]

            sampled_points.append(selected_indices)

        return torch.cat(sampled_points, dim=0)

    def _anisotropic_sample(self, point, batch, unique_batches, num_points_per_batch, ratios):
        """各向异性采样：沿某一轴方向非均匀采样"""
        sampled_points = []
        axis = torch.randint(0, 3, (1,)).item()

        for i, b in enumerate(unique_batches):
            mask = batch == b
            indices_in_batch = torch.where(mask)[0]
            batch_points = point.coord[indices_in_batch]

            # 沿选定轴排序
            sorted_indices_local = torch.argsort(batch_points[:, axis])
            sorted_indices_global = indices_in_batch[sorted_indices_local]

            num_sample = max(50, int(ratios[i] * num_points_per_batch[b]))

            # 简化非均匀采样逻辑
            if ratios[i] < 1.0:
                # 降采样时使用均匀间隔采样
                step = max(1, len(sorted_indices_local) // num_sample)
                selected = sorted_indices_global[::step][:num_sample]
            else:
                # 升采样时使用重复采样
                indices = torch.linspace(0, len(sorted_indices_local) - 1, num_sample,
                                         device=point.coord.device, dtype=torch.int64)
                indices = indices % len(sorted_indices_local)
                selected = sorted_indices_global[indices]

            sampled_points.append(selected)

        return torch.cat(sampled_points, dim=0)


@MODELS.register_module("Sonata-v1m2-MD-Generic")
class SonataMultiDensityGeneric(Sonata):
    def __init__(
            self,
            *args,
            num_density_views=2,  # 减少密度视图数量
            density_min_ratio=0.3,
            density_max_ratio=3.0,
            density_anisotropic_prob=0.3,
            cross_density_weight_start=0.2,
            cross_density_weight=1.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # 初始化通用密度增强器
        self.density_aug = GenericDensityAugmentor(
            num_density_views=num_density_views,
            min_ratio=density_min_ratio,
            max_ratio=density_max_ratio,
            prob_anisotropic=density_anisotropic_prob
        )
        self.cross_density_loss = CrossDensityLoss()
        self.cross_density_weight = cross_density_weight
        self.cross_density_weight_start = cross_density_weight_start

    def before_train(self):
        super().before_train()
        # 密度损失权重调度器
        total_steps = self.trainer.cfg.scheduler.total_steps
        self.density_weight_scheduler = CosineScheduler(
            start_value=self.cross_density_weight_start,
            base_value=self.cross_density_weight,
            final_value=self.cross_density_weight,
            total_iters=total_steps
        )

    def before_step(self):
        super().before_step()
        self.current_density_weight = self.density_weight_scheduler.step()

    def forward(self, data_dict, return_point=False):
        if return_point:
            return super().forward(data_dict, return_point)

        # 1. 生成多密度视图
        global_point = Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"],
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        density_views = self.density_aug(global_point)

        # 2. 原有损失计算
        base_result = super().forward(data_dict)

        # 3. 跨密度一致性损失计算 - 优化内存使用
        if self.training:  # 只在训练时计算
            with torch.no_grad():
                teacher_feats = []
                teacher_coords = []
                for view in density_views:
                    teacher_out = self.up_cast(self.teacher.backbone(view))
                    teacher_feats.append(teacher_out["feat"])
                    teacher_coords.append(teacher_out["coord"])

            student_feats = []
            student_coords = []
            for view in density_views:
                student_out = self.up_cast(self.student.backbone(view))
                student_feats.append(student_out["feat"])
                student_coords.append(student_out["coord"])

            # 计算跨密度损失
            cross_loss = self.cross_density_loss(student_feats, student_coords)
            with torch.no_grad():
                teacher_cross_loss = self.cross_density_loss(teacher_feats, teacher_coords)
            cross_loss = (cross_loss + teacher_cross_loss) * 0.5

            # 合并损失
            base_result["cross_density_loss"] = cross_loss
            base_result["loss"]+=(cross_loss * self.current_density_weight)

        return base_result


class CrossDensityLoss(nn.Module):
    """
    简化的跨密度视图特征一致性损失
    减少计算复杂度和内存占用
    """

    def __init__(
            self,
            temp=0.1,
            match_max_k=4,  # 减少KNN邻居数量，从8减到4
            use_sinkhorn=False,  # 默认不使用Sinkhorn-Knopp算法
    ):
        super().__init__()
        self.temp = temp
        self.match_max_k = match_max_k
        self.use_sinkhorn = use_sinkhorn

    def forward(self, feat_list, coord_list, offset_list=None):
        """
        Args:
            feat_list: 不同密度视图的特征列表
            coord_list: 不同密度视图的坐标列表
            offset_list: 不同密度视图的offset列表
        Returns:
            跨密度视图一致性损失
        """
        total_loss = 0.0
        num_views = len(feat_list)

        # 只计算相邻视图对之间的损失，减少计算量
        for i in range(num_views - 1):
            j = i + 1
            offset_i = offset_list[i] if offset_list is not None else None
            offset_j = offset_list[j] if offset_list is not None else None

            loss_ij = self._view_pair_loss(
                feat_i=feat_list[i],
                coord_i=coord_list[i],
                feat_j=feat_list[j],
                coord_j=coord_list[j],
                offset_i=offset_i,
                offset_j=offset_j
            )
            total_loss += loss_ij

        # 平均所有视图对的损失
        return total_loss / max(1, num_views - 1)

    def _view_pair_loss(self, feat_i, coord_i, feat_j, coord_j, offset_i=None, offset_j=None):
        """计算两个视图之间的跨密度损失"""
        # 特征归一化
        feat_i = F.normalize(feat_i, dim=1)
        feat_j = F.normalize(feat_j, dim=1)

        # 为没有提供offset的情况自动生成
        if offset_i is None:
            offset_i = torch.tensor([0, coord_i.size(0)], device=coord_i.device, dtype=torch.int32)
        if offset_j is None:
            offset_j = torch.tensor([0, coord_j.size(0)], device=coord_j.device, dtype=torch.int32)

        # 使用pointops进行KNN查询
        idx_j, _ = pointops.knn_query(
            self.match_max_k,
            coord_j.contiguous().float(),
            offset_j.contiguous().int(),
            coord_i.contiguous().float(),
            offset_i.contiguous().int()
        )

        # 获取匹配点的特征
        feat_j_matched = pointops.grouping(idx_j.contiguous(), feat_j.contiguous(), coord_j.contiguous())

        # 计算特征相似性
        sim_matrix = torch.einsum("nc,nkc->nk", feat_i, feat_j_matched)
        sim_matrix = sim_matrix / self.temp

        # 简化匹配策略：使用softmax直接匹配
        if self.use_sinkhorn:
            q_i = self.sinkhorn_knopp(sim_matrix, temp=1.0)
        else:
            q_i = F.softmax(sim_matrix, dim=1)

        # 计算InfoNCE损失
        loss_i = -torch.log(torch.sum(q_i * F.softmax(sim_matrix, dim=1), dim=1) + 1e-12)
        loss_i = loss_i.mean()

        # 对称计算损失（j->i）
        idx_i, _ = pointops.knn_query(
            self.match_max_k,
            coord_i.contiguous().float(),
            offset_i.contiguous().int(),
            coord_j.contiguous().float(),
            offset_j.contiguous().int()
        )

        feat_i_matched = pointops.grouping(idx_i.contiguous(), feat_i.contiguous(), coord_i.contiguous())
        sim_matrix_j = torch.einsum("nc,nkc->nk", feat_j, feat_i_matched)
        sim_matrix_j = sim_matrix_j / self.temp

        if self.use_sinkhorn:
            q_j = self.sinkhorn_knopp(sim_matrix_j, temp=1.0)
        else:
            q_j = F.softmax(sim_matrix_j, dim=1)

        loss_j = -torch.log(torch.sum(q_j * F.softmax(sim_matrix_j, dim=1), dim=1) + 1e-12)
        loss_j = loss_j.mean()

        return (loss_i + loss_j) * 0.5

    @staticmethod
    def sinkhorn_knopp(feat, temp=1.0, num_iter=2):  # 减少迭代次数
        """简化的Sinkhorn-Knopp算法"""
        feat = feat.float()
        q = torch.exp(feat / temp).t()  # [K, N]

        # 归一化
        sum_q = q.sum()
        if get_world_size() > 1:
            torch.distributed.all_reduce(sum_q)
        q = q / sum_q

        for _ in range(num_iter):
            # 行归一化
            sum_r = torch.sum(q, dim=1, keepdim=True)
            if get_world_size() > 1:
                torch.distributed.all_reduce(sum_r)
            q = q / sum_r

            # 列归一化
            sum_c = torch.sum(q, dim=0, keepdim=True)
            q = q / sum_c

        return q.t()