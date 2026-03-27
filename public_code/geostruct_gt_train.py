# geostruct_gt_train.py
"""
带图约束 Loss 的 gDSA 训练脚本

在原有 Loss 基础上增加结构约束惩罚：
1. 反对称约束 Loss: 惩罚 Up(A,B) 和 Up(B,A) 同时为高分
2. 树约束 Loss: 惩罚一个节点有多个 Parent
3. DAG 约束 Loss: 惩罚 Sequence 中的环

使用方法:
    python geostruct_gt_train.py \
        --yolo yolo11n.pt \
        --epochs 100 \
        --constraint-weight 0.1
"""

import os
import json
import time
import signal
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from geostruct_gt_dataset import YOLOgDSADataset
from geostruct_gt_model import YOLOgDSATransformerVisualOnly
from geostruct_gt_evaluator import gDSAEvaluator as gDSAEvaluatorNew


# Backbone 映射
BACKBONE_MAP = {
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo11l': 'yolo11l.pt',
    'yolo11x': 'yolo11x.pt',
    'resnet50': 'ultralytics/cfg/models/11/yolo11-resnet50.yaml',
    'convnext': 'ultralytics/cfg/models/11/yolo11-convnext.yaml',
    'swin': 'ultralytics/cfg/models/11/yolo11-swin.yaml',
    'efficientnet': 'ultralytics/cfg/models/11/yolo11-efficientnet.yaml',
    'mobilenet': 'ultralytics/cfg/models/11/yolo11-mobilenet.yaml',
}


# ============ 约束 Loss 模块 ============

class ConstraintLoss(nn.Module):
    """
    图结构约束 Loss（向量化版本，GPU 加速）
    
    惩罚违反文档关系图约束的预测：
    1. 反对称约束: Up/Down, Left/Right 不能同时存在于 (i,j) 和 (j,i)
    2. 单父约束: 每个节点最多一个 Parent
    3. DAG 约束: Sequence 关系不能形成 2-环
    """
    
    # 关系类型
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    PARENT, CHILD, SEQUENCE, REFERENCE = 4, 5, 6, 7
    
    def __init__(self, 
                 antisym_weight=1.0,
                 tree_weight=1.0, 
                 dag_weight=0.5):
        super().__init__()
        self.antisym_weight = antisym_weight
        self.tree_weight = tree_weight
        self.dag_weight = dag_weight
    
    def forward(self, rel_probs, edge_index, num_nodes):
        """
        Args:
            rel_probs: [E, num_relations] 关系概率 (sigmoid 后)
            edge_index: [2, E] 边索引
            num_nodes: 节点数量
            
        Returns:
            constraint_loss: 约束违反惩罚
            loss_dict: 各项 loss 的字典
        """
        E = rel_probs.size(0)
        device = rel_probs.device
        
        if E == 0 or num_nodes < 2:
            return torch.tensor(0.0, device=device), {}
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 预计算反向边映射（向量化）
        reverse_idx = self._compute_reverse_edge_idx(edge_index, num_nodes)
        
        # 1. 反对称约束 Loss（向量化）
        antisym_loss = self._antisymmetric_loss_vectorized(rel_probs, reverse_idx)
        loss_dict['antisym'] = antisym_loss.item()
        total_loss = total_loss + self.antisym_weight * antisym_loss
        
        # 2. 单父约束 Loss（已经是向量化的）
        tree_loss = self._tree_constraint_loss(rel_probs, edge_index, num_nodes)
        loss_dict['tree'] = tree_loss.item()
        total_loss = total_loss + self.tree_weight * tree_loss
        
        # 3. DAG 约束 Loss（向量化）
        if self.dag_weight > 0:
            dag_loss = self._dag_constraint_loss_vectorized(rel_probs, reverse_idx)
            loss_dict['dag'] = dag_loss.item()
            total_loss = total_loss + self.dag_weight * dag_loss
        
        return total_loss, loss_dict
    
    def _compute_reverse_edge_idx(self, edge_index, num_nodes):
        """
        预计算每条边的反向边索引（向量化）
        
        返回: [E] tensor，reverse_idx[i] = j 表示边 i 的反向边是 j
              如果没有反向边，则为 -1
        """
        E = edge_index.size(1)
        device = edge_index.device
        
        # 用 (src * num_nodes + tgt) 作为边的唯一 ID
        src, tgt = edge_index[0], edge_index[1]
        edge_ids = src * num_nodes + tgt  # [E]
        reverse_edge_ids = tgt * num_nodes + src  # [E]
        
        # 构建 edge_id -> edge_idx 的映射
        # 使用 scatter 构建稀疏映射
        max_id = num_nodes * num_nodes
        id_to_idx = torch.full((max_id,), -1, dtype=torch.long, device=device)
        edge_indices = torch.arange(E, device=device)
        id_to_idx.scatter_(0, edge_ids, edge_indices)
        
        # 查找每条边的反向边
        reverse_idx = id_to_idx[reverse_edge_ids]  # [E]
        
        return reverse_idx
    
    def _antisymmetric_loss_vectorized(self, rel_probs, reverse_idx):
        """
        反对称约束（向量化版本）
        
        惩罚: sum( P(r|i,j) * P(r|j,i) ) 对于所有有反向边的 (i,j) 对
        """
        device = rel_probs.device
        E = rel_probs.size(0)
        
        # 找出有反向边的边
        has_reverse = reverse_idx >= 0  # [E]
        
        if not has_reverse.any():
            return torch.tensor(0.0, device=device)
        
        # 获取有反向边的边的索引和对应的反向边索引
        valid_idx = torch.where(has_reverse)[0]  # [K]
        valid_reverse_idx = reverse_idx[valid_idx]  # [K]
        
        # 反对称关系: Up, Down, Left, Right
        antisym_rels = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        
        # 取出这些关系的概率 [K, 4]
        forward_probs = rel_probs[valid_idx][:, antisym_rels]
        reverse_probs = rel_probs[valid_reverse_idx][:, antisym_rels]
        
        # 惩罚: P(r|i,j) * P(r|j,i)
        loss = (forward_probs * reverse_probs).sum()
        
        # 归一化
        count = valid_idx.size(0) * len(antisym_rels)
        return loss / max(count, 1)
    
    def _tree_constraint_loss(self, rel_probs, edge_index, num_nodes):
        """
        树约束: 每个节点最多有一个 Parent（已经是向量化的）
        """
        device = rel_probs.device
        
        parent_probs = rel_probs[:, self.PARENT]
        tgt_nodes = edge_index[1]
        
        parent_sum = torch.zeros(num_nodes, device=device)
        parent_sum.scatter_add_(0, tgt_nodes, parent_probs)
        
        violation = F.softplus(parent_sum - 1.0)
        return violation.mean()
    
    def _dag_constraint_loss_vectorized(self, rel_probs, reverse_idx):
        """
        DAG 约束（向量化版本）
        
        惩罚 2-环: P(seq|i,j) * P(seq|j,i)
        """
        device = rel_probs.device
        
        has_reverse = reverse_idx >= 0
        
        if not has_reverse.any():
            return torch.tensor(0.0, device=device)
        
        valid_idx = torch.where(has_reverse)[0]
        valid_reverse_idx = reverse_idx[valid_idx]
        
        # Sequence 关系的概率
        forward_seq = rel_probs[valid_idx, self.SEQUENCE]
        reverse_seq = rel_probs[valid_reverse_idx, self.SEQUENCE]
        
        # 惩罚双向都有高概率
        loss = (forward_seq * reverse_seq).sum()
        
        return loss / max(valid_idx.size(0), 1)


# ============ 关系预测 Loss ============

class RelationLossWithConstraints(nn.Module):
    """
    关系预测 Loss + 约束 Loss
    
    constraint_mode:
        - 'full': 全部约束 (antisym + tree + dag)
        - 'tree_only': 只有单父约束 (tree)
        - 'none': 无约束
    """
    
    def __init__(self, 
                 spatial_weight=1.0, 
                 logic_weight=1.5, 
                 exist_weight=1.0,
                 constraint_weight=0.1,
                 hard_neg_ratio=3,
                 constraint_mode='full'):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.logic_weight = logic_weight
        self.exist_weight = exist_weight
        self.constraint_weight = constraint_weight
        self.hard_neg_ratio = hard_neg_ratio
        self.constraint_mode = constraint_mode
        
        # 根据 constraint_mode 设置约束权重
        if constraint_mode == 'full':
            antisym_weight = 1.0
            tree_weight = 1.0
            dag_weight = 0.5
        elif constraint_mode == 'tree_only':
            antisym_weight = 0.0  # 关闭反对称约束
            tree_weight = 1.0
            dag_weight = 0.0  # 关闭 DAG 约束
        else:  # 'none'
            antisym_weight = 0.0
            tree_weight = 0.0
            dag_weight = 0.0
        
        # 约束 Loss
        self.constraint_loss = ConstraintLoss(
            antisym_weight=antisym_weight,
            tree_weight=tree_weight,
            dag_weight=dag_weight
        )
        
        print(f"📋 约束模式: {constraint_mode} (antisym={antisym_weight}, tree={tree_weight}, dag={dag_weight})")
    
    def forward(self, relation_logits, exist_logits, edge_labels, 
                edge_index=None, num_nodes=None):
        """
        Args:
            relation_logits: [E, num_relations]
            exist_logits: [E]
            edge_labels: [E, num_relations] multi-hot
            edge_index: [2, E] 边索引（用于约束 Loss）
            num_nodes: 节点数量
        """
        E = relation_logits.size(0)
        device = relation_logits.device
        
        if E == 0:
            return torch.tensor(0.0, device=device), {
                'rel_loss': 0.0, 'exist_loss': 0.0, 'constraint_loss': 0.0
            }
        
        # 1. 关系分类 Loss
        has_relation = edge_labels.any(dim=1)  # [E]
        pos_mask = has_relation
        
        if pos_mask.sum() > 0:
            # 正样本的关系分类
            pos_logits = relation_logits[pos_mask]
            pos_labels = edge_labels[pos_mask].float()
            
            # 分空间和逻辑关系加权
            spatial_loss = F.binary_cross_entropy_with_logits(
                pos_logits[:, :4], pos_labels[:, :4], reduction='mean'
            )
            logic_loss = F.binary_cross_entropy_with_logits(
                pos_logits[:, 4:], pos_labels[:, 4:], reduction='mean'
            )
            rel_loss = self.spatial_weight * spatial_loss + self.logic_weight * logic_loss
        else:
            rel_loss = torch.tensor(0.0, device=device)
        
        # 2. 边存在性 Loss (Hard Negative Mining)
        exist_labels = has_relation.float()
        
        num_pos = pos_mask.sum().item()
        num_neg = int(num_pos * self.hard_neg_ratio)
        
        if num_pos > 0 and (~pos_mask).sum() > 0:
            neg_mask = ~pos_mask
            neg_scores = torch.sigmoid(exist_logits[neg_mask])
            
            if neg_scores.numel() > num_neg:
                _, hard_neg_indices = torch.topk(neg_scores, min(num_neg, neg_scores.numel()))
                neg_indices = torch.where(neg_mask)[0][hard_neg_indices]
                
                selected_mask = pos_mask.clone()
                selected_mask[neg_indices] = True
            else:
                selected_mask = torch.ones(E, dtype=torch.bool, device=device)
            
            exist_loss = F.binary_cross_entropy_with_logits(
                exist_logits[selected_mask],
                exist_labels[selected_mask],
                reduction='mean'
            )
        else:
            exist_loss = F.binary_cross_entropy_with_logits(
                exist_logits, exist_labels, reduction='mean'
            )
        
        # 3. 约束 Loss
        constraint_loss_val = torch.tensor(0.0, device=device)
        constraint_dict = {}
        
        if self.constraint_weight > 0 and edge_index is not None and num_nodes is not None:
            rel_probs = torch.sigmoid(relation_logits)
            constraint_loss_val, constraint_dict = self.constraint_loss(
                rel_probs, edge_index, num_nodes
            )
        
        # 总 Loss
        total_loss = (rel_loss + 
                      self.exist_weight * exist_loss + 
                      self.constraint_weight * constraint_loss_val)
        
        loss_dict = {
            'rel_loss': rel_loss.item(),
            'exist_loss': exist_loss.item(),
            'constraint_loss': constraint_loss_val.item(),
            **{f'c_{k}': v for k, v in constraint_dict.items()}
        }
        
        return total_loss, loss_dict


# ============ 训练器 ============

class ConstraintTrainer:
    """带约束 Loss 的训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")
        
        self.model = YOLOgDSATransformerVisualOnly(
            yolo_model_path=config['yolo_model_path'],
            num_classes=config['num_classes'],
            num_relations=config['num_relations'],
            visual_feature_dim=config['visual_feature_dim'],
            edge_geometric_dim=config['edge_geometric_dim'],
            transformer_hidden_dim=config['transformer_hidden_dim'],
            num_transformer_layers=config['num_transformer_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_neck_features=True,
            use_edge_features=config.get('use_edge_features', True)
        ).to(self.device)
        edge_status = "有边特征" if config.get('use_edge_features', True) else "无边特征"
        print(f"✅ 使用 Transformer 模型 ({edge_status})")
        
        # 解冻 YOLO 和 Backbone（允许端到端训练）
        self.model.unfreeze_yolo()
        if hasattr(self.model, 'backbone_extractor'):
            self.model.backbone_extractor.unfreeze_backbone()
        elif hasattr(self.model, 'feature_extractor'):
            self.model.feature_extractor.unfreeze_backbone()
        
        # Loss - 根据 constraint_mode 配置约束
        self.relation_loss = RelationLossWithConstraints(
            spatial_weight=1.0,
            logic_weight=1.5,
            exist_weight=1.0,
            constraint_weight=config['constraint_weight'],
            hard_neg_ratio=3,
            constraint_mode=config['constraint_mode']
        )
        
        # 保存约束模式用于验证
        self.constraint_mode = config['constraint_mode']
        
        # 数据集
        self.train_dataset = YOLOgDSADataset(
            img_dir=config['train_img_dir'],
            yolo_dir=config['train_yolo_dir'],
            spatial_rel_dir=config['spatial_rel_dir'],
            logic_rel_dir=config['logic_rel_dir'],
            classes_file=config['classes_file'],
            mode='train',
            img_size=config['img_size'],
            use_full_graph=True,
            use_original_size=config.get('use_original_size', False),
            max_size=config.get('max_size', 1280)
        )
        
        self.val_dataset = YOLOgDSADataset(
            img_dir=config['val_img_dir'],
            yolo_dir=config['val_yolo_dir'],
            spatial_rel_dir=config['spatial_rel_dir'],
            logic_rel_dir=config['logic_rel_dir'],
            classes_file=config['classes_file'],
            mode='val',
            img_size=config['img_size'],
            use_full_graph=True,
            use_original_size=config.get('use_original_size', False),
            max_size=config.get('max_size', 1280)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=4
        )
        
        print(f"📊 训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_metrics = {}
        self.best_metrics_05 = {}  # mAP@0.5 最佳
        self.best_metrics_075 = {}  # mAP@0.75 最佳
        self.best_metrics_095 = {}  # mAP@0.95 最佳
        self.history = defaultdict(list)
        
        # 早停（监控三个 mAP 指标）
        self.early_stop_patience = config.get('early_stop_patience', 0)  # 0 表示不启用
        self.early_stop_counter = 0
        self.best_early_stop_metrics = {
            'mAP_g@0.5': 0.0,
            'mAP_g@0.75': 0.0,
            'mAP_g@0.95': 0.0
        }
        
        # 时间统计
        self.train_start_time = None
        self.total_train_time = 0.0
        
        # 保存目录
        os.makedirs(config['save_dir'], exist_ok=True)

    
    def train_epoch(self, epoch):
        """训练一个 epoch（支持梯度累积）"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        
        # pred 模式统计
        pred_mode_stats = {
            'total_samples': 0,
            'valid_samples': 0,  # 有足够匹配的样本
            'total_pred_boxes': 0,
            'total_matched': 0
        }
        
        accum_steps = self.config.get('accum_steps', 1)
        accum_loss = 0.0
        accum_count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            gt_boxes = batch['boxes'].squeeze(0).to(self.device)
            gt_classes = batch['classes'].squeeze(0).to(self.device)
            gt_edge_index = batch['edge_index'].squeeze(0).to(self.device)
            gt_edge_labels = batch['edge_labels'].squeeze(0).to(self.device)
            
            if len(gt_boxes) < 2:
                continue
            
            rel_train_mode = self.config.get('rel_train_mode', 'gt')
            if rel_train_mode == 'pred':
                pred_mode_stats['total_samples'] += 1
                
                # ====== pred 模式：用检测框训练关系（优化：尝试复用前向传播）======
                # 早期训练降低置信度阈值，让检测器输出更多框
                conf_thresh = 0.01 if epoch <= 10 else 0.05 if epoch <= 50 else 0.25
                
                # 🔥 优化方案3尝试：使用 detect_and_get_raw_preds 获取检测框和原始预测
                # 注意：YOLO loss 计算需要训练模式的前向传播，无法完全复用 eval 模式的预测
                # 因此这里仍然会有两次前向传播，但第一次更快（no_grad + eval）
                if hasattr(self.model, 'detect_and_get_raw_preds'):
                    pred_boxes, pred_classes, _, raw_preds = self.model.detect_and_get_raw_preds(images, conf_thresh=conf_thresh)
                else:
                    # 回退到原始方法
                    det_model = getattr(self.model, 'det_model', None)
                    if det_model is not None:
                        was_training = det_model.training
                        det_model.eval()
                        
                        with torch.no_grad():
                            pred_boxes, pred_classes, _ = self.model._detect_with_det_model(images, conf_thresh=conf_thresh)
                        
                        det_model.train(was_training)
                    else:
                        pred_boxes, pred_classes, _ = self.model._detect(images, conf_thresh=conf_thresh)
                
                pred_mode_stats['total_pred_boxes'] += len(pred_boxes)

                # IoU+类别 matching: {gt_idx: pred_idx}
                matching = self._instance_matching_drgg_style(
                    pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5
                )
                
                pred_mode_stats['total_matched'] += len(matching)

                # 只在匹配到的 pred 节点子图上训练（类似 DRGG：matched queries）
                if len(matching) >= 2:
                    pred_mode_stats['valid_samples'] += 1
                    matched_gt_indices = sorted(matching.keys())
                    matched_pred_indices = [matching[gi] for gi in matched_gt_indices]
                    matched_pred_indices = torch.tensor(matched_pred_indices, dtype=torch.long, device=self.device)
                    matched_gt_indices = torch.tensor(matched_gt_indices, dtype=torch.long, device=self.device)

                    boxes = pred_boxes[matched_pred_indices]
                    classes = pred_classes[matched_pred_indices]
                    M = len(boxes)

                    # 构建全连接图 (M nodes)
                    src = torch.arange(M, device=self.device).repeat_interleave(M)
                    tgt = torch.arange(M, device=self.device).repeat(M)
                    mask = src != tgt
                    pred_edge_index = torch.stack([src[mask], tgt[mask]], dim=0)

                    # 构建 GT dense 关系矩阵 [N_gt, N_gt, R]
                    R = self.config['num_relations']
                    gt_rel_dense = torch.zeros(len(gt_boxes), len(gt_boxes), R, device=self.device)
                    if gt_edge_index.numel() > 0 and gt_edge_labels.numel() > 0:
                        gt_rel_dense[gt_edge_index[0], gt_edge_index[1]] = gt_edge_labels.float()

                    # 将 pred 子图边映射到 gt 子图边，生成 pred_edge_labels
                    gt_idx_per_pred = matched_gt_indices  # (M,)
                    src_gt = gt_idx_per_pred[pred_edge_index[0]]
                    tgt_gt = gt_idx_per_pred[pred_edge_index[1]]
                    pred_edge_labels = gt_rel_dense[src_gt, tgt_gt]  # (E, R)

                    # 前向传播：用检测框 + 全连接图做关系预测
                    output = self.model(
                        images,
                        gt_boxes=boxes,
                        gt_classes=classes,
                        edge_index=pred_edge_index
                    )

                    if isinstance(output, dict):
                        rel_logits = output['relation_logits']
                        exist_logits = output['exist_logits']
                    else:
                        _, rel_logits, exist_logits = output

                    loss, loss_dict = self.relation_loss(
                        rel_logits, exist_logits, pred_edge_labels,
                        edge_index=pred_edge_index,
                        num_nodes=M
                    )
                else:
                    # 匹配失败，跳过该样本
                    loss = torch.tensor(0.0, device=self.device)
                    loss_dict = {'rel_loss': 0.0, 'exist_loss': 0.0, 'constraint_loss': 0.0}
                
                # 🔥 pred 模式：第二次前向计算 det_loss（需要梯度）
                yolo_boxes = gt_boxes.clone()
                yolo_boxes_xywh = torch.zeros_like(yolo_boxes)
                yolo_boxes_xywh[:, 0] = (yolo_boxes[:, 0] + yolo_boxes[:, 2]) / 2
                yolo_boxes_xywh[:, 1] = (yolo_boxes[:, 1] + yolo_boxes[:, 3]) / 2
                yolo_boxes_xywh[:, 2] = yolo_boxes[:, 2] - yolo_boxes[:, 0]
                yolo_boxes_xywh[:, 3] = yolo_boxes[:, 3] - yolo_boxes[:, 1]
                det_loss, _ = self.model.compute_det_loss(images, yolo_boxes_xywh, gt_classes)
                
            else:
                # ====== gt 模式：用 GT boxes + GT edge_index 训练关系（当前默认）======
                output = self.model(
                    images,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    edge_index=gt_edge_index
                )

                rel_logits = output['relation_logits']
                exist_logits = output['exist_logits']

                # 计算 Loss（包含约束）
                loss, loss_dict = self.relation_loss(
                    rel_logits, exist_logits, gt_edge_labels,
                    edge_index=gt_edge_index,
                    num_nodes=len(gt_boxes)
                )
                
                # gt 模式：计算 det_loss
                yolo_boxes = gt_boxes.clone()
                yolo_boxes_xywh = torch.zeros_like(yolo_boxes)
                yolo_boxes_xywh[:, 0] = (yolo_boxes[:, 0] + yolo_boxes[:, 2]) / 2
                yolo_boxes_xywh[:, 1] = (yolo_boxes[:, 1] + yolo_boxes[:, 3]) / 2
                yolo_boxes_xywh[:, 2] = yolo_boxes[:, 2] - yolo_boxes[:, 0]
                yolo_boxes_xywh[:, 3] = yolo_boxes[:, 3] - yolo_boxes[:, 1]
                det_loss, _ = self.model.compute_det_loss(images, yolo_boxes_xywh, gt_classes)
            
            # 总 Loss（除以累积步数进行缩放）
            total = (loss + self.config['det_weight'] * det_loss) / accum_steps
            
            total.backward()
            
            accum_loss += total.item() * accum_steps  # 还原真实 loss 用于记录
            accum_count += 1
            
            # 梯度累积：每 accum_steps 步更新一次
            if accum_count % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += accum_loss
                for k, v in loss_dict.items():
                    loss_components[k] += v
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{accum_loss:.4f}",
                    'rel': f"{loss_dict.get('rel_loss', 0):.3f}",
                    'cst': f"{loss_dict.get('constraint_loss', 0):.3f}"
                })
                
                accum_loss = 0.0
        
        # 平均
        avg_loss = total_loss / max(num_batches, 1)
        for k in loss_components:
            loss_components[k] /= max(num_batches, 1)
        
        # 打印 pred 模式统计
        if self.config.get('rel_train_mode') == 'pred':
            print(f"\n📊 pred模式统计:")
            print(f"   总样本: {pred_mode_stats['total_samples']}")
            print(f"   有效样本: {pred_mode_stats['valid_samples']} ({pred_mode_stats['valid_samples']/max(1,pred_mode_stats['total_samples'])*100:.1f}%)")
            print(f"   平均检测框数: {pred_mode_stats['total_pred_boxes']/max(1,pred_mode_stats['total_samples']):.1f}")
            print(f"   平均匹配数: {pred_mode_stats['total_matched']/max(1,pred_mode_stats['total_samples']):.1f}")
        
        return avg_loss, dict(loss_components)

    
    def validate(self):
        """验证（真正的 E2E：使用检测框做关系预测）"""
        self.model.eval()
        
        # 使用新的评估器（支持 raw + VAS 双轨）
        evaluator = gDSAEvaluatorNew(
            num_relations=self.config['num_relations'],
            iou_threshold=0.5,
            relation_thresholds=[0.5, 0.75, 0.95],
            topk_per_relation=2000
        )
        
        # DLA 指标统计
        dla_stats = {
            'all_pred_boxes': [],
            'all_pred_scores': [],
            'all_pred_classes': [],
            'all_gt_boxes': [],
            'all_gt_classes': []
        }
        
        # 推理速度统计
        inference_times = []
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                gt_boxes = batch['boxes'].squeeze(0).to(self.device)
                gt_classes = batch['classes'].squeeze(0).to(self.device)
                gt_edge_index = batch['edge_index'].squeeze(0).to(self.device)
                gt_edge_labels = batch['edge_labels'].squeeze(0).to(self.device)
                
                # 计时开始
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                t_start = time.perf_counter()
                
                # E2E 评估：使用检测框（不是 GT 框）
                if hasattr(self.model, '_detect_with_det_model'):
                    pred_boxes, pred_classes, pred_scores = self.model._detect_with_det_model(
                        images, conf_thresh=0.25
                    )
                else:
                    pred_boxes, pred_classes, pred_scores = self.model._detect(
                        images, conf_thresh=0.25
                    )
                
                # DLA 统计
                if len(pred_boxes) > 0 and pred_scores is not None:
                    dla_stats['all_pred_boxes'].append(pred_boxes.cpu())
                    dla_stats['all_pred_scores'].append(pred_scores.cpu())
                    dla_stats['all_pred_classes'].append(pred_classes.cpu())
                
                dla_stats['all_gt_boxes'].append(gt_boxes.cpu())
                dla_stats['all_gt_classes'].append(gt_classes.cpu())
                
                # gDSA E2E 评估需要至少 2 个 GT 框和 2 个检测框
                if len(gt_boxes) < 2 or len(pred_boxes) < 2:
                    # 计时结束
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    t_end = time.perf_counter()
                    inference_times.append(t_end - t_start)
                    num_samples += 1
                    continue
                
                # 构建全连接图
                N = len(pred_boxes)
                src = torch.arange(N, device=self.device).repeat_interleave(N)
                tgt = torch.arange(N, device=self.device).repeat(N)
                mask = src != tgt
                pred_edge_index = torch.stack([src[mask], tgt[mask]], dim=0)
                
                # 用检测框做关系预测
                output = self.model(
                    images,
                    gt_boxes=pred_boxes,
                    gt_classes=pred_classes,
                    edge_index=pred_edge_index
                )
                
                # 计时结束
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                t_end = time.perf_counter()
                inference_times.append(t_end - t_start)
                num_samples += 1
                
                rel_logits = output['relation_logits']
                exist_logits = output['exist_logits']
                
                rel_probs = torch.sigmoid(rel_logits)
                exist_probs = torch.sigmoid(exist_logits)
                
                # 构建 dense 概率矩阵 [N, N, R]（向量化版本）
                R = self.config['num_relations']
                pred_rel_probs_dense = torch.zeros(N, N, R, device=self.device)
                # 向量化写入：使用 edge_index 直接索引
                src_idx = pred_edge_index[0]  # [E]
                tgt_idx = pred_edge_index[1]  # [E]
                scores = rel_probs * exist_probs.unsqueeze(1)  # [E, R]
                pred_rel_probs_dense[src_idx, tgt_idx] = scores
                
                # 转换为评估器需要的格式
                pred_boxes_np = pred_boxes.cpu().numpy()
                pred_classes_np = pred_classes.cpu().numpy()
                gt_boxes_np = gt_boxes.cpu().numpy()
                gt_classes_np = gt_classes.cpu().numpy()
                
                # GT 关系列表（向量化版本，避免 Python 双循环和 GPU 同步）
                gt_edge_index_cpu = gt_edge_index.cpu()
                gt_edge_labels_cpu = gt_edge_labels.cpu()
                pos = (gt_edge_labels_cpu > 0).nonzero(as_tuple=False)  # [K, 2]，(e, r)
                e_idx = pos[:, 0]
                r_idx = pos[:, 1]
                src_gt = gt_edge_index_cpu[0, e_idx].numpy()
                tgt_gt = gt_edge_index_cpu[1, e_idx].numpy()
                rel_gt = r_idx.numpy()
                gt_relations = list(zip(src_gt.tolist(), tgt_gt.tolist(), rel_gt.tolist()))
                
                # 添加到评估器（使用 VAS）
                evaluator.add_sample(
                    pred_boxes=pred_boxes_np,
                    pred_classes=pred_classes_np,
                    pred_relations=[],  # 会从 dense 提取
                    pred_scores=pred_scores.cpu().numpy() if pred_scores is not None else None,
                    gt_boxes=gt_boxes_np,
                    gt_classes=gt_classes_np,
                    gt_relations=gt_relations,
                    pred_rel_probs=pred_rel_probs_dense,
                    use_vas=True,
                    lam=5.0, alpha=0.5, beta=0.5, gamma=1.0
                )
        
        # 计算 gDSA 指标（raw + VW-mAP）
        metrics_raw = evaluator.compute_metrics(use_vas=False)
        
        # 计算 VW-mAP 指标（方案 A: VW-mAP = mAP × V_eff）
        metrics_vw = evaluator.compute_vw_metrics(mode='mul', use_vas_for_map=False)
        
        # 合并指标（统一键名格式）
        metrics = {}
        # Raw 指标: mR_g@0.5 -> mR_g@0.5_raw
        for k, v in metrics_raw.items():
            metrics[k + '_raw'] = v
        
        # VW-mAP 指标
        for k, v in metrics_vw.items():
            metrics[k] = v
        
        # 为了兼容旧代码，保留 _e2e 后缀的指标（使用 raw 版本）
        metrics['mR_g@0.5_e2e'] = metrics_raw.get('mR_g@0.5', 0.0)
        metrics['mAP_g@0.5_e2e'] = metrics_raw.get('mAP_g@0.5', 0.0)
        metrics['mR_g@0.75_e2e'] = metrics_raw.get('mR_g@0.75', 0.0)
        metrics['mAP_g@0.75_e2e'] = metrics_raw.get('mAP_g@0.75', 0.0)
        metrics['mR_g@0.95_e2e'] = metrics_raw.get('mR_g@0.95', 0.0)
        metrics['mAP_g@0.95_e2e'] = metrics_raw.get('mAP_g@0.95', 0.0)
        
        # 计算 DLA 指标（可选）
        if not self.config.get('skip_dla', False):
            dla_metrics = self._compute_dla_metrics(dla_stats)
            metrics.update(dla_metrics)
        else:
            metrics['mAP50'] = 0.0
            metrics['mAP50-95'] = 0.0
        
        # 推理速度统计
        if inference_times:
            avg_time = np.mean(inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            metrics['inference_time_ms'] = avg_time * 1000
            metrics['inference_fps'] = fps
            print(f"   ⏱️ 推理速度: {avg_time*1000:.1f}ms/sample, {fps:.1f} FPS")
        
        return metrics
    
    def _instance_matching_drgg_style(self, pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5):
        """
        与 DRGG 评估器完全相同的实例匹配逻辑（完全 GPU 向量化版本）
        
        返回: matching = {gt_idx: pred_idx}
        """
        N_pred = len(pred_boxes)
        N_gt = len(gt_boxes)
        
        if N_pred == 0 or N_gt == 0:
            return {}
        
        device = pred_boxes.device
        
        # 计算 IoU 矩阵 [N_pred, N_gt]
        iou_matrix = self._box_iou(pred_boxes, gt_boxes)  # 保持在 GPU
        
        # 向量化类别匹配：[N_pred, N_gt]
        class_match = (pred_classes.unsqueeze(1) == gt_classes.unsqueeze(0)).float()
        iou_matrix = iou_matrix * class_match  # 类别不匹配的位置 IoU 置零
        
        # 对每个 pred，找到最佳 gt（向量化）
        max_ious, best_gt_indices = iou_matrix.max(dim=1)  # [N_pred]
        
        # 过滤掉低于阈值的匹配
        valid_mask = max_ious > iou_thresh  # [N_pred]
        
        if not valid_mask.any():
            return {}
        
        # 获取有效匹配（全程在 GPU）
        valid_pred_indices = torch.where(valid_mask)[0]  # [K]
        valid_gt_indices = best_gt_indices[valid_mask]  # [K]
        valid_ious = max_ious[valid_mask]  # [K]
        
        # 处理多个 pred 匹配到同一个 gt 的情况（向量化）
        # 策略：对每个 gt，保留 IoU 最大的 pred
        
        # 创建稀疏映射：gt_idx -> (pred_idx, iou)
        # 使用 scatter_reduce 找到每个 gt 的最大 IoU
        gt_max_ious = torch.zeros(N_gt, device=device)
        gt_max_ious.scatter_reduce_(
            0, valid_gt_indices, valid_ious, 
            reduce='amax', include_self=False
        )
        
        # 找出每个 gt 对应的最佳 pred（IoU 最大的那个）
        # 对于每个有效匹配，检查它是否是该 gt 的最佳匹配
        is_best_match = (valid_ious == gt_max_ious[valid_gt_indices])  # [K]
        
        # 过滤出最佳匹配
        best_pred_indices = valid_pred_indices[is_best_match]
        best_gt_indices = valid_gt_indices[is_best_match]
        
        # 转换为字典（只在最后做一次 CPU 转换）
        matching = {}
        if len(best_gt_indices) > 0:
            # 一次性转换到 CPU
            gt_list = best_gt_indices.cpu().tolist()
            pred_list = best_pred_indices.cpu().tolist()
            matching = dict(zip(gt_list, pred_list))
        
        return matching
    
    def _compute_dla_metrics(self, dla_stats):
        """计算 DLA (Document Layout Analysis) 指标（优化版）"""
        metrics = {}
        
        if not dla_stats['all_pred_boxes']:
            print(f"   ⚠️ DLA: 没有检测到任何框")
            return {'mAP50': 0.0, 'mAP50-95': 0.0}
        
        # 调试信息
        total_pred = sum(len(b) for b in dla_stats['all_pred_boxes'])
        total_gt = sum(len(b) for b in dla_stats['all_gt_boxes'])
        print(f"   📊 DLA: 预测框={total_pred}, GT框={total_gt}, 样本数={len(dla_stats['all_pred_boxes'])}/{len(dla_stats['all_gt_boxes'])}")
        
        num_classes = self.config['num_classes']
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5 to 0.95
        
        # 预先收集所有样本的数据（避免重复计算）
        num_samples = len(dla_stats['all_gt_boxes'])
        
        # 按类别统计
        class_aps = {c: [] for c in range(num_classes)}
        
        for iou_thresh in iou_thresholds:
            for c in range(num_classes):
                all_scores = []
                all_tps = []
                num_gt = 0
                
                for i in range(num_samples):
                    gt_boxes = dla_stats['all_gt_boxes'][i]
                    gt_classes = dla_stats['all_gt_classes'][i]
                    
                    # GT boxes for this class
                    gt_mask = gt_classes == c
                    gt_boxes_c = gt_boxes[gt_mask]
                    num_gt += len(gt_boxes_c)
                    
                    if i < len(dla_stats['all_pred_boxes']):
                        pred_boxes = dla_stats['all_pred_boxes'][i]
                        pred_scores = dla_stats['all_pred_scores'][i]
                        pred_classes = dla_stats['all_pred_classes'][i]
                        
                        # Pred boxes for this class
                        pred_mask = pred_classes == c
                        pred_boxes_c = pred_boxes[pred_mask]
                        pred_scores_c = pred_scores[pred_mask]
                        
                        if len(pred_boxes_c) > 0 and len(gt_boxes_c) > 0:
                            # 计算 IoU（已经是向量化的）
                            ious = self._box_iou(pred_boxes_c, gt_boxes_c)
                            
                            # 向量化匹配：按分数降序处理
                            sorted_idx = torch.argsort(pred_scores_c, descending=True)
                            scores_sorted = pred_scores_c[sorted_idx].cpu().numpy()
                            ious_sorted = ious[sorted_idx].cpu().numpy()
                            
                            # 贪心匹配
                            matched_gt = np.zeros(len(gt_boxes_c), dtype=bool)
                            tps = np.zeros(len(pred_boxes_c), dtype=int)
                            
                            for j in range(len(sorted_idx)):
                                # 找未匹配的最大 IoU
                                iou_row = ious_sorted[j].copy()
                                iou_row[matched_gt] = -1  # 已匹配的设为 -1
                                best_gt = np.argmax(iou_row)
                                best_iou = iou_row[best_gt]
                                
                                if best_iou >= iou_thresh:
                                    tps[j] = 1
                                    matched_gt[best_gt] = True
                            
                            all_scores.extend(scores_sorted.tolist())
                            all_tps.extend(tps.tolist())
                        elif len(pred_boxes_c) > 0:
                            all_scores.extend(pred_scores_c.cpu().numpy().tolist())
                            all_tps.extend([0] * len(pred_boxes_c))
                
                # 计算 AP
                if num_gt > 0 and len(all_tps) > 0:
                    scores_arr = np.array(all_scores)
                    tps_arr = np.array(all_tps)
                    
                    # 按分数排序
                    sorted_idx = np.argsort(-scores_arr)
                    tp_sorted = tps_arr[sorted_idx]
                    
                    # 累积
                    tp_cumsum = np.cumsum(tp_sorted)
                    fp_cumsum = np.cumsum(1 - tp_sorted)
                    
                    recalls = tp_cumsum / num_gt
                    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                    
                    # 插值 AP（向量化）
                    precisions_interp = np.maximum.accumulate(precisions[::-1])[::-1]
                    
                    # 计算 AP
                    recall_diff = np.diff(np.concatenate([[0], recalls]))
                    ap = np.sum(recall_diff * precisions_interp)
                    
                    class_aps[c].append(ap)
                else:
                    class_aps[c].append(0.0)
        
        # 计算 mAP
        ap50_list = []
        ap50_95_list = []
        
        for c in range(num_classes):
            if class_aps[c]:
                ap50_list.append(class_aps[c][0])  # IoU=0.5
                ap50_95_list.append(np.mean(class_aps[c]))  # 0.5-0.95
        
        metrics['mAP50'] = np.mean(ap50_list) if ap50_list else 0.0
        metrics['mAP50-95'] = np.mean(ap50_95_list) if ap50_95_list else 0.0
        
        return metrics
    
    def _box_iou(self, boxes1, boxes2):
        """计算两组框的 IoU"""
        # boxes: [N, 4] in xyxy format
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 交集
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # IoU
        iou = inter / (area1[:, None] + area2[None, :] - inter + 1e-6)
        return iou
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 (约束权重: {self.config['constraint_weight']})")
        print(f"{'='*60}\n")
        
        # 记录训练开始时间
        self.train_start_time = datetime.now()
        self.history['train_start_time'] = self.train_start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_loss, loss_components = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            for k, v in loss_components.items():
                self.history[f'train_{k}'].append(v)
            
            # 记录 epoch 训练时间
            epoch_train_time = time.time() - epoch_start_time
            self.history['epoch_train_time'].append(epoch_train_time)
            self.total_train_time += epoch_train_time
            
            # 验证（按间隔执行）
            val_interval = self.config.get('val_interval', 1)
            if epoch % val_interval == 0 or epoch == self.config['epochs']:
                val_start_time = time.time()
                metrics = self.validate()
                val_time = time.time() - val_start_time
                self.history['epoch_val_time'].append(val_time)
                
                for k, v in metrics.items():
                    self.history[k].append(v)
                
                # 打印验证指标
                print(f"\nEpoch {epoch}/{self.config['epochs']} (训练: {epoch_train_time:.1f}s, 验证: {val_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Constraint Loss: {loss_components.get('constraint_loss', 0):.4f}")
                print(f"  DLA: mAP50-95={metrics.get('mAP50-95', 0):.4f}")
                print(f"  gDSA (Raw):    mR_g@0.5={metrics.get('mR_g@0.5_raw', 0):.4f}  mAP_g@0.5={metrics.get('mAP_g@0.5_raw', 0):.4f}  mAP_g@0.75={metrics.get('mAP_g@0.75_raw', 0):.4f}  mAP_g@0.95={metrics.get('mAP_g@0.95_raw', 0):.4f}")
                # Always print mAP_norm if available
                print(f"  gDSA (Norm):   mAP_norm@0.5={metrics.get('mAP_norm@0.5_raw', 0):.4f}  mAP_norm@0.75={metrics.get('mAP_norm@0.75_raw', 0):.4f}  mAP_norm@0.95={metrics.get('mAP_norm@0.95_raw', 0):.4f}")
                print(f"  Validity:      V@0.5={metrics.get('V@0.5', 0):.4f}  C_E={metrics.get('C_E@0.5', 0):.4f}  V_eff={metrics.get('V_eff@0.5', 0):.4f}  (V_tree={metrics.get('V_tree@0.5', 0):.4f})")
                print(f"  VW-mAP:        VW-mAP@0.5={metrics.get('VW-mAP_g@0.5', 0):.4f}  VW-mAP@0.75={metrics.get('VW-mAP_g@0.75', 0):.4f}  VW-mAP@0.95={metrics.get('VW-mAP_g@0.95', 0):.4f}")
                
                # 保存最佳模型 - mAP_g@0.5
                current_map_05 = metrics['mAP_g@0.5_e2e']
                if current_map_05 > self.best_metrics_05.get('mAP_g@0.5_e2e', 0):
                    self.best_metrics_05 = metrics.copy()
                    self.save_checkpoint('best_mAP_g_05.pth')
                    print(f"  ✅ 新最佳 mAP_g@0.5: {current_map_05:.4f}")
                
                # 保存最佳模型 - mAP_g@0.75
                current_map_075 = metrics['mAP_g@0.75_e2e']
                if current_map_075 > self.best_metrics_075.get('mAP_g@0.75_e2e', 0):
                    self.best_metrics_075 = metrics.copy()
                    self.save_checkpoint('best_mAP_g_075.pth')
                    print(f"  ✅ 新最佳 mAP_g@0.75: {current_map_075:.4f}")
                
                # 保存最佳模型 - mAP_g@0.95
                current_map_095 = metrics['mAP_g@0.95_e2e']
                if current_map_095 > self.best_metrics_095.get('mAP_g@0.95_e2e', 0):
                    self.best_metrics_095 = metrics.copy()
                    self.save_checkpoint('best_mAP_g_095.pth')
                    print(f"  ✅ 新最佳 mAP_g@0.95: {current_map_095:.4f}")
                
                # 更新综合最佳（用于兼容）
                if current_map_05 > self.best_metrics.get('mAP_g@0.5_e2e', 0):
                    self.best_metrics = metrics.copy()
                
                # 早停检查（监控三个 mAP 指标，任何一个提升都重置计数器）
                if self.early_stop_patience > 0:
                    has_improvement = False
                    
                    # 检查 mAP_g@0.5
                    if current_map_05 > self.best_early_stop_metrics['mAP_g@0.5']:
                        self.best_early_stop_metrics['mAP_g@0.5'] = current_map_05
                        has_improvement = True
                        print(f"  📈 mAP_g@0.5 提升: {current_map_05:.4f}")
                    
                    # 检查 mAP_g@0.75
                    if current_map_075 > self.best_early_stop_metrics['mAP_g@0.75']:
                        self.best_early_stop_metrics['mAP_g@0.75'] = current_map_075
                        has_improvement = True
                        print(f"  📈 mAP_g@0.75 提升: {current_map_075:.4f}")
                    
                    # 检查 mAP_g@0.95
                    if current_map_095 > self.best_early_stop_metrics['mAP_g@0.95']:
                        self.best_early_stop_metrics['mAP_g@0.95'] = current_map_095
                        has_improvement = True
                        print(f"  📈 mAP_g@0.95 提升: {current_map_095:.4f}")
                    
                    # 更新早停计数器
                    if has_improvement:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        print(f"  ⏳ 早停计数: {self.early_stop_counter}/{self.early_stop_patience} (无任何指标提升)")
                        
                        if self.early_stop_counter >= self.early_stop_patience:
                            print(f"\n⚠️ 早停触发！{self.early_stop_patience} 轮无任何指标提升")
                            print(f"最佳指标:")
                            print(f"  mAP_g@0.5:  {self.best_early_stop_metrics['mAP_g@0.5']:.4f}")
                            print(f"  mAP_g@0.75: {self.best_early_stop_metrics['mAP_g@0.75']:.4f}")
                            print(f"  mAP_g@0.95: {self.best_early_stop_metrics['mAP_g@0.95']:.4f}")
                            break
            else:
                # 不验证时只打印训练 loss
                print(f"\nEpoch {epoch}/{self.config['epochs']} (训练: {epoch_train_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Constraint Loss: {loss_components.get('constraint_loss', 0):.4f}")
            
            # 学习率调度
            self.scheduler.step()
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
            
            # 更新总训练时间并保存
            self.history['total_train_time_hours'] = self.total_train_time / 3600
            self.save_history()
        
        # 训练结束
        train_end_time = datetime.now()
        total_time = (train_end_time - self.train_start_time).total_seconds()
        self.history['train_end_time'] = train_end_time.strftime('%Y-%m-%d %H:%M:%S')
        self.history['total_time_hours'] = total_time / 3600
        self.save_history()
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"\n最佳指标 (合法图):")
        print(f"  DLA: mAP50-95={self.best_metrics.get('mAP50-95', 0):.4f}")
        print(f"  gDSA mAP_g@0.5:  {self.best_metrics_05.get('mAP_g@0.5_e2e', 0):.4f} (best_mAP_g_05.pth)")
        print(f"  gDSA mAP_g@0.75: {self.best_metrics_075.get('mAP_g@0.75_e2e', 0):.4f} (best_mAP_g_075.pth)")
        print(f"  gDSA mAP_g@0.95: {self.best_metrics_095.get('mAP_g@0.95_e2e', 0):.4f} (best_mAP_g_095.pth)")
        print(f"{'='*60}")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics,
            'best_metrics_05': self.best_metrics_05,
            'best_metrics_075': self.best_metrics_075,
            'best_metrics_095': self.best_metrics_095,
            'config': self.config,
            'total_train_time_hours': self.total_train_time / 3600,
        }
        path = os.path.join(self.config['save_dir'], filename)
        torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史到 JSON 文件"""
        path = os.path.join(self.config['save_dir'], 'training_history.json')
        
        # 确保目录存在
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 转换 defaultdict 为普通 dict
        history_dict = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                # 确保列表中的值可以序列化
                history_dict[key] = [float(v) if isinstance(v, (int, float, np.number)) else v for v in value]
            else:
                history_dict[key] = value
        
        try:
            with open(path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            # print(f"✅ 训练历史已保存: {path}")  # 可选：调试信息
        except Exception as e:
            print(f"⚠️ 保存训练历史失败: {e}")
            print(f"   路径: {path}")
            print(f"   历史数据键: {list(history_dict.keys())}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='带约束 Loss 的 gDSA 训练')
    parser.add_argument('--yolo', type=str, default='yolo11n.pt', help='YOLO 模型路径')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
                                 'resnet50', 'convnext', 'swin', 'efficientnet', 'mobilenet'],
                        help='Backbone 类型 (会覆盖 --yolo 参数)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--constraint-weight', type=float, default=0.1, 
                        help='约束 Loss 权重')
    parser.add_argument('--det-weight', type=float, default=0.5, help='检测 Loss 权重')
    parser.add_argument('--accum', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--save-dir', type=str, default=None, help='保存目录')
    parser.add_argument('--constraint-mode', type=str, default='full',
                        choices=['full', 'tree_only', 'none'],
                        help='约束模式: full=全部约束, tree_only=只有单父约束(快), none=无约束')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='验证间隔（每N个epoch验证一次，默认1）')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像尺寸 (默认640，可选 960/1280/1920)')
    parser.add_argument('--use-original-size', action='store_true',
                        help='使用原始图片尺寸（不做letterbox，避免坐标转换问题）')
    parser.add_argument('--max-size', type=int, default=1280,
                        help='原始尺寸模式下的最大边长限制（默认1280，设为0则不限制）')
    parser.add_argument('--skip-dla', action='store_true',
                        help='跳过 DLA 指标计算（加速验证）')
    parser.add_argument('--early-stop', type=int, default=0,
                        help='早停轮数（默认0不启用，例如30表示30轮无提升则停止）')
    parser.add_argument('--rel-train-mode', type=str, default='gt',
                        choices=['gt', 'pred'],
                        help='关系训练模式: gt=用GT框/GT边训练(默认), pred=用检测框训练(匹配投影GT标签)')
    parser.add_argument('--no-edge-features', action='store_true',
                        help='不使用边几何特征（对照实验：只用节点特征）')
    args = parser.parse_args()
    
    # 处理 backbone 参数
    if args.backbone:
        yolo_path = BACKBONE_MAP.get(args.backbone, args.yolo)
    else:
        yolo_path = args.yolo
    
    # 自动生成保存目录名
    if args.save_dir is None:
        backbone_name = args.backbone or 'yolo11n'
        edge_suffix = '_noedge' if args.no_edge_features else ''
        save_dir = f'runs/geostruct_gt_{backbone_name}_cw{args.constraint_weight}{edge_suffix}'
    else:
        save_dir = args.save_dir
    
    # 配置
    config = {
        # 数据
        'train_img_dir': './dataset_gdsa/images/train',
        'train_yolo_dir': './dataset_gdsa/labels/train',
        'val_img_dir': './dataset_gdsa/images/val',
        'val_yolo_dir': './dataset_gdsa/labels/val',
        'spatial_rel_dir': './dict_spatial_rels',
        'logic_rel_dir': './dict_logic_rels',
        'classes_file': './dataset_gdsa/classes.txt',
        'img_size': args.img_size,
        'use_original_size': args.use_original_size,
        'max_size': args.max_size,
        
        # 模型
        'yolo_model_path': yolo_path,
        'model_type': 'transformer',
        'num_classes': 8,
        'num_relations': 8,
        'visual_feature_dim': 256,
        'edge_geometric_dim': 64,
        'transformer_hidden_dim': 256,
        'num_transformer_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        
        # 训练
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'constraint_weight': args.constraint_weight,
        'det_weight': args.det_weight,
        'accum_steps': args.accum,
        'constraint_mode': args.constraint_mode,
        'val_interval': args.val_interval,
        'skip_dla': args.skip_dla,
        'early_stop_patience': args.early_stop,
        'rel_train_mode': args.rel_train_mode,
        'use_edge_features': not args.no_edge_features,  # 是否使用边特征
        
        # 保存
        'save_dir': save_dir,
    }
    
    print("=" * 60)
    print("gDSA 训练 (带约束 Loss)")
    print("=" * 60)
    print(f"Backbone: {args.backbone or 'yolo11n'}")
    print("模型类型: transformer")
    print(f"YOLO路径: {config['yolo_model_path']}")
    print(f"Epochs: {config['epochs']}")
    print(f"约束模式: {config['constraint_mode']}")
    print(f"约束权重: {config['constraint_weight']}")
    print(f"梯度累积: {config['accum_steps']}")
    print(f"验证间隔: 每 {config['val_interval']} 个 epoch")
    print(f"图片尺寸: {'原始尺寸(max=' + str(config['max_size']) + ')' if config['use_original_size'] else config['img_size']}")
    print(f"边特征: {'禁用 (对照实验)' if args.no_edge_features else '启用'}")
    print(f"保存目录: {config['save_dir']}")
    print("=" * 60)
    
    trainer = ConstraintTrainer(config)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n⚠️ 收到中断信号，保存中...")
        trainer.save_history()
        trainer.save_checkpoint('interrupted.pth')
        print("✅ 已保存")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    trainer.train()


if __name__ == "__main__":
    main()
