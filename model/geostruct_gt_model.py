# geostruct_gt_model.py - 精简版：只保留实际使用的模块
"""
纯视觉版本的 YOLO + Graph Transformer gDSA 模型
用于 geostruct_gt_train.py

特点：
- 节点特征：只有视觉特征（无节点几何特征）
- 边特征：保留边几何特征
- 支持多种 backbone（YOLO11, ResNet50, ConvNeXt 等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
import torchvision.ops as ops
import math


# ============ 特征提取器 ============

class YOLONeckFeatureExtractor(nn.Module):
    """从 YOLO 模型中提取 Neck 输出的特征（已融合的多尺度特征）"""
    
    def __init__(self, yolo_model, freeze=False):
        super().__init__()
        
        self.__dict__['_yolo_model'] = yolo_model
        
        # 找到 Detect 层及其输入层索引
        from ultralytics.nn.modules.head import Detect
        
        self.detect_layer_idx = None
        self.neck_feature_indices = []
        
        for i, module in enumerate(self._yolo_model.model):
            if isinstance(module, Detect):
                self.detect_layer_idx = i
                if hasattr(module, 'f') and isinstance(module.f, list):
                    self.neck_feature_indices = module.f
                break
        
        if self.detect_layer_idx is None:
            raise ValueError("未找到 Detect 层")
        
        print(f"✅ YOLO Neck 特征提取器")
        print(f"   Detect 层索引: {self.detect_layer_idx}")
        print(f"   Neck 输出层索引: {self.neck_feature_indices}")
        
        if freeze:
            self.freeze_backbone()
        else:
            print("🔓 Backbone + Neck 可训练")
    
    @property
    def _yolo_model(self):
        return self.__dict__['_yolo_model']
    
    def freeze_backbone(self):
        """冻结 Backbone + Neck"""
        from ultralytics.nn.modules.head import Detect
        
        for module in self._yolo_model.model:
            if not isinstance(module, Detect):
                for param in module.parameters():
                    param.requires_grad = False
        print("🔒 Backbone + Neck 已冻结")
    
    def unfreeze_backbone(self):
        """解冻所有参数"""
        for param in self._yolo_model.parameters():
            param.requires_grad = True
        print("🔓 Backbone + Neck 已解冻")
    
    def forward(self, x):
        """提取 Neck 输出的多尺度特征"""
        from ultralytics.nn.modules.head import Detect
        
        y = []
        
        for i, m in enumerate(self._yolo_model.model):
            # 处理输入来源
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f] if m.f < len(y) else x
                else:
                    x = [y[j] if j < len(y) else x for j in m.f]
            
            x = m(x)
            y.append(x)
            
            if isinstance(m, Detect):
                break
        
        # 收集 Neck 输出的特征
        neck_features = []
        for idx in self.neck_feature_indices:
            if idx < len(y):
                feat = y[idx]
                if isinstance(feat, torch.Tensor) and feat.dim() == 4:
                    neck_features.append(feat)
        
        if len(neck_features) == 0:
            raise RuntimeError(f"未能提取 Neck 特征")
        
        return neck_features


class RoIFeatureExtractor(nn.Module):
    """从特征图中提取 RoI 特征（单尺度版本）"""
    
    def __init__(self, output_dim=256, roi_size=7):
        super().__init__()
        self.output_dim = output_dim
        self.roi_size = roi_size
        self.feature_proj = None  # 动态初始化
    
    def _init_projection(self, in_features, device):
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, self.output_dim * 2),
            nn.LayerNorm(self.output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        ).to(device)
    
    def forward(self, features, boxes, img_size=640):
        if len(boxes) == 0:
            device = features[0].device
            return torch.zeros(0, self.output_dim, device=device)
        
        feature_map = features[-1]
        B, C, H, W = feature_map.shape
        
        boxes_abs = boxes.clone()
        boxes_abs[:, [0, 2]] *= W
        boxes_abs[:, [1, 3]] *= H
        
        batch_idx = torch.zeros(len(boxes), device=boxes.device)
        rois = torch.cat([batch_idx.unsqueeze(1), boxes_abs], dim=1)
        
        roi_features = ops.roi_align(
            feature_map, rois,
            output_size=self.roi_size,
            spatial_scale=1.0,
            aligned=True
        )
        
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        if self.feature_proj is None:
            in_channels = roi_features.size(1)
            self._init_projection(in_channels, roi_features.device)
            print(f"🔧 动态初始化 RoI 投影层: {in_channels} -> {self.output_dim}")
        
        return self.feature_proj(roi_features)


class EdgeGeometricExtractor(nn.Module):
    """提取边的几何特征"""
    
    def __init__(self, output_dim=64):
        super().__init__()
        self.input_norm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, boxes, edge_index):
        if edge_index.size(1) == 0:
            return torch.zeros(0, self.mlp[-2].out_features, device=boxes.device)
        
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        
        src_cx, src_cy = cx[src_idx], cy[src_idx]
        tgt_cx, tgt_cy = cx[tgt_idx], cy[tgt_idx]
        src_w, src_h = w[src_idx], h[src_idx]
        tgt_w, tgt_h = w[tgt_idx], h[tgt_idx]
        
        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy
        distance = torch.sqrt(dx**2 + dy**2 + 1e-6)
        angle = torch.atan2(dy, dx) / math.pi
        
        src_area = src_w * src_h
        tgt_area = tgt_w * tgt_h
        size_ratio = torch.log(tgt_area / src_area + 1e-6)
        
        src_aspect = src_w / src_h
        tgt_aspect = tgt_w / tgt_h
        aspect_diff = torch.log(tgt_aspect / src_aspect + 1e-6)
        
        w_ratio = torch.log(tgt_w / src_w + 1e-6)
        h_ratio = torch.log(tgt_h / src_h + 1e-6)
        
        edge_features = torch.stack([
            dx, dy, distance, angle, size_ratio, aspect_diff, w_ratio, h_ratio
        ], dim=1)
        
        edge_features = self.input_norm(edge_features)
        return self.mlp(edge_features)


# ============ Graph Transformer ============

class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer"""
    
    def __init__(self, hidden_dim, num_heads=8, edge_dim=64, dropout=0.1, use_edge_bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias
        
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 只在使用边特征时创建投影层
        if use_edge_bias:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        else:
            self.edge_proj = None
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        
        if N == 0:
            return x
        
        residual = x
        x = self.norm1(x)
        
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) / self.scale
        
        if edge_index.size(1) > 0:
            adj_mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
            adj_mask[edge_index[0], edge_index[1]] = True
            adj_mask.fill_diagonal_(True)
            
            # 只在启用边特征时添加注意力偏置
            if self.use_edge_bias and self.edge_proj is not None and edge_attr is not None:
                edge_bias = self.edge_proj(edge_attr)
                attn_bias = torch.zeros(N, N, self.num_heads, device=x.device)
                attn_bias[edge_index[0], edge_index[1]] = edge_bias
                attn_scores = attn_scores + attn_bias
            
            attn_scores = attn_scores.masked_fill(~adj_mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.einsum('nmh,mhd->nhd', attn_weights, V)
        out = out.reshape(N, self.hidden_dim)
        out = self.out_proj(out)
        
        x = residual + self.dropout(out)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x


class GraphTransformer(nn.Module):
    """Graph Transformer for relation prediction"""
    
    def __init__(self, 
                 node_dim,
                 hidden_dim=256,
                 edge_dim=64,
                 num_layers=4,
                 num_heads=8,
                 num_relations=8,
                 dropout=0.1,
                 use_edge_features=True):  # 新增参数
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.use_edge_features = use_edge_features  # 保存配置
        
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, edge_dim, dropout, use_edge_bias=use_edge_features)
            for _ in range(num_layers)
        ])
        
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))
        
        # 根据是否使用边特征调整输入维度
        classifier_input_dim = hidden_dim * 2 + (edge_dim if use_edge_features else 0)
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations)
        )
        
        self.edge_existence = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, node_features, edge_index, edge_attr):
        N = node_features.size(0)
        E = edge_index.size(1)
        
        if N == 0 or E == 0:
            device = node_features.device
            return (torch.zeros(E, self.num_relations, device=device),
                    torch.zeros(E, device=device))
        
        x = self.node_proj(node_features)
        
        layer_outputs = [x]
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
        
        weights = F.softmax(self.layer_weights, dim=0)
        x = sum(w * feat for w, feat in zip(weights, layer_outputs))
        
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        src_feat = x[src_idx]
        tgt_feat = x[tgt_idx]
        
        # 根据配置决定是否使用边特征
        if self.use_edge_features and edge_attr is not None:
            edge_repr = torch.cat([src_feat, tgt_feat, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([src_feat, tgt_feat], dim=1)
        
        relation_logits = self.edge_classifier(edge_repr)
        exist_logits = self.edge_existence(edge_repr).squeeze(-1)
        
        return relation_logits, exist_logits


# ============ 主模型 ============

class YOLOgDSATransformerVisualOnly(nn.Module):
    """
    纯视觉版本的 YOLO + Graph Transformer gDSA 模型
    
    特点：
    - 节点特征：只有视觉特征（无节点几何特征）
    - 边特征：保留边几何特征
    - 支持多种 backbone
    """
    
    def __init__(self,
                 yolo_model_path='yolo11n.pt',
                 num_classes=8,
                 num_relations=8,
                 visual_feature_dim=256,
                 edge_geometric_dim=64,
                 transformer_hidden_dim=256,
                 num_transformer_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 use_neck_features=True,
                 use_edge_features=True):  # 新增参数
        super().__init__()
        
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.use_neck_features = use_neck_features
        self.use_edge_features = use_edge_features  # 保存配置
        
        # 1. YOLO 模型
        yolo = YOLO(yolo_model_path)
        
        if isinstance(yolo_model_path, str) and yolo_model_path.lower().endswith(('.yaml', '.yml')):
            cfg_dict = yaml_model_load(yolo_model_path)
            yolo.model = DetectionModel(cfg_dict, nc=num_classes, verbose=False)
            from types import SimpleNamespace
            from ultralytics.cfg import DEFAULT_CFG_DICT
            yolo.model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            print(f"✅ 从 YAML 创建 YOLO 模型: {yolo_model_path}, nc={num_classes}")
        else:
            model_nc = yolo.model.nc if hasattr(yolo.model, 'nc') else 80
            if model_nc != num_classes:
                print(f"⚠️ 预训练权重类别数 ({model_nc}) 与目标类别数 ({num_classes}) 不匹配")
                cfg_path = 'ultralytics/cfg/models/11/yolo11n.yaml'
                cfg_dict = yaml_model_load(cfg_path)
                new_model = DetectionModel(cfg_dict, nc=num_classes, verbose=False)
                
                old_state = yolo.model.state_dict()
                new_state = new_model.state_dict()
                transferred = 0
                for key in new_state.keys():
                    if key in old_state and old_state[key].shape == new_state[key].shape:
                        new_state[key] = old_state[key]
                        transferred += 1
                new_model.load_state_dict(new_state)
                
                from types import SimpleNamespace
                from ultralytics.cfg import DEFAULT_CFG_DICT
                new_model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
                yolo.model = new_model
                print(f"✅ 迁移了 {transferred} 个参数")
            else:
                print(f"✅ 加载 YOLO 预训练权重: {yolo_model_path}, nc={num_classes}")
        
        # 确保 args 是 SimpleNamespace
        from types import SimpleNamespace
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if not hasattr(yolo.model, 'args') or isinstance(yolo.model.args, dict):
            yolo.model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        
        self.det_model = yolo.model
        yolo.model = self.det_model
        self.__dict__['_yolo'] = yolo
        
        # 2. 特征提取器
        self.feature_extractor = YOLONeckFeatureExtractor(self.det_model, freeze=False)
        self.backbone_extractor = self.feature_extractor
        print(f"✅ 使用 Neck 特征（已融合的多尺度特征）")
        
        # 3. RoI 特征提取
        self.roi_extractor = RoIFeatureExtractor(output_dim=visual_feature_dim)
        print(f"✅ 使用单尺度 RoI 特征提取 (dim={visual_feature_dim})")
        
        # 4. 边几何特征提取
        if use_edge_features:
            self.edge_geometric_extractor = EdgeGeometricExtractor(output_dim=edge_geometric_dim)
            print(f"✅ 使用边几何特征 (dim={edge_geometric_dim})")
        else:
            self.edge_geometric_extractor = None
            print(f"⚠️ 不使用边几何特征（对照实验）")
        
        # 5. Graph Transformer
        self.graph_transformer = GraphTransformer(
            node_dim=visual_feature_dim,
            hidden_dim=transformer_hidden_dim,
            edge_dim=edge_geometric_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            num_relations=num_relations,
            dropout=dropout,
            use_edge_features=use_edge_features  # 传递参数
        )
        
        print(f"✅ 纯视觉 Graph Transformer 初始化完成")
        print(f"   节点特征维度: {visual_feature_dim} (纯视觉)")
        print(f"   边特征维度: {edge_geometric_dim if use_edge_features else 0} {'(已禁用)' if not use_edge_features else ''}")
        print(f"   隐藏层维度: {transformer_hidden_dim}")
        print(f"   Transformer 层数: {num_transformer_layers}")
    
    def train(self, mode=True):
        super().train(mode)
        self.det_model.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self, images, gt_boxes=None, gt_classes=None, edge_index=None, conf_thresh=0.25, max_nodes=200):
        B = images.size(0)
        assert B == 1, "目前只支持 batch_size=1"
        
        features = self.feature_extractor(images)
        device = images.device
        
        conf = None
        if gt_boxes is not None:
            boxes = gt_boxes
            classes = gt_classes
        else:
            boxes, classes, conf = self._detect_with_det_model(images, conf_thresh)
            if max_nodes is not None and len(boxes) > int(max_nodes):
                if conf is not None:
                    topk = torch.topk(conf, k=int(max_nodes), largest=True).indices
                    boxes = boxes[topk]
                    classes = classes[topk]
                    conf = conf[topk]
                else:
                    boxes = boxes[:int(max_nodes)]
                    classes = classes[:int(max_nodes)]
            if len(boxes) == 0:
                return {
                    'relation_logits': torch.zeros(0, self.num_relations, device=device),
                    'exist_logits': torch.zeros(0, device=device),
                    'boxes': boxes,
                    'classes': classes,
                    'conf': None,
                    'edge_index': torch.zeros(2, 0, dtype=torch.long, device=device)
                }
        
        N = len(boxes)
        
        # 构建全连接图
        if edge_index is None:
            src = torch.arange(N, device=device).repeat_interleave(N)
            tgt = torch.arange(N, device=device).repeat(N)
            mask = src != tgt
            edge_index = torch.stack([src[mask], tgt[mask]], dim=0)
        
        # 提取节点特征（纯视觉）
        node_features = self.roi_extractor(features, boxes)
        
        # 提取边特征（如果启用）
        if self.use_edge_features and self.edge_geometric_extractor is not None:
            edge_geometric_features = self.edge_geometric_extractor(boxes, edge_index)
        else:
            # 不使用边特征时，传入 None 或零张量
            edge_geometric_features = None
        
        # Graph Transformer
        relation_logits, exist_logits = self.graph_transformer(
            node_features, edge_index, edge_geometric_features
        )
        
        return {
            'relation_logits': relation_logits,
            'exist_logits': exist_logits,
            'boxes': boxes,
            'classes': classes,
            'conf': conf,
            'edge_index': edge_index
        }
    
    def _detect_with_det_model(self, images, conf_thresh=0.25):
        """检测推理"""
        try:
            from ultralytics.utils.ops import non_max_suppression
        except ImportError:
            from ultralytics.utils.nms import non_max_suppression
        device = images.device
        
        was_training = self.det_model.training
        self.det_model.eval()
        with torch.no_grad():
            preds = self.det_model(images)
        if was_training:
            self.det_model.train()
        
        results = non_max_suppression(
            preds, conf_thres=conf_thresh, iou_thres=0.45,
            classes=None, max_det=300, nc=self.num_classes
        )
        
        det = results[0]
        if det is None or len(det) == 0:
            return (torch.zeros(0, 4, device=device),
                    torch.zeros(0, dtype=torch.long, device=device), None)
        
        _, _, H, W = images.shape
        boxes = det[:, :4].clone()
        boxes[:, [0, 2]] /= W
        boxes[:, [1, 3]] /= H
        boxes = boxes.clamp(0, 1)
        
        return boxes, det[:, 5].long(), det[:, 4]
    
    def detect_and_get_raw_preds(self, images, conf_thresh=0.25):
        """
        检测并返回原始预测（用于复用前向传播结果）
        
        Returns:
            pred_boxes: 检测框 [N, 4] (normalized xyxy)
            pred_classes: 类别 [N]
            pred_scores: 置信度 [N]
            raw_preds: 原始预测（用于计算 det_loss）
        """
        try:
            from ultralytics.utils.ops import non_max_suppression
        except ImportError:
            from ultralytics.utils.nms import non_max_suppression
        device = images.device
        
        # 一次前向传播，保存原始预测
        was_training = self.det_model.training
        self.det_model.eval()
        with torch.no_grad():
            raw_preds = self.det_model(images)
        if was_training:
            self.det_model.train()
        
        # NMS 获取检测框
        results = non_max_suppression(
            raw_preds, conf_thres=conf_thresh, iou_thres=0.45,
            classes=None, max_det=300, nc=self.num_classes
        )
        
        det = results[0]
        if det is None or len(det) == 0:
            pred_boxes = torch.zeros(0, 4, device=device)
            pred_classes = torch.zeros(0, dtype=torch.long, device=device)
            pred_scores = None
        else:
            _, _, H, W = images.shape
            boxes = det[:, :4].clone()
            boxes[:, [0, 2]] /= W
            boxes[:, [1, 3]] /= H
            pred_boxes = boxes.clamp(0, 1)
            pred_classes = det[:, 5].long()
            pred_scores = det[:, 4]
        
        return pred_boxes, pred_classes, pred_scores, raw_preds
    
    def compute_det_loss(self, images, yolo_boxes_xywh, yolo_classes):
        """计算 YOLO 检测 loss"""
        device = images.device
        if yolo_boxes_xywh.numel() == 0:
            return torch.tensor(0.0, device=device), torch.zeros(3, device=device)
        
        batch_idx = torch.zeros(yolo_boxes_xywh.size(0), device=device)
        det_batch = {
            'img': images,
            'bboxes': yolo_boxes_xywh,
            'cls': yolo_classes,
            'batch_idx': batch_idx,
        }
        loss_vec, loss_items = self.det_model(det_batch)
        loss = loss_vec.sum() if loss_vec.numel() else loss_vec
        return loss, loss_items
    
    def freeze_yolo(self):
        for param in self.det_model.parameters():
            param.requires_grad = False
        print("🔒 YOLO 已冻结")
    
    def unfreeze_yolo(self):
        for param in self.det_model.parameters():
            param.requires_grad = True
        print("🔓 YOLO 已解冻")
    
    @property
    def yolo(self):
        return self._yolo


# ============ 向后兼容别名 ============
# 为了兼容旧代码，提供别名
YOLOgDSATransformerModel = YOLOgDSATransformerVisualOnly
