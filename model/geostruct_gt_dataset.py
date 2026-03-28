"""Dataset for GeoStruct-GT."""

import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class YOLOgDSADataset(Dataset):
    """Dataset loader."""
    
    def __init__(self,
                 img_dir,
                 yolo_dir,
                 spatial_rel_dir=None,
                 logic_rel_dir=None,
                 classes_file=None,
                 mode='train',
                 use_spatial=True,
                 use_logic=True,
                 img_size=640,
                 use_full_graph=False,
                 use_original_size=False,
                 max_size=1280):
        
        self.img_dir = img_dir
        self.yolo_dir = yolo_dir
        self.spatial_rel_dir = spatial_rel_dir
        self.logic_rel_dir = logic_rel_dir
        self.mode = mode
        self.use_spatial = use_spatial
        self.use_logic = use_logic
        self.img_size = img_size
        self.use_full_graph = use_full_graph  # 是否使用全连接图（包含负样本）
        self.use_original_size = use_original_size  # 是否使用原始图片大小
        self.max_size = max_size if max_size > 0 else 99999  # 原始尺寸模式下的最大边长限制
        
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ['handwritten', 'headword', 'figure', 'source', 
                          'form', 'pinyin', 'meaning', 'title']
        
        self.num_classes = len(self.classes)
        
        self.spatial_rel_types = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }
        
        self.logic_rel_types = {
            4: 'parent',
            5: 'child',
            6: 'sequence',
            7: 'reference'
        }
        
        self.img_files = self._get_image_files()
        
        print(f"📊 加载 {mode} 数据集: {len(self.img_files)} 张图片")
        print(f"   - 节点类别数: {self.num_classes}")
        print(f"   - 空间关系类型: {len(self.spatial_rel_types)}")
        print(f"   - 逻辑关系类型: {len(self.logic_rel_types)}")
        print(f"   - 图片尺寸: {'原始尺寸' if use_original_size else img_size}")
        print(f"   - 使用全连接图: {'是' if use_full_graph else '否'} {'(包含负样本)' if use_full_graph else ''}")
        print(f"   - 预处理: {'直接resize到原始尺寸' if use_original_size else 'letterbox (保持比例)'}")
    
    def _letterbox(self, img, new_size=640):
        """
        Letterbox resize - 保持比例，填充灰边
        和 YOLO 官方预处理一致
        """
        h, w = img.shape[:2]
        ratio = min(new_size / h, new_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = (new_size - new_w) // 2
        pad_h = (new_size - new_h) // 2
        
        img_padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
        
        return img_padded, ratio, (pad_w, pad_h)
    
    def _get_image_files(self):
        """获取所有图片文件"""
        img_ext = ('.jpg', '.jpeg', '.png')
        img_files = []
        
        for f in os.listdir(self.img_dir):
            if f.lower().endswith(img_ext):
                img_files.append(f)
        
        return sorted(img_files)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        返回单个样本
        
        Returns:
            dict: {
                'image': [3, H, W] 图像tensor
                'boxes': [N, 4] 归一化bbox (x1, y1, x2, y2) - 原图归一化坐标
                'classes': [N] 类别ID
                'edge_index': [2, E] 边索引
                'edge_labels': [E, K] 多标签边类型 (0/1), K=num_relations
                'img_path': str 图片路径
                'img_size': (H, W) 原始图像尺寸
            }
        """
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_id = Path(img_name).stem
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        orig_h, orig_w = image.shape[:2]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_original_size:
            max_side = max(orig_h, orig_w)
            if max_side > self.max_size:
                scale = self.max_size / max_side
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
            else:
                new_h, new_w = orig_h, orig_w
            
            new_h = ((new_h + 31) // 32) * 32
            new_w = ((new_w + 31) // 32) * 32
            
            if new_h != orig_h or new_w != orig_w:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image_resized, ratio, (pad_w, pad_h) = self._letterbox(image, self.img_size)
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        yolo_path = os.path.join(self.yolo_dir, f"{img_id}.txt")
        boxes, classes, yolo_boxes = self._load_yolo_annotations(yolo_path)
        
        if len(boxes) == 0:
            return {
                'image': image_tensor,
                'boxes': torch.zeros(0, 4),
                'yolo_boxes': torch.zeros(0, 4),
                'classes': torch.zeros(0, dtype=torch.long),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_labels': torch.zeros(0, len(self.spatial_rel_types) + len(self.logic_rel_types), dtype=torch.float32),
                'img_path': img_path,
                'img_size': (orig_h, orig_w)
            }
        
        edge_index, edge_labels = self._load_relations(img_id, len(boxes))
        
        return {
            'image': image_tensor,
            'boxes': boxes,  # 原图归一化坐标
            'yolo_boxes': yolo_boxes,
            'classes': classes,
            'edge_index': edge_index,
            'edge_labels': edge_labels,
            'img_path': img_path,
            'img_size': (orig_h, orig_w)
        }
    
    def _load_yolo_annotations(self, yolo_path):
        """
        读取YOLO标注
        
        Returns:
            boxes: [N, 4] 归一化bbox (x1, y1, x2, y2)
            classes: [N] 类别ID
            yolo_boxes: [N, 4] 归一化bbox (cx, cy, w, h)
        """
        boxes = []
        yolo_boxes = []
        classes = []
        
        if not os.path.exists(yolo_path):
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), torch.zeros(0, 4)
        
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    yolo_boxes.append([cx, cy, w, h])
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(class_id)
        
        if len(boxes) == 0:
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), torch.zeros(0, 4)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        yolo_boxes = torch.tensor(yolo_boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        
        return boxes, classes, yolo_boxes
    
    def _load_relations(self, img_id, num_nodes):
        """
        读取关系标注
        
        Returns:
            edge_index: [2, E] 边索引
            edge_labels: [E, K] 多标签边类型 (0/1)
        """
        num_relations = len(self.spatial_rel_types) + len(self.logic_rel_types)

        rel_dict = {}

        if self.use_spatial and self.spatial_rel_dir:
            spatial_path = os.path.join(self.spatial_rel_dir, f"{img_id}.json")
            if os.path.exists(spatial_path):
                with open(spatial_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rel in data.get('spatial_rels', []):
                        src = rel['source_idx']
                        tgt = rel['target_idx']
                        rel_id = rel['rel_id']

                        if src < num_nodes and tgt < num_nodes and 0 <= rel_id < num_relations:
                            key = (src, tgt)
                            if key not in rel_dict:
                                rel_dict[key] = [0] * num_relations
                            rel_dict[key][rel_id] = 1

        if self.use_logic and self.logic_rel_dir:
            logic_path = os.path.join(self.logic_rel_dir, f"{img_id}.json")
            if os.path.exists(logic_path):
                with open(logic_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rel in data.get('logic_rels', []):
                        src = rel['source_idx']
                        tgt = rel['target_idx']
                        rel_id = rel['rel_id']

                        if src < num_nodes and tgt < num_nodes and 0 <= rel_id < num_relations:
                            key = (src, tgt)
                            if key not in rel_dict:
                                rel_dict[key] = [0] * num_relations
                            rel_dict[key][rel_id] = 1

        if self.use_full_graph and num_nodes > 0:
            all_edges = []
            all_labels = []

            for src in range(num_nodes):
                for tgt in range(num_nodes):
                    if src == tgt:
                        continue
                    all_edges.append([src, tgt])
                    key = (src, tgt)
                    if key in rel_dict:
                        all_labels.append(rel_dict[key])
                    else:
                        all_labels.append([0] * num_relations)

            if len(all_edges) == 0:
                return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, num_relations, dtype=torch.float32)

            edge_index = torch.tensor(all_edges, dtype=torch.long).t()  # [2, E]
            edge_labels = torch.tensor(all_labels, dtype=torch.float32)  # [E, K]
            return edge_index, edge_labels

        if len(rel_dict) == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, num_relations, dtype=torch.float32),
            )

        edges = []
        labels = []
        for (src, tgt), vec in rel_dict.items():
            edges.append([src, tgt])
            labels.append(vec)

        edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
        edge_labels = torch.tensor(labels, dtype=torch.float32)  # [E, K]

        return edge_index, edge_labels


def collate_fn(batch):
    """
    自定义collate函数，处理不同数量的节点和边
    
    注意：这里假设batch_size=1（每次处理一张图）
    如果需要batch>1，需要更复杂的处理
    """
    if len(batch) > 1:
        raise NotImplementedError("当前只支持batch_size=1")
    
    return batch[0]

