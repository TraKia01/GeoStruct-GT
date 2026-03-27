"""
GeoStruct-GT 推理脚本

特点：
1. 仅支持公开版 Transformer 模型
2. 不做图约束后处理
3. 输出原始关系预测结果
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch

from geostruct_gt_model import YOLOgDSATransformerVisualOnly


RELATION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Parent', 'Child', 'Sequence', 'Reference']
CLASS_NAMES = ['handwritten', 'headword', 'figure', 'source', 'form', 'pinyin', 'meaning', 'title']
BOX_COLOR = (200, 180, 100)
RELATION_COLORS_BGR = {
    'Up': (0, 165, 255),
    'Down': (0, 165, 255),
    'Left': (0, 165, 255),
    'Right': (0, 165, 255),
    'Parent': (0, 215, 255),
    'Child': (255, 144, 30),
    'Sequence': (255, 0, 255),
    'Reference': (128, 0, 128),
}


def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    ratio = min(new_size / h, new_size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_size - new_w) // 2
    pad_h = (new_size - new_h) // 2
    img_padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
    return img_padded, ratio, (pad_w, pad_h)


def resize_to_original_size(img, max_size=1280):
    h, w = img.shape[:2]
    if max_size > 0:
        scale = min(max_size / max(h, w), 1.0)
        h = int(h * scale)
        w = int(w * scale)
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img_resized, (new_h, new_w)


def scale_boxes_back(boxes, ratio, pad, orig_size, img_size=640):
    h, w = orig_size
    pad_w, pad_h = pad
    boxes_pixel = boxes.clone()
    boxes_pixel[:, [0, 2]] *= img_size
    boxes_pixel[:, [1, 3]] *= img_size
    boxes_pixel[:, [0, 2]] = (boxes_pixel[:, [0, 2]] - pad_w) / ratio
    boxes_pixel[:, [1, 3]] = (boxes_pixel[:, [1, 3]] - pad_h) / ratio
    boxes_pixel[:, [0, 2]] = boxes_pixel[:, [0, 2]].clamp(0, w)
    boxes_pixel[:, [1, 3]] = boxes_pixel[:, [1, 3]].clamp(0, h)
    return boxes_pixel


def load_model(checkpoint_path, device):
    print(f"📦 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    yolo_model_path = config.get('yolo_model_path', 'yolo11n.pt')

    model = YOLOgDSATransformerVisualOnly(
        yolo_model_path=yolo_model_path,
        num_classes=config.get('num_classes', 8),
        num_relations=config.get('num_relations', 8),
        visual_feature_dim=config.get('visual_feature_dim', 256),
        edge_geometric_dim=config.get('edge_geometric_dim', 64),
        transformer_hidden_dim=config.get('transformer_hidden_dim', 256),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_heads=config.get('num_heads', 8),
        dropout=0.0,
        use_neck_features=True,
        use_edge_features=config.get('use_edge_features', True),
    )

    roi_proj_key = 'roi_extractor.feature_proj.0.weight'
    if roi_proj_key in state_dict:
        in_features = state_dict[roi_proj_key].shape[1]
        model.roi_extractor._init_projection(in_features, device)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, config


def inference(model, image_path, device, conf_thresh=0.25, rel_thresh=0.3,
              img_size=640, use_original_size=False, max_size=1280):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_original_size:
        image_resized, (new_h, new_w) = resize_to_original_size(image_rgb, max_size)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        ratio = None
        pad = None
    else:
        image_resized, ratio, pad = letterbox(image_rgb, img_size)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, conf_thresh=conf_thresh)

    boxes = output['boxes']
    classes = output['classes']
    edge_index = output['edge_index']
    rel_logits = output['relation_logits']
    exist_logits = output['exist_logits']
    conf = output.get('conf', None)

    rel_probs = torch.sigmoid(rel_logits)
    exist_probs = torch.sigmoid(exist_logits)
    final_scores = exist_probs.unsqueeze(1) * rel_probs

    pred_relations = []
    for e in range(edge_index.size(1)):
        src_idx = edge_index[0, e].item()
        tgt_idx = edge_index[1, e].item()
        for rel_type in range(len(RELATION_NAMES)):
            score = final_scores[e, rel_type].item()
            if score > rel_thresh:
                pred_relations.append((src_idx, tgt_idx, rel_type, score))

    if use_original_size:
        boxes_orig = boxes.clone()
        boxes_orig[:, [0, 2]] *= new_w
        boxes_orig[:, [1, 3]] *= new_h
        boxes_orig[:, [0, 2]] *= orig_w / new_w
        boxes_orig[:, [1, 3]] *= orig_h / new_h
        boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clamp(0, orig_w)
        boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clamp(0, orig_h)
    else:
        boxes_orig = scale_boxes_back(boxes, ratio, pad, (orig_h, orig_w), img_size)

    return {
        'boxes': boxes_orig.cpu(),
        'classes': classes.cpu(),
        'conf': conf.cpu() if conf is not None else None,
        'relations': pred_relations,
        'orig_size': (orig_h, orig_w),
    }


def draw_relations(image, results, class_names, output_path):
    boxes = results['boxes'].numpy()
    classes = results['classes'].numpy()
    relations = results['relations']
    conf = results['conf'].numpy() if results['conf'] is not None else None

    centers = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

    for idx, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, _, _ = map(int, box)
        cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f"cls{cls}"
        label = f"[{idx}] {cls_name}"
        if conf is not None:
            label = f"{label}:{conf[idx]:.2f}"
        cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)

    for src, tgt, rel_type, score in relations:
        if src >= len(centers) or tgt >= len(centers):
            continue
        rel_name = RELATION_NAMES[rel_type]
        color = RELATION_COLORS_BGR[rel_name]
        cv2.arrowedLine(image, centers[src], centers[tgt], color, 2, tipLength=0.03)
        mid_x = (centers[src][0] + centers[tgt][0]) // 2
        mid_y = (centers[src][1] + centers[tgt][1]) // 2
        cv2.putText(image, f"{rel_name}:{score:.2f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(output_path, image)
    print(f"💾 结果已保存: {output_path}")


def print_relations(results, class_names):
    boxes = results['boxes'].numpy()
    classes = results['classes'].numpy()
    relations = results['relations']

    print(f"🔍 检测到 {len(boxes)} 个元素")
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f"cls{cls}"
        print(f"  [{i}] {cls_name}: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    rel_counts = defaultdict(int)
    for _, _, rel_type, _ in relations:
        rel_counts[RELATION_NAMES[rel_type]] += 1
    print(f"🔗 原始关系数: {len(relations)}")
    print(f"   分布: {dict(rel_counts)}")


def main():
    parser = argparse.ArgumentParser(description='GeoStruct-GT 推理脚本（无后处理）')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output', type=str, default='./inference_results', help='输出目录')
    parser.add_argument('--classes', type=str, default='./dataset_gdsa/classes.txt', help='类别文件')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='检测置信度阈值')
    parser.add_argument('--rel-thresh', type=float, default=0.3, help='关系分数阈值')
    parser.add_argument('--img-size', type=int, default=None, help='推理图像尺寸')
    parser.add_argument('--use-original-size', action='store_true', help='使用原始图片尺寸')
    parser.add_argument('--max-size', type=int, default=1280, help='原始尺寸模式下的最大边长限制')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.checkpoint, device)

    if os.path.exists(args.classes):
        with open(args.classes, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = CLASS_NAMES

    use_original_size = args.use_original_size or config.get('use_original_size', False)
    max_size = args.max_size if args.max_size != 1280 else config.get('max_size', 1280)
    img_size = args.img_size if args.img_size is not None else config.get('img_size', 640)

    results = inference(
        model,
        args.image,
        device,
        conf_thresh=args.conf_thresh,
        rel_thresh=args.rel_thresh,
        img_size=img_size,
        use_original_size=use_original_size,
        max_size=max_size,
    )

    print_relations(results, class_names)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{Path(args.image).stem}_raw.png")
    image = cv2.imread(args.image)
    draw_relations(image, results, class_names, output_path)


if __name__ == "__main__":
    main()
