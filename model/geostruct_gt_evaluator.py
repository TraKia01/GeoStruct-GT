"""Evaluation utilities for GeoStruct-GT."""

import numpy as np
import torch


class gDSAEvaluator:
    """Evaluator for relation recall and AP."""

    def __init__(self, num_relations=8, iou_threshold=0.5, relation_thresholds=None, topk_per_relation=2000):
        self.num_relations = num_relations
        self.iou_threshold = iou_threshold
        self.relation_thresholds = relation_thresholds or [0.5, 0.75]
        self.topk_per_relation = topk_per_relation
        self.reset()

    def reset(self):
        self.pred_records = {r: [] for r in range(self.num_relations)}
        self.num_gt = {r: 0 for r in range(self.num_relations)}
        self.sample_counter = 0

    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        inter_xmin = max(x1_1, x1_2)
        inter_ymin = max(y1_1, y1_2)
        inter_xmax = min(x2_1, x2_2)
        inter_ymax = min(y2_1, y2_2)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        return inter_area / (union_area + 1e-6)

    def instance_matching(self, pred_boxes, pred_classes, gt_boxes, gt_classes):
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)
        if n_pred == 0 or n_gt == 0:
            return {}

        iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i in range(n_pred):
            for j in range(n_gt):
                if pred_classes[i] == gt_classes[j]:
                    iou_matrix[i, j] = self.compute_iou(pred_boxes[i], gt_boxes[j])

        matching = {}
        matching_iou = {}
        for pred_idx in range(n_pred):
            best_gt_idx = -1
            best_iou = 0.0
            for gt_idx in range(n_gt):
                iou = iou_matrix[pred_idx, gt_idx]
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx != -1:
                if best_gt_idx not in matching or best_iou > matching_iou[best_gt_idx]:
                    matching[best_gt_idx] = pred_idx
                    matching_iou[best_gt_idx] = best_iou
        return matching

    def _extract_relations_from_dense(self, pred_rel_probs, threshold=0.0):
        if hasattr(pred_rel_probs, 'device'):
            probs = pred_rel_probs.detach().cpu()
        else:
            probs = torch.tensor(pred_rel_probs)

        n = probs.shape[0]
        num_relations = probs.shape[2]
        pred_relations = []
        for rel_type in range(num_relations):
            rel_scores = probs[:, :, rel_type]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    score = float(rel_scores[i, j].item())
                    if score > threshold:
                        pred_relations.append((i, j, rel_type, score))
        return pred_relations

    def add_sample(self, pred_boxes, pred_classes, pred_relations, pred_scores,
                   gt_boxes, gt_classes, gt_relations, pred_rel_probs=None):
        sample_id = self.sample_counter
        self.sample_counter += 1

        if pred_rel_probs is not None:
            pred_relations = self._extract_relations_from_dense(pred_rel_probs)

        matching = self.instance_matching(pred_boxes, pred_classes, gt_boxes, gt_classes)
        inv_matching = {v: k for k, v in matching.items()}

        gt_rel_by_type = {r: set() for r in range(self.num_relations)}
        for src_gt, tgt_gt, rel_type in gt_relations:
            if 0 <= rel_type < self.num_relations:
                gt_rel_by_type[rel_type].add((src_gt, tgt_gt))

        for rel_type in range(self.num_relations):
            self.num_gt[rel_type] += len(gt_rel_by_type[rel_type])

        for src_pred, tgt_pred, rel_type, score in pred_relations:
            if not (0 <= rel_type < self.num_relations):
                continue
            if src_pred not in inv_matching or tgt_pred not in inv_matching:
                continue
            src_gt = inv_matching[src_pred]
            tgt_gt = inv_matching[tgt_pred]
            is_tp = (src_gt, tgt_gt) in gt_rel_by_type[rel_type]
            gt_pair = (sample_id, src_gt, tgt_gt) if is_tp else None
            self.pred_records[rel_type].append((float(score), bool(is_tp), gt_pair))

    def compute_metrics(self):
        metrics = {}
        for threshold in self.relation_thresholds:
            recalls = []
            aps = []
            for rel_type in range(self.num_relations):
                n_gt = self.num_gt[rel_type]
                if n_gt == 0:
                    continue

                matched_gt_pairs = set()
                for score, is_tp, gt_pair in self.pred_records[rel_type]:
                    if score > threshold and is_tp and gt_pair is not None:
                        matched_gt_pairs.add(gt_pair)
                recalls.append(len(matched_gt_pairs) / n_gt)

                filtered = [
                    (score, is_tp, gt_pair)
                    for score, is_tp, gt_pair in self.pred_records[rel_type]
                    if score > threshold
                ]
                if not filtered:
                    aps.append(0.0)
                    continue

                filtered.sort(key=lambda x: -x[0])
                tps = 0
                fps = 0
                precisions = []
                recalls_curve = []
                matched_gt_in_ap = set()

                for score, is_tp, gt_pair in filtered:
                    if is_tp and gt_pair is not None and gt_pair not in matched_gt_in_ap:
                        tps += 1
                        matched_gt_in_ap.add(gt_pair)
                    else:
                        fps += 1
                    precisions.append(tps / (tps + fps))
                    recalls_curve.append(tps / n_gt)

                precisions = np.array(precisions, dtype=np.float32)
                for i in range(len(precisions) - 2, -1, -1):
                    precisions[i] = max(precisions[i], precisions[i + 1])

                ap = 0.0
                prev_r = 0.0
                for rec, prec in zip(recalls_curve, precisions):
                    if rec > prev_r:
                        ap += (rec - prev_r) * prec
                        prev_r = rec
                aps.append(float(ap))

            metrics[f'mR_g@{threshold}'] = float(np.mean(recalls)) if recalls else 0.0
            metrics[f'mAP_g@{threshold}'] = float(np.mean(aps)) if aps else 0.0
        return metrics
