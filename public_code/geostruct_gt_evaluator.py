# geostruct_gt_evaluator.py - GeoStruct-GT 任务评估指标（严格按照论文 Algorithm 1）
"""
根据论文 Algorithm 1:

Input:
- I_out: 预测实例
- I_gt: GT实例  
- R: 预测关系 (i_s, p, i_o, s_p) 其中 s_p 是置信度
- R_gt: GT关系 (x_s, p, x_o)
- T_IoU: IoU阈值（用于实例匹配）
- T_R: 关系置信度阈值

Step 1: Instance Matching
- 对每个预测实例找最佳匹配GT（IoU > T_IoU 且类别相同）

Step 2: Relation Evaluation
- G: GT关系集合（全量，不过滤 matching）
- X_T: 预测关系集合（置信度 > T_R 且端点都被匹配的）
- mR_g@T_R: mean recall（每个 GT 边最多被匹配一次）
- mAP_g@T_R: mean average precision（GT 去重，重复命中算 FP）

VAS (Violation-Aware Scoring):
- 对每条边计算违规风险 v_e (范围 [0,1])
- s' = s * exp(-λ * v_e)
- 违规包括：互斥、反对称、多父、2-cycle

VW-mAP (Validity-Weighted mAP):
- VW-mAP = mAP × V_eff
- V_eff = V × C_E^γ（有效合法性）
- V: 纯结构合法性（树/空间/序列约束）
- C_E: 结构覆盖度 = min(1, |E_T| / |G|)
"""

import numpy as np
import torch


# 关系类型常量
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
PARENT, CHILD, SEQUENCE, REFERENCE = 4, 5, 6, 7


class gDSAEvaluator:
    """gDSA任务评估器（严格按照论文 Algorithm 1）"""

    def __init__(self, num_relations=8, iou_threshold=0.5, relation_thresholds=None,
                 topk_per_relation=2000):
        """
        Args:
            num_relations: 关系类别数
            iou_threshold: T_IoU，实例匹配的IoU阈值（论文中通常固定为0.5）
            relation_thresholds: T_R 列表，关系置信度阈值
            topk_per_relation: 从 dense 提取时每类关系保留的 TopK 数量
        """
        self.num_relations = num_relations
        self.iou_threshold = iou_threshold
        self.relation_thresholds = relation_thresholds or [0.5, 0.75]
        self.topk_per_relation = topk_per_relation
        
        self.relation_names = {
            0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right',
            4: 'Parent', 5: 'Child', 6: 'Sequence', 7: 'Reference'
        }
        
        self.reset()

    def reset(self):
        """重置统计信息"""
        # Raw 预测记录（原始分数）: [(score, is_tp, gt_pair), ...]
        # gt_pair = (sample_id, src_gt, tgt_gt) if is_tp else None
        self.pred_records_raw = {r: [] for r in range(self.num_relations)}
        # VAS 预测记录（违规感知分数）
        self.pred_records_vas = {r: [] for r in range(self.num_relations)}
        # GT 数量
        self.num_gt = {r: 0 for r in range(self.num_relations)}
        # 样本计数器（用于生成唯一的 sample_id）
        self.sample_counter = 0



        # Validity-Weighted metrics accumulators (VW-mAP)
        # 方案 A: VW-mAP = mAP × V_eff，其中 V_eff = V × C_E^γ
        # C_E = min(1, |E_T| / (|G| + ε))，结构覆盖度（不是检索召回）
        self.validity_stats = {
            float(thr): {
                'V_tree': [],
                'V_spatial': [],
                'V_seq': [],
                'V': [],       # 纯结构合法性 V@T_R
                'C_E': [],     # 结构覆盖度 C_E@T_R = |E_T| / |G|
                'V_eff': []    # 有效合法性 V_eff = V × C_E^γ
            } for thr in self.relation_thresholds
        }

    def compute_iou(self, box1, box2):
        """计算IoU"""
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
        """Step 1: Instance Matching"""
        N_pred = len(pred_boxes)
        N_gt = len(gt_boxes)

        if N_pred == 0 or N_gt == 0:
            return {}

        iou_matrix = np.zeros((N_pred, N_gt))
        for i in range(N_pred):
            for j in range(N_gt):
                if pred_classes[i] == gt_classes[j]:
                    iou_matrix[i, j] = self.compute_iou(pred_boxes[i], gt_boxes[j])

        matching = {}
        matching_iou = {}

        for pred_idx in range(N_pred):
            best_gt_idx = -1
            best_iou = 0.0
            
            for gt_idx in range(N_gt):
                iou = iou_matrix[pred_idx, gt_idx]
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                if best_gt_idx not in matching or best_iou > matching_iou[best_gt_idx]:
                    matching[best_gt_idx] = pred_idx
                    matching_iou[best_gt_idx] = best_iou

        return matching

    def relation_evaluation(self, pred_relations, gt_relations, matching,
                           pred_rel_probs=None, use_vas=False,
                           lam=5.0, alpha=0.5, beta=0.5, gamma=1.0, sample_id=None):
        """
        Step 2: Relation Evaluation（完全向量化版本）
        
        Args:
            sample_id: 样本 ID（用于跨图片去重 GT）
        """
        # 如果没有提供 sample_id，使用内部计数器
        if sample_id is None:
            sample_id = self.sample_counter
        
        inv_matching = {v: k for k, v in matching.items()}
        
        gt_rel_by_type = {r: set() for r in range(self.num_relations)}
        for src_gt, tgt_gt, rel_type in gt_relations:
            if 0 <= rel_type < self.num_relations:
                gt_rel_by_type[rel_type].add((src_gt, tgt_gt))

        for r in range(self.num_relations):
            self.num_gt[r] += len(gt_rel_by_type[r])

        if not pred_relations:
            return

        # 转换 pred_rel_probs 为 numpy
        P = None
        if pred_rel_probs is not None:
            if hasattr(pred_rel_probs, 'cpu'):
                P = pred_rel_probs.cpu().numpy()
            else:
                P = np.array(pred_rel_probs)

        # 向量化处理所有预测边
        # 先过滤出有效的预测（两端都被匹配的）
        valid_preds = []
        for src_pred, tgt_pred, rel_type, score in pred_relations:
            if not (0 <= rel_type < self.num_relations):
                continue
            if src_pred in inv_matching and tgt_pred in inv_matching:
                src_gt = inv_matching[src_pred]
                tgt_gt = inv_matching[tgt_pred]
                is_tp = (src_gt, tgt_gt) in gt_rel_by_type[rel_type]
                # 关键修复：gt_pair 包含 sample_id，确保跨图片唯一
                gt_pair = (sample_id, src_gt, tgt_gt) if is_tp else None
                valid_preds.append((src_pred, tgt_pred, rel_type, score, is_tp, gt_pair))
        
        if not valid_preds:
            return
        
        # 转为 numpy 数组
        src_arr = np.array([p[0] for p in valid_preds], dtype=np.int32)
        tgt_arr = np.array([p[1] for p in valid_preds], dtype=np.int32)
        rel_arr = np.array([p[2] for p in valid_preds], dtype=np.int32)
        score_arr = np.array([p[3] for p in valid_preds], dtype=np.float32)
        is_tp_arr = np.array([p[4] for p in valid_preds], dtype=bool)
        gt_pair_list = [p[5] for p in valid_preds]  # 保留为 list，因为可能是 None
        
        # 记录 raw 分数（包含 gt_pair）
        for i, (rel_type, score, is_tp, gt_pair) in enumerate(zip(rel_arr, score_arr, is_tp_arr, gt_pair_list)):
            self.pred_records_raw[rel_type].append((float(score), bool(is_tp), gt_pair))
        
        # 计算 VAS 分数（向量化）
        if use_vas and P is not None:
            violations = self._compute_vas_violations_vectorized(
                src_arr, tgt_arr, rel_arr, score_arr, P, alpha, beta, gamma,
                threshold=0.05  # 有效候选阈值，过滤低分噪声
            )
            scores_vas = score_arr * np.exp(-lam * violations)
            
            for i, (rel_type, score_vas, is_tp, gt_pair) in enumerate(zip(rel_arr, scores_vas, is_tp_arr, gt_pair_list)):
                self.pred_records_vas[rel_type].append((float(score_vas), bool(is_tp), gt_pair))
        else:
            for i, (rel_type, score, is_tp, gt_pair) in enumerate(zip(rel_arr, score_arr, is_tp_arr, gt_pair_list)):
                self.pred_records_vas[rel_type].append((float(score), bool(is_tp), gt_pair))

    def _compute_vas_violations_vectorized(self, src_arr, tgt_arr, rel_arr, score_arr, P, 
                                           alpha=0.5, beta=0.5, gamma=1.0, 
                                           threshold=0.05):
        """
        向量化计算所有边的违规风险
        
        Args:
            src_arr, tgt_arr, rel_arr: [K] 边的源、目标、关系类型
            P: [N, N, 8] dense 概率矩阵
            alpha, beta, gamma: 违规权重
            threshold: 有效候选阈值（默认0.05，过滤低分噪声）
            
        Returns:
            violations: [K] 每条边的违规风险
        """
        K = len(src_arr)
        violations = np.zeros(K, dtype=np.float32)
        
        # 关键优化：对 P 做阈值截断，只统计"有效候选"
        P_filtered = P * (P > threshold)
        
        # 将对角线置零（自环不应该计入）
        N = P.shape[0]
        for i in range(N):
            P_filtered[i, i, :] = 0.0
        
        # 预计算多父项（使用过滤后的 P）
        parent_sum = np.sum(P_filtered[:, :, PARENT], axis=0) + np.sum(P_filtered[:, :, CHILD], axis=1)
        
        # 空间关系 mask
        up_mask = rel_arr == UP
        down_mask = rel_arr == DOWN
        left_mask = rel_arr == LEFT
        right_mask = rel_arr == RIGHT
        parent_mask = rel_arr == PARENT
        child_mask = rel_arr == CHILD
        seq_mask = rel_arr == SEQUENCE
        
        # Up: v = alpha * P[src,tgt,DOWN] + beta * P[tgt,src,UP]
        if up_mask.any():
            idx = np.where(up_mask)[0]
            v = alpha * P[src_arr[idx], tgt_arr[idx], DOWN] + beta * P[tgt_arr[idx], src_arr[idx], UP]
            violations[idx] = np.minimum(1.0, v)
        
        # Down
        if down_mask.any():
            idx = np.where(down_mask)[0]
            v = alpha * P[src_arr[idx], tgt_arr[idx], UP] + beta * P[tgt_arr[idx], src_arr[idx], DOWN]
            violations[idx] = np.minimum(1.0, v)
        
        # Left
        if left_mask.any():
            idx = np.where(left_mask)[0]
            v = alpha * P[src_arr[idx], tgt_arr[idx], RIGHT] + beta * P[tgt_arr[idx], src_arr[idx], LEFT]
            violations[idx] = np.minimum(1.0, v)
        
        # Right
        if right_mask.any():
            idx = np.where(right_mask)[0]
            v = alpha * P[src_arr[idx], tgt_arr[idx], LEFT] + beta * P[tgt_arr[idx], src_arr[idx], RIGHT]
            violations[idx] = np.minimum(1.0, v)
        
        # Parent: v = gamma * max(0, parent_sum[tgt] - 1)
        # 改进：只惩罚非最高分的 Parent 边
        if parent_mask.any():
            idx = np.where(parent_mask)[0]
            
            # 对每个目标节点，找出指向它的最高分 Parent 边
            tgt_nodes = tgt_arr[idx]
            scores_parent = score_arr[idx]  # 需要传入 score_arr
            
            # 为每个目标节点记录最高分
            max_score_per_tgt = {}
            for i, (t, s) in enumerate(zip(tgt_nodes, scores_parent)):
                if t not in max_score_per_tgt or s > max_score_per_tgt[t]:
                    max_score_per_tgt[t] = s
            
            # 只惩罚非最高分的边
            for i, (t, s) in enumerate(zip(tgt_nodes, scores_parent)):
                if parent_sum[t] > 1.0:
                    # 如果是最高分的边，不惩罚或轻微惩罚
                    if s >= max_score_per_tgt[t] * 0.95:  # 允许 5% 误差
                        violations[idx[i]] = 0.0  # 不惩罚最高分的边
                    else:
                        # 惩罚非最高分的边
                        v = gamma * (parent_sum[t] - 1.0)
                        violations[idx[i]] = min(1.0, v)
        
        # Child: v = gamma * max(0, parent_sum[src] - 1)
        # 同样的改进策略
        if child_mask.any():
            idx = np.where(child_mask)[0]
            
            # Child(A, B) 表示 B 是 A 的子节点，即 A 是 B 的父
            # 所以检查的是 src 节点（A）的 parent_sum
            src_nodes = src_arr[idx]
            scores_child = score_arr[idx]
            
            # 为每个源节点记录最高分
            max_score_per_src = {}
            for i, (s_node, s) in enumerate(zip(src_nodes, scores_child)):
                if s_node not in max_score_per_src or s > max_score_per_src[s_node]:
                    max_score_per_src[s_node] = s
            
            # 只惩罚非最高分的边
            for i, (s_node, s) in enumerate(zip(src_nodes, scores_child)):
                if parent_sum[s_node] > 1.0:
                    if s >= max_score_per_src[s_node] * 0.95:
                        violations[idx[i]] = 0.0
                    else:
                        v = gamma * (parent_sum[s_node] - 1.0)
                        violations[idx[i]] = min(1.0, v)
        
        # Sequence: v = P[tgt,src,SEQUENCE]
        if seq_mask.any():
            idx = np.where(seq_mask)[0]
            violations[idx] = P[tgt_arr[idx], src_arr[idx], SEQUENCE]
        
        # Reference: v = 0 (already initialized to 0)
        
        return violations

    def add_sample(self, pred_boxes, pred_classes, pred_relations, pred_scores,
                   gt_boxes, gt_classes, gt_relations, pred_rel_probs=None,
                   use_vas=False, lam=5.0, alpha=0.5, beta=0.5, gamma=1.0):
        """
        添加一个样本进行评估
        """
        # 获取当前样本的 ID
        sample_id = self.sample_counter
        self.sample_counter += 1
        
        matching = self.instance_matching(
            pred_boxes, pred_classes, gt_boxes, gt_classes
        )

        if pred_rel_probs is not None:
            pred_relations = self._extract_relations_from_dense(pred_rel_probs)

        self.relation_evaluation(
            pred_relations, gt_relations, matching,
            pred_rel_probs=pred_rel_probs, use_vas=use_vas,
            lam=lam, alpha=alpha, beta=beta, gamma=gamma,
            sample_id=sample_id  # 传递 sample_id
        )


        # --- VW-mAP validity: compute validity scores on thresholded prediction graph ---
        # (independent of use_vas; VW-mAP is reported as a separate metric family)
        # 计算 GT 边数（用于 Edge_Recall）
        num_gt_edges = len(gt_relations)
        self._update_validity_stats(pred_relations, matching, num_gt_edges=num_gt_edges)
    
    def _extract_relations_from_dense(self, pred_rel_probs, threshold=0.0):
        """
        从 dense 概率张量中提取关系预测
        
        注意：dense 矩阵已经通过 exist_probs 二值化过滤（> 0.5 的边），
        这里只需要提取所有非零预测即可
        
        Args:
            pred_rel_probs: [N, N, num_relations] 概率张量
            threshold: 最低阈值（默认 0.0，因为已经在 exist 阶段过滤）
            
        Returns:
            pred_relations: [(src, tgt, rel_type, score), ...]
        """
        # 保持在 GPU 上处理（如果输入是 tensor）
        is_tensor = hasattr(pred_rel_probs, 'device')
        if is_tensor:
            P = pred_rel_probs
            device = P.device
        else:
            P = torch.from_numpy(pred_rel_probs)
            device = P.device
        
        N = P.shape[0]
        R = P.shape[2] if len(P.shape) > 2 else self.num_relations
        
        pred_relations = []
        
        # 对每类关系，提取所有非零预测（排除对角线）
        for rel_type in range(R):
            rel_scores = P[:, :, rel_type]  # [N, N]
            
            # 找出所有 > threshold 的预测（排除对角线）
            for i in range(N):
                for j in range(N):
                    if i != j:  # 排除自环
                        score = float(rel_scores[i, j].item())
                        if score > threshold:
                            pred_relations.append((i, j, rel_type, score))
        
        return pred_relations

    def _compute_metrics_from_records(self, pred_records):
        """从指定的 records 计算指标（GT 去重版本）"""
        metrics = {}
        
        for threshold in self.relation_thresholds:
            recalls = []
            aps = []
            
            for r in range(self.num_relations):
                num_gt = self.num_gt[r]
                rel_name = self.relation_names.get(r, f'rel_{r}')
                
                if num_gt == 0:
                    # 没有 GT，跳过但记录 AP=0
                    metrics[f'AP_{rel_name}@{threshold}'] = 0.0
                    metrics[f'R_{rel_name}@{threshold}'] = 0.0
                    continue
                
                # Recall：GT 去重（每个 GT 边最多匹配一次）
                matched_gt_pairs = set()
                for score, is_tp, gt_pair in pred_records[r]:
                    if score > threshold and is_tp and gt_pair is not None:
                        matched_gt_pairs.add(gt_pair)
                tp_count = len(matched_gt_pairs)
                recall = tp_count / num_gt
                recalls.append(recall)
                metrics[f'R_{rel_name}@{threshold}'] = float(recall)
                
                # AP: 筛选置信度 > threshold 的预测，按分数降序排序
                filtered = [(score, is_tp, gt_pair) for score, is_tp, gt_pair in pred_records[r] 
                           if score > threshold]
                if not filtered:
                    aps.append(0.0)
                    metrics[f'AP_{rel_name}@{threshold}'] = 0.0
                    continue
                
                # 按分数降序排序
                filtered.sort(key=lambda x: -x[0])
                
                # 计算 PR 曲线（GT 去重：重复命中算 FP）
                tps, fps = 0, 0
                precisions, recalls_curve = [], []
                matched_gt_in_ap = set()
                
                for score, is_tp, gt_pair in filtered:
                    if is_tp and gt_pair is not None:
                        if gt_pair not in matched_gt_in_ap:
                            # 首次命中这个 GT，算 TP
                            tps += 1
                            matched_gt_in_ap.add(gt_pair)
                        else:
                            # 重复命中，算 FP
                            fps += 1
                    else:
                        # 不是 TP，算 FP
                        fps += 1
                    
                    precisions.append(tps / (tps + fps))
                    recalls_curve.append(tps / num_gt)
                
                # 插值 AP
                precisions = np.array(precisions)
                for i in range(len(precisions) - 2, -1, -1):
                    precisions[i] = max(precisions[i], precisions[i + 1])
                
                ap = 0.0
                prev_r = 0.0
                for rec, prec in zip(recalls_curve, precisions):
                    if rec > prev_r:
                        ap += (rec - prev_r) * prec
                        prev_r = rec
                aps.append(ap)
                metrics[f'AP_{rel_name}@{threshold}'] = float(ap)
            
            mR = np.mean(recalls) if recalls else 0.0
            mAP = np.mean(aps) if aps else 0.0
            
            metrics[f'mR_g@{threshold}'] = float(mR)
            metrics[f'mAP_g@{threshold}'] = float(mAP)
        
        # 计算归一化 AP：APnorm(T_R) = mean(AP_c(T_R) / R_c(0.5))
        # 注意：per-class normalize then mean，而不是 mean(AP) / R_max(0.5)
        if 0.5 in self.relation_thresholds:
            # 第一步：计算每个类别在 T_R=0.5 时的 Recall（用于归一化）
            recall_per_class_at_05 = {}  # {r: recall_05}
            for r in range(self.num_relations):
                num_gt = self.num_gt[r]
                if num_gt == 0:
                    recall_per_class_at_05[r] = 0.0
                    continue
                matched_gt_pairs = set()
                for score, is_tp, gt_pair in pred_records[r]:
                    if score > 0.5 and is_tp and gt_pair is not None:
                        matched_gt_pairs.add(gt_pair)
                recall_05 = len(matched_gt_pairs) / num_gt
                recall_per_class_at_05[r] = recall_05
            
            # 第二步：对每个阈值计算归一化 AP
            for threshold in self.relation_thresholds:
                ap_norm_list = []  # 存储每个类别的归一化 AP
                
                for r in range(self.num_relations):
                    num_gt = self.num_gt[r]
                    if num_gt == 0:
                        # 没有 GT 的类别跳过（不参与平均）
                        continue
                    
                    rel_name = self.relation_names.get(r, f'rel_{r}')
                    ap = metrics.get(f'AP_{rel_name}@{threshold}', 0.0)
                    recall_05 = recall_per_class_at_05[r]
                    
                    # 每个类别用自己的 R@0.5 归一化
                    if recall_05 > 0:
                        ap_norm = ap / recall_05
                    else:
                        # 如果该类别在 0.5 阈值下 recall=0，说明没有任何预测命中
                        # 此时 AP 也应该是 0，归一化后仍为 0
                        ap_norm = 0.0
                    
                    ap_norm_list.append(ap_norm)
                    # 保存每个类别的归一化 AP
                    metrics[f'AP_norm_{rel_name}@{threshold}'] = float(ap_norm)
                
                # mAP_norm = mean(AP_norm_c)
                mAP_norm = np.mean(ap_norm_list) if ap_norm_list else 0.0
                metrics[f'mAP_norm@{threshold}'] = float(mAP_norm)
        
        return metrics

    def compute_metrics(self, use_vas=False):
        """计算指标"""
        records = self.pred_records_vas if use_vas else self.pred_records_raw
        return self._compute_metrics_from_records(records)
    
    def compute_all_metrics(self):
        """计算 raw 和 vas 两套指标"""
        metrics = {}
        
        raw_metrics = self._compute_metrics_from_records(self.pred_records_raw)
        for k, v in raw_metrics.items():
            metrics[k + '_raw'] = v
        
        vas_metrics = self._compute_metrics_from_records(self.pred_records_vas)
        for k, v in vas_metrics.items():
            metrics[k + '_vas'] = v
        
        return metrics

    def compute_per_class_metrics(self, threshold=0.5, use_vas=False):
        """按类别统计（GT 去重版本）"""
        records = self.pred_records_vas if use_vas else self.pred_records_raw
        per_class = {}
        
        for r in range(self.num_relations):
            num_gt = self.num_gt[r]
            
            # TP: GT 去重
            matched_gt_pairs = set()
            for score, is_tp, gt_pair in records[r]:
                if score > threshold and is_tp and gt_pair is not None:
                    matched_gt_pairs.add(gt_pair)
            tp = len(matched_gt_pairs)
            
            # FP: 所有预测中，不是 TP 的（包括重复命中）
            fp = 0
            for score, is_tp, gt_pair in records[r]:
                if score > threshold:
                    if not is_tp:
                        fp += 1
                    elif gt_pair in matched_gt_pairs:
                        # 首次命中不算 FP，但这里我们已经统计过了
                        pass
                    else:
                        # 重复命中算 FP
                        fp += 1
            
            # 重新计算 FP：总预测数 - TP
            total_preds = sum(1 for score, is_tp, gt_pair in records[r] if score > threshold)
            fp = total_preds - tp
            
            fn = num_gt - tp if num_gt > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / num_gt if num_gt > 0 else 0.0
            
            per_class[r] = {
                'name': self.relation_names.get(r, f'rel_{r}'),
                'tp': tp, 'fp': fp, 'fn': fn,
                'num_gt': num_gt,
                'recall': recall,
                'precision': precision,
            }
        
        return per_class

    def print_metrics(self, metrics):
        """打印指标"""
        print("\n" + "=" * 60)
        print("gDSA Evaluation Metrics")
        print(f"Instance Matching IoU Threshold: {self.iou_threshold}")
        print("=" * 60)
        
        for threshold in self.relation_thresholds:
            print(f"\n📊 Threshold T_R = {threshold}")
            print(f"   mR_g:       {metrics.get(f'mR_g@{threshold}', 0.0):.4f}")
            print(f"   mAP_g:      {metrics.get(f'mAP_g@{threshold}', 0.0):.4f}")
            if f'mAP_norm@{threshold}' in metrics:
                print(f"   mAP_norm:   {metrics.get(f'mAP_norm@{threshold}', 0.0):.4f}")
        
        print("=" * 60)


    # --------------------------------------------------------------------------
    # Validity-Weighted mAP (VW-mAP)
    # --------------------------------------------------------------------------
    def _collect_threshold_edges(self, pred_relations, matching, thr):
        """Collect thresholded edges whose endpoints are both matched.

        Returns:
            nodes: list of matched predicted node indices
            parents: list of directed parent edges (u -> v)
            spat: dict rel_name -> set of (u, v)
            seq: set of (u, v)
        """
        inv_matching = {v: k for k, v in matching.items()}
        nodes = sorted(inv_matching.keys())
        node_set = set(nodes)

        parents = []
        spat = {'Up': set(), 'Down': set(), 'Left': set(), 'Right': set()}
        seq = set()

        if not pred_relations:
            return nodes, parents, spat, seq

        for src, tgt, rel_type, score in pred_relations:
            if score <= thr:
                continue
            if src not in node_set or tgt not in node_set:
                continue

            if rel_type == PARENT:
                parents.append((src, tgt))
            elif rel_type == CHILD:
                # Child(src, tgt) 也表示 src -> tgt (src是tgt的父节点)，不需要反转
                parents.append((src, tgt))
            elif rel_type == UP:
                spat['Up'].add((src, tgt))
            elif rel_type == DOWN:
                spat['Down'].add((src, tgt))
            elif rel_type == LEFT:
                spat['Left'].add((src, tgt))
            elif rel_type == RIGHT:
                spat['Right'].add((src, tgt))
            elif rel_type == SEQUENCE:
                seq.add((src, tgt))
            else:
                # Reference: no structural constraint for validity
                pass

        return nodes, parents, spat, seq

    def _count_cycle_nodes_directed(self, nodes, edges):
        """Return number of nodes that belong to at least one directed cycle."""
        if not nodes or not edges:
            return 0
        idx = {n: i for i, n in enumerate(nodes)}
        adj = [[] for _ in nodes]
        for u, v in edges:
            if u in idx and v in idx:
                adj[idx[u]].append(idx[v])

        color = [0] * len(nodes)  # 0=unvisited,1=visiting,2=done
        in_cycle = [False] * len(nodes)
        stack = []
        pos_in_stack = {}

        def dfs(u):
            color[u] = 1
            pos_in_stack[u] = len(stack)
            stack.append(u)
            for v in adj[u]:
                if color[v] == 0:
                    dfs(v)
                elif color[v] == 1:
                    # back-edge => cycle nodes are stack[pos[v]:]
                    start = pos_in_stack.get(v, None)
                    if start is not None:
                        for w in stack[start:]:
                            in_cycle[w] = True
            stack.pop()
            pos_in_stack.pop(u, None)
            color[u] = 2

        for u in range(len(nodes)):
            if color[u] == 0:
                dfs(u)

        return int(sum(in_cycle))

    def _compute_validity(self, nodes, parents, spat, seq, wt=0.5, ws=0.3, wq=0.2):
        """Compute V@T_R and its components on the thresholded graph."""
        N = len(nodes)
        if N == 0:
            return 0.0, 0.0, 0.0, 0.0

        # --- Tree validity (Parent/Child) ---
        indeg = {n: 0 for n in nodes}
        for u, v in parents:
            if v in indeg:
                indeg[v] += 1
        V_sp = sum(1 for n in nodes if indeg[n] <= 1) / N

        cycle_nodes = self._count_cycle_nodes_directed(nodes, parents)
        V_acyc = 1.0 - (cycle_nodes / N)
        V_tree = float(V_sp * V_acyc)

        # --- Spatial validity (Up/Down/Left/Right) ---
        # Conflicts are measured on the delivered edge set (no projection).
        total_spatial = sum(len(s) for s in spat.values())
        if total_spatial == 0:
            V_spatial = 1.0
        else:
            conflict_edges = 0
            # mutual exclusivity: Up vs Down, Left vs Right on same ordered pair
            conflict_edges += 2 * len(spat['Up'].intersection(spat['Down']))
            conflict_edges += 2 * len(spat['Left'].intersection(spat['Right']))
            # antisymmetry: r(i,j) and r(j,i)
            for key in ['Up', 'Down', 'Left', 'Right']:
                rev = {(b, a) for (a, b) in spat[key]}
                conflict_edges += 2 * len(spat[key].intersection(rev))
            r_s = min(1.0, conflict_edges / max(1, total_spatial))
            V_spatial = float(1.0 - r_s)

        # --- Sequence validity (2-cycle) ---
        total_seq = len(seq)
        if total_seq == 0:
            V_seq = 1.0
        else:
            rev = {(b, a) for (a, b) in seq}
            conflict_edges = 2 * len(seq.intersection(rev))
            r_2c = min(1.0, conflict_edges / max(1, total_seq))
            V_seq = float(1.0 - r_2c)

        V = float(wt * V_tree + ws * V_spatial + wq * V_seq)
        return V_tree, V_spatial, V_seq, V

    def _update_validity_stats(self, pred_relations, matching, num_gt_edges=0, wt=0.5, ws=0.3, wq=0.2, gamma=1.0):
        """Update per-threshold validity statistics for VW-mAP.
        
        方案 A: VW-mAP = mAP × V_eff
        - V: 纯结构合法性（多父/空间冲突/2-cycle）
        - C_E = min(1, |E_T| / (|G| + ε)): 结构覆盖度（不是检索召回）
        - V_eff = V × C_E^γ: 有效合法性（惩罚边太少）
        
        Args:
            gamma: C_E 的指数，默认 1.0，若担心早期惩罚太强可取 0.5
        """
        eps = 1e-6
        for thr in self.relation_thresholds:
            nodes, parents, spat, seq = self._collect_threshold_edges(pred_relations, matching, float(thr))
            V_tree, V_spatial, V_seq, V = self._compute_validity(nodes, parents, spat, seq, wt=wt, ws=ws, wq=wq)
            
            # 计算 |E_T|：交付图的预测边数（在 matched endpoints 上，且 score > T_R）
            num_pred_edges = len(parents) + sum(len(s) for s in spat.values()) + len(seq)
            
            # C_E@T_R = min(1, |E_T| / (|G| + ε))
            # |G| = num_gt_edges（同一 matched endpoints 上的 GT 边数）
            C_E = min(1.0, num_pred_edges / (num_gt_edges + eps)) if num_gt_edges > 0 else (1.0 if num_pred_edges == 0 else 0.0)
            
            # V_eff = V × C_E^γ
            V_eff = V * (C_E ** gamma)
            
            self.validity_stats[float(thr)]['V_tree'].append(V_tree)
            self.validity_stats[float(thr)]['V_spatial'].append(V_spatial)
            self.validity_stats[float(thr)]['V_seq'].append(V_seq)
            self.validity_stats[float(thr)]['V'].append(V)
            self.validity_stats[float(thr)]['C_E'].append(C_E)
            self.validity_stats[float(thr)]['V_eff'].append(V_eff)

    def compute_validity_metrics(self):
        """Compute mean validity metrics aggregated over samples."""
        out = {}
        for thr in self.relation_thresholds:
            thr = float(thr)
            stats = self.validity_stats.get(thr, None)
            if not stats or len(stats['V']) == 0:
                out[f'V@{thr}'] = 0.0
                out[f'V_tree@{thr}'] = 0.0
                out[f'V_spatial@{thr}'] = 0.0
                out[f'V_seq@{thr}'] = 0.0
                out[f'C_E@{thr}'] = 0.0
                out[f'V_eff@{thr}'] = 0.0
                continue
            out[f'V@{thr}'] = float(sum(stats['V']) / len(stats['V']))
            out[f'V_tree@{thr}'] = float(sum(stats['V_tree']) / len(stats['V_tree']))
            out[f'V_spatial@{thr}'] = float(sum(stats['V_spatial']) / len(stats['V_spatial']))
            out[f'V_seq@{thr}'] = float(sum(stats['V_seq']) / len(stats['V_seq']))
            out[f'C_E@{thr}'] = float(sum(stats['C_E']) / len(stats['C_E']))
            out[f'V_eff@{thr}'] = float(sum(stats['V_eff']) / len(stats['V_eff']))
        return out

    def compute_vw_metrics(self, mode='mul', eps=1e-12, use_vas_for_map=False):
        """Compute VW-mAP_g@T_R (方案 A).

        公式：VW-mAP_g@T_R = mAP_g@T_R × V_eff@T_R
        其中：V_eff = V × C_E^γ
              C_E = min(1, |E_T| / |G|)，结构覆盖度
              V = 纯结构合法性

        Args:
            mode: 'mul' (推荐，严格不超过 raw) 或 'harmonic'
            eps: numerical stabilizer
            use_vas_for_map: if True, use VAS-scored AP as the AP term

        Returns:
            dict with keys:
              - VW-mAP_g@{thr}: 最终指标
              - mAP_g@{thr}: 原始 mAP
              - V@{thr}: 纯结构合法性
              - C_E@{thr}: 结构覆盖度
              - V_eff@{thr}: 有效合法性 = V × C_E^γ
        """
        ap_metrics = self.compute_metrics(use_vas=bool(use_vas_for_map))
        v_metrics = self.compute_validity_metrics()
        out = {}
        for thr in self.relation_thresholds:
            thr = float(thr)
            mAP = float(ap_metrics.get(f'mAP_g@{thr}', 0.0))
            V = float(v_metrics.get(f'V@{thr}', 0.0))
            C_E = float(v_metrics.get(f'C_E@{thr}', 0.0))
            V_eff = float(v_metrics.get(f'V_eff@{thr}', 0.0))
            
            # VW-mAP = mAP × V_eff（严格不超过 mAP）
            if mode == 'mul':
                vw = mAP * V_eff
            else:  # harmonic（保留兼容）
                vw = (2.0 * mAP * V_eff) / (mAP + V_eff + eps)
            
            out[f'mAP_g@{thr}'] = mAP
            out[f'V@{thr}'] = V
            out[f'C_E@{thr}'] = C_E
            out[f'V_eff@{thr}'] = V_eff
            out[f'V_tree@{thr}'] = float(v_metrics.get(f'V_tree@{thr}', 0.0))
            out[f'V_spatial@{thr}'] = float(v_metrics.get(f'V_spatial@{thr}', 0.0))
            out[f'V_seq@{thr}'] = float(v_metrics.get(f'V_seq@{thr}', 0.0))
            out[f'VW-mAP_g@{thr}'] = float(vw)
        return out



class gDSAEvaluatorE2E(gDSAEvaluator):
    """
    端到端 gDSA 评估器
    
    与基础版本的区别：
    - 支持从检测结果直接评估（不需要预先匹配的实例）
    - 支持多 IoU 阈值评估 (0.5, 0.75, 0.5:0.95)
    - 同时计算 raw 和 VAS 指标
    """
    
    def __init__(self, num_relations=8, iou_thresholds=None, relation_thresholds=None,
                 topk_per_relation=2000):
        """
        Args:
            num_relations: 关系类别数
            iou_thresholds: IoU 阈值列表，默认 [0.5, 0.75, 0.5:0.95]
            relation_thresholds: 关系置信度阈值列表
            topk_per_relation: 从 dense 提取时每类关系保留的 TopK 数量
        """
        super().__init__(num_relations=num_relations, 
                        relation_thresholds=relation_thresholds or [0.5, 0.75],
                        topk_per_relation=topk_per_relation)
        
        # 多 IoU 阈值
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
        else:
            self.iou_thresholds = iou_thresholds
        
        self.reset()
    
    def reset(self):
        """重置统计信息（支持多 IoU 阈值）"""
        # 每个 IoU 阈值一套 records
        self.pred_records_raw_by_iou = {
            iou: {r: [] for r in range(self.num_relations)} 
            for iou in self.iou_thresholds
        }
        self.pred_records_vas_by_iou = {
            iou: {r: [] for r in range(self.num_relations)} 
            for iou in self.iou_thresholds
        }
        self.num_gt_by_iou = {
            iou: {r: 0 for r in range(self.num_relations)} 
            for iou in self.iou_thresholds
        }
        
        # 样本计数器（用于生成唯一的 sample_id）
        self.sample_counter = 0
        
        # 兼容基类
        self.pred_records_raw = self.pred_records_raw_by_iou.get(0.5, {r: [] for r in range(self.num_relations)})
        self.pred_records_vas = self.pred_records_vas_by_iou.get(0.5, {r: [] for r in range(self.num_relations)})
        self.num_gt = self.num_gt_by_iou.get(0.5, {r: 0 for r in range(self.num_relations)})
    
    def add_sample(self, pred_boxes, pred_classes, pred_relations, pred_scores,
                   gt_boxes, gt_classes, gt_relations, pred_rel_probs=None,
                   use_vas=True, lam=5.0, alpha=0.5, beta=0.5, gamma=1.0):
        """
        添加一个样本进行评估（支持多 IoU 阈值）
        """
        # 获取当前样本的 ID
        sample_id = self.sample_counter
        self.sample_counter += 1
        
        # 如果提供了 dense 概率，提取关系
        if pred_rel_probs is not None:
            pred_relations = self._extract_relations_from_dense(pred_rel_probs)
        
        # 转换 pred_rel_probs 为 numpy
        P = None
        if pred_rel_probs is not None:
            if hasattr(pred_rel_probs, 'cpu'):
                P = pred_rel_probs.cpu().numpy()
            else:
                P = np.array(pred_rel_probs)
        
        # 对每个 IoU 阈值分别评估
        for iou_thresh in self.iou_thresholds:
            self.iou_threshold = iou_thresh
            matching = self.instance_matching(
                pred_boxes, pred_classes, gt_boxes, gt_classes
            )
            
            self._relation_evaluation_for_iou(
                pred_relations, gt_relations, matching, iou_thresh,
                P=P, use_vas=use_vas, lam=lam, alpha=alpha, beta=beta, gamma=gamma,
                sample_id=sample_id  # 传递 sample_id
            )
    
    def _relation_evaluation_for_iou(self, pred_relations, gt_relations, matching, iou_thresh,
                                     P=None, use_vas=True, lam=5.0, alpha=0.5, beta=0.5, gamma=1.0, sample_id=None):
        """针对特定 IoU 阈值的关系评估（完全向量化版本，GT 去重）
        
        Args:
            sample_id: 样本 ID（用于跨图片去重 GT）
        """
        inv_matching = {v: k for k, v in matching.items()}
        
        # 构建 GT 关系集合（全量，不过滤 matching）
        gt_rel_by_type = {r: set() for r in range(self.num_relations)}
        for src_gt, tgt_gt, rel_type in gt_relations:
            if 0 <= rel_type < self.num_relations:
                gt_rel_by_type[rel_type].add((src_gt, tgt_gt))
        
        # 更新 GT 数量
        for r in range(self.num_relations):
            self.num_gt_by_iou[iou_thresh][r] += len(gt_rel_by_type[r])
        
        if not pred_relations:
            return
        
        # 向量化处理所有预测边
        valid_preds = []
        for src_pred, tgt_pred, rel_type, score in pred_relations:
            if not (0 <= rel_type < self.num_relations):
                continue
            if src_pred in inv_matching and tgt_pred in inv_matching:
                src_gt = inv_matching[src_pred]
                tgt_gt = inv_matching[tgt_pred]
                is_tp = (src_gt, tgt_gt) in gt_rel_by_type[rel_type]
                # 关键修复：gt_pair 包含 sample_id，确保跨图片唯一
                gt_pair = (sample_id, src_gt, tgt_gt) if is_tp else None
                valid_preds.append((src_pred, tgt_pred, rel_type, score, is_tp, gt_pair))
        
        if not valid_preds:
            return
        
        # 转为 numpy 数组
        src_arr = np.array([p[0] for p in valid_preds], dtype=np.int32)
        tgt_arr = np.array([p[1] for p in valid_preds], dtype=np.int32)
        rel_arr = np.array([p[2] for p in valid_preds], dtype=np.int32)
        score_arr = np.array([p[3] for p in valid_preds], dtype=np.float32)
        is_tp_arr = np.array([p[4] for p in valid_preds], dtype=bool)
        gt_pair_list = [p[5] for p in valid_preds]
        
        # 记录 raw 分数（包含 gt_pair）
        for rel_type, score, is_tp, gt_pair in zip(rel_arr, score_arr, is_tp_arr, gt_pair_list):
            self.pred_records_raw_by_iou[iou_thresh][rel_type].append((float(score), bool(is_tp), gt_pair))
        
        # 计算 VAS 分数（向量化）
        if use_vas and P is not None:
            violations = self._compute_vas_violations_vectorized(
                src_arr, tgt_arr, rel_arr, score_arr, P, alpha, beta, gamma,
                threshold=0.05  # 有效候选阈值，过滤低分噪声
            )
            scores_vas = score_arr * np.exp(-lam * violations)
            
            for rel_type, score_vas, is_tp, gt_pair in zip(rel_arr, scores_vas, is_tp_arr, gt_pair_list):
                self.pred_records_vas_by_iou[iou_thresh][rel_type].append((float(score_vas), bool(is_tp), gt_pair))
        else:
            for rel_type, score, is_tp, gt_pair in zip(rel_arr, score_arr, is_tp_arr, gt_pair_list):
                self.pred_records_vas_by_iou[iou_thresh][rel_type].append((float(score), bool(is_tp), gt_pair))
    
    def _compute_metrics_for_iou(self, iou_thresh, pred_records_by_iou):
        """计算特定 IoU 阈值的指标（GT 去重版本）"""
        metrics = {}
        pred_records = pred_records_by_iou[iou_thresh]
        num_gt = self.num_gt_by_iou[iou_thresh]
        
        for threshold in self.relation_thresholds:
            recalls = []
            aps = []
            
            for r in range(self.num_relations):
                n_gt = num_gt[r]
                if n_gt == 0:
                    continue
                
                # Recall: GT 去重
                matched_gt_pairs = set()
                for score, is_tp, gt_pair in pred_records[r]:
                    if score > threshold and is_tp and gt_pair is not None:
                        matched_gt_pairs.add(gt_pair)
                tp_count = len(matched_gt_pairs)
                recalls.append(tp_count / n_gt)
                
                # AP: GT 去重
                filtered = [(score, is_tp, gt_pair) for score, is_tp, gt_pair in pred_records[r] 
                           if score > threshold]
                if not filtered:
                    aps.append(0.0)
                    continue
                
                filtered.sort(key=lambda x: -x[0])
                
                tps, fps = 0, 0
                precisions, recalls_curve = [], []
                matched_gt_in_ap = set()
                
                for score, is_tp, gt_pair in filtered:
                    if is_tp and gt_pair is not None:
                        if gt_pair not in matched_gt_in_ap:
                            tps += 1
                            matched_gt_in_ap.add(gt_pair)
                        else:
                            fps += 1
                    else:
                        fps += 1
                    
                    precisions.append(tps / (tps + fps))
                    recalls_curve.append(tps / n_gt)
                
                # 插值 AP
                precisions = np.array(precisions)
                for i in range(len(precisions) - 2, -1, -1):
                    precisions[i] = max(precisions[i], precisions[i + 1])
                
                ap = 0.0
                prev_r = 0.0
                for rec, prec in zip(recalls_curve, precisions):
                    if rec > prev_r:
                        ap += (rec - prev_r) * prec
                        prev_r = rec
                aps.append(ap)
            
            mR = np.mean(recalls) if recalls else 0.0
            mAP = np.mean(aps) if aps else 0.0
            
            metrics[f'mR_g@{threshold}'] = float(mR)
            metrics[f'mAP_g@{threshold}'] = float(mAP)
        
        return metrics
    
    def compute_metrics(self, use_vas=False):
        """计算指标（兼容基类接口）"""
        records = self.pred_records_vas_by_iou if use_vas else self.pred_records_raw_by_iou
        
        # 默认返回 IoU=0.5 的指标
        if 0.5 in self.iou_thresholds:
            return self._compute_metrics_for_iou(0.5, records)
        else:
            return self._compute_metrics_for_iou(self.iou_thresholds[0], records)
    
    def compute_all_metrics(self):
        """
        计算完整指标集：
        - mAP_g@0.5, mAP_g@0.75, mAP_g@0.5:0.95
        - raw 和 vas 两套
        """
        metrics = {}
        
        # IoU=0.5
        if 0.5 in self.iou_thresholds:
            raw_05 = self._compute_metrics_for_iou(0.5, self.pred_records_raw_by_iou)
            vas_05 = self._compute_metrics_for_iou(0.5, self.pred_records_vas_by_iou)
            for k, v in raw_05.items():
                metrics[k + '_iou50_raw'] = v
            for k, v in vas_05.items():
                metrics[k + '_iou50_vas'] = v
        
        # IoU=0.75
        if 0.75 in self.iou_thresholds:
            raw_75 = self._compute_metrics_for_iou(0.75, self.pred_records_raw_by_iou)
            vas_75 = self._compute_metrics_for_iou(0.75, self.pred_records_vas_by_iou)
            for k, v in raw_75.items():
                metrics[k + '_iou75_raw'] = v
            for k, v in vas_75.items():
                metrics[k + '_iou75_vas'] = v
        
        # IoU=0.5:0.95 (COCO style)
        coco_iou_thresholds = [t for t in self.iou_thresholds if 0.5 <= t < 1.0]
        if len(coco_iou_thresholds) >= 2:
            # 对每个 IoU 阈值计算指标，然后平均
            raw_metrics_list = []
            vas_metrics_list = []
            
            for iou_t in coco_iou_thresholds:
                raw_m = self._compute_metrics_for_iou(iou_t, self.pred_records_raw_by_iou)
                vas_m = self._compute_metrics_for_iou(iou_t, self.pred_records_vas_by_iou)
                raw_metrics_list.append(raw_m)
                vas_metrics_list.append(vas_m)
            
            # 平均
            for threshold in self.relation_thresholds:
                raw_mAPs = [m.get(f'mAP_g@{threshold}', 0.0) for m in raw_metrics_list]
                vas_mAPs = [m.get(f'mAP_g@{threshold}', 0.0) for m in vas_metrics_list]
                raw_mRs = [m.get(f'mR_g@{threshold}', 0.0) for m in raw_metrics_list]
                vas_mRs = [m.get(f'mR_g@{threshold}', 0.0) for m in vas_metrics_list]
                
                metrics[f'mAP_g@{threshold}_iou50-95_raw'] = float(np.mean(raw_mAPs))
                metrics[f'mAP_g@{threshold}_iou50-95_vas'] = float(np.mean(vas_mAPs))
                metrics[f'mR_g@{threshold}_iou50-95_raw'] = float(np.mean(raw_mRs))
                metrics[f'mR_g@{threshold}_iou50-95_vas'] = float(np.mean(vas_mRs))
        
        return metrics
    
    def compute_standard_metrics(self, use_vas=False):
        """
        计算标准指标（简化版）：
        - mAP_g@0.5 (IoU=0.5)
        - mAP_g@0.75 (IoU=0.75)  
        - mAP_g@0.5:0.95 (COCO style)
        """
        records = self.pred_records_vas_by_iou if use_vas else self.pred_records_raw_by_iou
        metrics = {}
        
        # 使用默认关系阈值 0.5
        rel_thresh = 0.5
        
        # IoU=0.5
        if 0.5 in self.iou_thresholds:
            m = self._compute_metrics_for_iou(0.5, records)
            metrics['mAP_g@0.5'] = m.get(f'mAP_g@{rel_thresh}', 0.0)
            metrics['mR_g@0.5'] = m.get(f'mR_g@{rel_thresh}', 0.0)
        
        # IoU=0.75
        if 0.75 in self.iou_thresholds:
            m = self._compute_metrics_for_iou(0.75, records)
            metrics['mAP_g@0.75'] = m.get(f'mAP_g@{rel_thresh}', 0.0)
            metrics['mR_g@0.75'] = m.get(f'mR_g@{rel_thresh}', 0.0)
        
        # IoU=0.5:0.95
        coco_iou_thresholds = [t for t in self.iou_thresholds if 0.5 <= t < 1.0]
        if len(coco_iou_thresholds) >= 2:
            mAPs = []
            mRs = []
            for iou_t in coco_iou_thresholds:
                m = self._compute_metrics_for_iou(iou_t, records)
                mAPs.append(m.get(f'mAP_g@{rel_thresh}', 0.0))
                mRs.append(m.get(f'mR_g@{rel_thresh}', 0.0))
            metrics['mAP_g@0.5:0.95'] = float(np.mean(mAPs))
            metrics['mR_g@0.5:0.95'] = float(np.mean(mRs))
        
        return metrics
    
    def print_metrics(self, metrics=None):
        """打印指标"""
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        print("\n" + "=" * 70)
        print("gDSA E2E Evaluation Metrics")
        print("=" * 70)
        
        # Raw 指标
        print("\n📊 Raw Predictions (无后处理):")
        for iou_name in ['iou50', 'iou75', 'iou50-95']:
            mAP_key = f'mAP_g@0.5_{iou_name}_raw'
            mR_key = f'mR_g@0.5_{iou_name}_raw'
            if mAP_key in metrics:
                print(f"   {iou_name}: mAP_g={metrics[mAP_key]:.4f}, mR_g={metrics[mR_key]:.4f}")
        
        # VAS 指标
        print("\n📊 VAS Predictions (违规感知评分):")
        for iou_name in ['iou50', 'iou75', 'iou50-95']:
            mAP_key = f'mAP_g@0.5_{iou_name}_vas'
            mR_key = f'mR_g@0.5_{iou_name}_vas'
            if mAP_key in metrics:
                print(f"   {iou_name}: mAP_g={metrics[mAP_key]:.4f}, mR_g={metrics[mR_key]:.4f}")
        
        print("=" * 70)
