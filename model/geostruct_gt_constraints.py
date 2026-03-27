# geostruct_gt_constraints.py
"""
gDSA 图约束后处理模块 (v2 - 加入 MST 算法)

将模型的原始边预测转换为合法的文档关系图：
1. 空间关系 (Up/Down/Left/Right): 反对称约束 + mutual exclusivity
2. Parent/Child: Chu-Liu/Edmonds 最大生成树 (MST)
3. Sequence: DAG 约束（贪心 + 环检测）

使用方法:
    from geostruct_gt_constraints import GraphConstraintSolver
    solver = GraphConstraintSolver()
    legal_relations = solver.solve(pred_relations, num_nodes)
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ============ Chu-Liu/Edmonds 最大生成树算法 ============

class ChuLiuEdmonds:
    """
    Chu-Liu/Edmonds 算法 - 有向图最大生成树
    
    用于 Parent/Child 关系：找到权重最大的树结构，
    保证每个节点最多有一个 Parent。
    """
    
    @staticmethod
    def max_spanning_arborescence(nodes: List[int], 
                                   edges: List[Tuple[int, int, float]],
                                   root: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """
        找到有向图的最大生成树（arborescence）
        
        Args:
            nodes: 节点列表
            edges: 边列表 [(src, tgt, weight), ...]
            root: 根节点（如果为 None，则尝试所有可能的根）
            
        Returns:
            最大生成树的边列表 [(src, tgt, weight), ...]
        """
        if len(nodes) <= 1 or not edges:
            return []
        
        # 如果没有指定根，尝试找最佳根
        if root is None:
            best_tree = []
            best_weight = -float('inf')
            
            for candidate_root in nodes:
                tree = ChuLiuEdmonds._edmonds(nodes, edges, candidate_root)
                weight = sum(w for _, _, w in tree)
                if weight > best_weight:
                    best_weight = weight
                    best_tree = tree
            
            return best_tree
        else:
            return ChuLiuEdmonds._edmonds(nodes, edges, root)
    
    @staticmethod
    def _edmonds(nodes: List[int], 
                 edges: List[Tuple[int, int, float]], 
                 root: int) -> List[Tuple[int, int, float]]:
        """Edmonds 算法核心实现"""
        
        # 构建邻接表：tgt -> [(src, weight), ...]
        incoming = defaultdict(list)
        for src, tgt, weight in edges:
            if tgt != root:  # 根节点不能有入边
                incoming[tgt].append((src, weight))
        
        # Step 1: 对每个非根节点，选择权重最大的入边
        max_in_edge = {}  # tgt -> (src, weight)
        for tgt in nodes:
            if tgt == root:
                continue
            if incoming[tgt]:
                best_src, best_weight = max(incoming[tgt], key=lambda x: x[1])
                max_in_edge[tgt] = (best_src, best_weight)
        
        # 检查是否所有非根节点都有入边
        non_root_nodes = [n for n in nodes if n != root]
        if len(max_in_edge) < len(non_root_nodes):
            # 有些节点无法到达，返回能构建的部分
            return [(src, tgt, w) for tgt, (src, w) in max_in_edge.items()]
        
        # Step 2: 检测环
        cycle = ChuLiuEdmonds._find_cycle(max_in_edge, root)
        
        if cycle is None:
            # 无环，直接返回
            return [(src, tgt, w) for tgt, (src, w) in max_in_edge.items()]
        
        # Step 3: 收缩环
        cycle_set = set(cycle)
        new_node = max(nodes) + 1  # 新的超级节点
        
        # 构建收缩后的图
        new_nodes = [n for n in nodes if n not in cycle_set] + [new_node]
        new_edges = []
        
        # 记录边的来源，用于恢复
        edge_origin = {}  # (new_src, new_tgt) -> (orig_src, orig_tgt, orig_weight)
        
        for src, tgt, weight in edges:
            if src in cycle_set and tgt in cycle_set:
                continue  # 环内部的边忽略
            
            if tgt in cycle_set:
                # 进入环的边：调整权重
                # 新权重 = 原权重 - 环中该节点的最大入边权重
                _, cycle_weight = max_in_edge[tgt]
                new_weight = weight - cycle_weight
                new_edges.append((src, new_node, new_weight))
                edge_origin[(src, new_node)] = (src, tgt, weight)
            elif src in cycle_set:
                # 离开环的边
                new_edges.append((new_node, tgt, weight))
                edge_origin[(new_node, tgt)] = (src, tgt, weight)
            else:
                # 与环无关的边
                new_edges.append((src, tgt, weight))
        
        # 递归求解
        sub_tree = ChuLiuEdmonds._edmonds(new_nodes, new_edges, root)
        
        # Step 4: 展开环
        result = []
        entering_node = None  # 进入环的节点
        
        for src, tgt, weight in sub_tree:
            if tgt == new_node:
                # 找到进入环的边
                orig_src, orig_tgt, orig_weight = edge_origin.get((src, new_node), (src, tgt, weight))
                result.append((orig_src, orig_tgt, orig_weight))
                entering_node = orig_tgt
            elif src == new_node:
                # 离开环的边
                orig_src, orig_tgt, orig_weight = edge_origin.get((new_node, tgt), (src, tgt, weight))
                result.append((orig_src, orig_tgt, orig_weight))
            else:
                result.append((src, tgt, weight))
        
        # 添加环中的边（除了被替换的那条）
        for node in cycle:
            if node != entering_node and node in max_in_edge:
                src, weight = max_in_edge[node]
                result.append((src, node, weight))
        
        return result
    
    @staticmethod
    def _find_cycle(max_in_edge: Dict[int, Tuple[int, float]], 
                    root: int) -> Optional[List[int]]:
        """检测环，返回环中的节点列表"""
        visited = set()
        path = []
        path_set = set()
        
        def dfs(node):
            if node == root:
                return None
            if node in path_set:
                # 找到环
                cycle_start = path.index(node)
                return path[cycle_start:]
            if node in visited:
                return None
            
            visited.add(node)
            path.append(node)
            path_set.add(node)
            
            if node in max_in_edge:
                parent, _ = max_in_edge[node]
                result = dfs(parent)
                if result is not None:
                    return result
            
            path.pop()
            path_set.remove(node)
            return None
        
        for node in max_in_edge:
            if node not in visited:
                cycle = dfs(node)
                if cycle is not None:
                    return cycle
        
        return None


# ============ 图约束求解器 ============

class GraphConstraintSolver:
    """
    图约束求解器 - 将原始预测转换为合法文档关系图
    
    v2 改进：
    - Parent/Child 使用 Chu-Liu/Edmonds MST 算法（全局最优）
    - 空间关系增加 mutual exclusivity
    """
    
    # 关系类型定义
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    PARENT, CHILD, SEQUENCE, REFERENCE = 4, 5, 6, 7
    
    def __init__(self, 
                 apply_spatial_constraints=True,
                 apply_tree_constraints=True,
                 apply_dag_constraints=True,
                 use_mst=True,  # 是否使用 MST 算法
                 score_threshold=0.3):
        """
        Args:
            apply_spatial_constraints: 是否应用空间关系约束
            apply_tree_constraints: 是否应用 Parent/Child 树约束
            apply_dag_constraints: 是否应用 Sequence DAG 约束
            use_mst: 是否使用 MST 算法（否则用贪心）
            score_threshold: 边的分数阈值
        """
        self.apply_spatial = apply_spatial_constraints
        self.apply_tree = apply_tree_constraints
        self.apply_dag = apply_dag_constraints
        self.use_mst = use_mst
        self.score_threshold = score_threshold
    
    def solve(self, pred_relations, num_nodes):
        """
        将原始预测转换为合法图
        
        Args:
            pred_relations: list of (src, tgt, rel_type, score)
            num_nodes: 节点数量
            
        Returns:
            legal_relations: list of (src, tgt, rel_type, score)
        """
        if not pred_relations or num_nodes < 2:
            return []
        
        # 按关系类型分组（不做阈值过滤，让约束算法处理所有边）
        relations_by_type = defaultdict(list)
        for src, tgt, rel_type, score in pred_relations:
            # 只有当 score_threshold > 0 时才过滤
            if self.score_threshold <= 0 or score >= self.score_threshold:
                relations_by_type[rel_type].append((src, tgt, score))
        
        legal_relations = []
        
        # 1. 空间关系 (Up/Down/Left/Right)
        if self.apply_spatial:
            spatial_legal = self._process_spatial_relations(relations_by_type, num_nodes)
            legal_relations.extend(spatial_legal)
        else:
            for rel_type in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
                for src, tgt, score in relations_by_type.get(rel_type, []):
                    legal_relations.append((src, tgt, rel_type, score))
        
        # 2. Parent/Child 关系 - 使用 MST
        if self.apply_tree:
            tree_legal = self._process_tree_relations(relations_by_type, num_nodes)
            legal_relations.extend(tree_legal)
        else:
            for rel_type in [self.PARENT, self.CHILD]:
                for src, tgt, score in relations_by_type.get(rel_type, []):
                    legal_relations.append((src, tgt, rel_type, score))
        
        # 3. Sequence 关系 - DAG 约束
        if self.apply_dag:
            seq_edges = relations_by_type.get(self.SEQUENCE, [])
            legal_seq = self._apply_dag_constraint(seq_edges, num_nodes)
            for src, tgt, score in legal_seq:
                legal_relations.append((src, tgt, self.SEQUENCE, score))
        else:
            for src, tgt, score in relations_by_type.get(self.SEQUENCE, []):
                legal_relations.append((src, tgt, self.SEQUENCE, score))
        
        # 4. Reference 关系 - 无特殊约束
        for src, tgt, score in relations_by_type.get(self.REFERENCE, []):
            legal_relations.append((src, tgt, self.REFERENCE, score))
        
        return legal_relations
    
    def _process_spatial_relations(self, relations_by_type, num_nodes):
        """处理空间关系：反对称 + mutual exclusivity"""
        legal = []
        
        # Up/Down 互斥
        up_edges = relations_by_type.get(self.UP, [])
        down_edges = relations_by_type.get(self.DOWN, [])
        legal_up, legal_down = self._apply_mutual_exclusivity(
            up_edges, down_edges, num_nodes
        )
        for src, tgt, score in legal_up:
            legal.append((src, tgt, self.UP, score))
        for src, tgt, score in legal_down:
            legal.append((src, tgt, self.DOWN, score))
        
        # Left/Right 互斥
        left_edges = relations_by_type.get(self.LEFT, [])
        right_edges = relations_by_type.get(self.RIGHT, [])
        legal_left, legal_right = self._apply_mutual_exclusivity(
            left_edges, right_edges, num_nodes
        )
        for src, tgt, score in legal_left:
            legal.append((src, tgt, self.LEFT, score))
        for src, tgt, score in legal_right:
            legal.append((src, tgt, self.RIGHT, score))
        
        return legal
    
    def _apply_mutual_exclusivity(self, edges_a, edges_b, num_nodes):
        """
        Mutual Exclusivity 约束（优化版）
        
        对于 (i, j) 对：
        - A(i,j) 和 B(i,j) 互斥（不能同时 Up 和 Down）
        - A(i,j) 和 A(j,i) 反对称
        """
        # 使用字典代替矩阵，只存储非零值
        score_a = {}  # (src, tgt) -> score
        score_b = {}
        
        for src, tgt, score in edges_a:
            key = (src, tgt)
            if key not in score_a or score > score_a[key]:
                score_a[key] = score
        
        for src, tgt, score in edges_b:
            key = (src, tgt)
            if key not in score_b or score > score_b[key]:
                score_b[key] = score
        
        # 收集所有涉及的节点对
        all_pairs = set()
        for src, tgt in score_a:
            pair = (min(src, tgt), max(src, tgt))
            all_pairs.add(pair)
        for src, tgt in score_b:
            pair = (min(src, tgt), max(src, tgt))
            all_pairs.add(pair)
        
        legal_a, legal_b = [], []
        
        for i, j in all_pairs:
            # 收集所有相关分数
            candidates = []
            if (i, j) in score_a:
                candidates.append((score_a[(i, j)], 'A', i, j))
            if (j, i) in score_a:
                candidates.append((score_a[(j, i)], 'A', j, i))
            if (i, j) in score_b:
                candidates.append((score_b[(i, j)], 'B', i, j))
            if (j, i) in score_b:
                candidates.append((score_b[(j, i)], 'B', j, i))
            
            if candidates:
                best_score, best_type, best_src, best_tgt = max(candidates, key=lambda x: x[0])
                if best_type == 'A':
                    legal_a.append((best_src, best_tgt, best_score))
                else:
                    legal_b.append((best_src, best_tgt, best_score))
        
        return legal_a, legal_b

    
    def _process_tree_relations(self, relations_by_type, num_nodes):
        """
        处理 Parent/Child 关系 - 分别处理，保持原始类型
        
        Parent(A, B) 表示 A 是 B 的父节点 (A → B)
        Child(A, B) 表示 B 是 A 的子节点 (A → B，A 是父)
        
        约束：
        - Parent: 每个节点最多一个 Parent（作为 tgt）
        - Child: 每个节点最多一个 Parent（作为 tgt，即 Child 的 tgt）
        
        注意：Parent 和 Child 是同一关系的两种表达，需要统一处理
        """
        legal = []
        
        parent_edges = relations_by_type.get(self.PARENT, [])
        child_edges = relations_by_type.get(self.CHILD, [])
        
        # 分别处理 Parent 和 Child，但共享树约束
        # 构建统一的 "父子关系" 图：src 是 tgt 的父节点
        
        # Parent(src, tgt): src 是 tgt 的父 → tgt 的入边
        # Child(src, tgt): src 是 tgt 的父 → tgt 的入边
        
        # 合并所有边，记录原始类型
        all_edges = []  # (src, tgt, score, rel_type)
        
        for src, tgt, score in parent_edges:
            all_edges.append((src, tgt, score, self.PARENT))
        
        for src, tgt, score in child_edges:
            all_edges.append((src, tgt, score, self.CHILD))
        
        if not all_edges:
            return legal
        
        nodes = list(range(num_nodes))
        
        if self.use_mst:
            # 使用 MST，但需要保留原始类型
            # 先用 MST 确定哪些 (src, tgt) 对被选中
            edges_for_mst = [(src, tgt, score) for src, tgt, score, _ in all_edges]
            
            virtual_root = num_nodes
            edges_with_root = edges_for_mst.copy()
            for node in nodes:
                edges_with_root.append((virtual_root, node, 0.0))
            
            mst_edges = ChuLiuEdmonds.max_spanning_arborescence(
                nodes + [virtual_root], 
                edges_with_root, 
                root=virtual_root
            )
            
            # 找出被选中的 (src, tgt) 对
            selected_pairs = set()
            for src, tgt, score in mst_edges:
                if src != virtual_root and score > 0:
                    selected_pairs.add((src, tgt))
            
            # 对于每个选中的对，找原始类型中分数最高的
            for src, tgt in selected_pairs:
                best_score = 0
                best_type = self.PARENT
                for s, t, score, rel_type in all_edges:
                    if s == src and t == tgt and score > best_score:
                        best_score = score
                        best_type = rel_type
                legal.append((src, tgt, best_type, best_score))
        else:
            # 贪心：分别处理 Parent 和 Child
            # Parent: 每个 tgt 最多一个入边
            legal_parent = self._greedy_single_parent(parent_edges, num_nodes)
            for src, tgt, score in legal_parent:
                legal.append((src, tgt, self.PARENT, score))
            
            # Child: 每个 tgt 最多一个入边（与 Parent 共享约束）
            # 需要排除已经有 Parent 的节点
            has_parent = set(tgt for _, tgt, _ in legal_parent)
            filtered_child = [(s, t, sc) for s, t, sc in child_edges if t not in has_parent]
            legal_child = self._greedy_single_parent(filtered_child, num_nodes)
            for src, tgt, score in legal_child:
                legal.append((src, tgt, self.CHILD, score))
        
        return legal
    
    def _greedy_single_parent(self, edges, num_nodes):
        """贪心：每个节点最多一个 Parent（优化版）"""
        if not edges:
            return []
        
        sorted_edges = sorted(edges, key=lambda x: -x[2])
        has_parent = set()
        adj = defaultdict(set)
        legal = []
        
        for src, tgt, score in sorted_edges:
            if tgt not in has_parent:
                # 检查是否会形成环
                if not self._can_reach(adj, tgt, src):
                    legal.append((src, tgt, score))
                    has_parent.add(tgt)
                    adj[src].add(tgt)
        
        return legal
    
    def _apply_dag_constraint(self, seq_edges, num_nodes):
        """
        DAG 约束（Sequence）：无环（优化版）
        
        使用增量式检测：维护已添加边的可达性
        """
        if not seq_edges:
            return []
        
        sorted_edges = sorted(seq_edges, key=lambda x: -x[2])
        
        # 使用邻接表和可达性缓存
        adj = defaultdict(set)
        legal = []
        
        for src, tgt, score in sorted_edges:
            # 快速检查：tgt 能否到达 src
            if not self._can_reach(adj, tgt, src):
                legal.append((src, tgt, score))
                adj[src].add(tgt)
        
        return legal
    
    def _can_reach(self, adj, start, target):
        """BFS 检查 start 能否到达 target（使用 deque 优化）"""
        if start == target:
            return True
        
        from collections import deque
        visited = {start}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()  # O(1) 而不是 list.pop(0) 的 O(n)
            for neighbor in adj[node]:
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
# ============ 便捷函数 ============

def apply_graph_constraints(pred_relations, num_nodes, 
                           apply_spatial=True, 
                           apply_tree=True, 
                           apply_dag=True,
                           use_mst=True,
                           score_threshold=0.3):
    """
    便捷函数：应用图约束
    
    Args:
        pred_relations: list of (src, tgt, rel_type, score)
        num_nodes: 节点数量
        apply_spatial: 是否应用空间约束
        apply_tree: 是否应用树约束
        apply_dag: 是否应用 DAG 约束
        use_mst: 是否使用 MST 算法
        score_threshold: 分数阈值
        
    Returns:
        legal_relations: list of (src, tgt, rel_type, score)
    """
    solver = GraphConstraintSolver(
        apply_spatial_constraints=apply_spatial,
        apply_tree_constraints=apply_tree,
        apply_dag_constraints=apply_dag,
        use_mst=use_mst,
        score_threshold=score_threshold
    )
    return solver.solve(pred_relations, num_nodes)


def apply_graph_constraints_numpy(src_arr, tgt_arr, rel_arr, scores_arr, num_nodes,
                                   apply_spatial=True, apply_tree=True, apply_dag=True,
                                   use_mst=False):
    """
    向量化版本的图约束求解（numpy 加速）
    
    Args:
        src_arr: numpy array [N] 源节点
        tgt_arr: numpy array [N] 目标节点
        rel_arr: numpy array [N] 关系类型
        scores_arr: numpy array [N] 分数
        num_nodes: 节点数量
        
    Returns:
        (legal_src, legal_tgt, legal_rel, legal_scores) 四个 numpy 数组
    """
    if len(src_arr) == 0 or num_nodes < 2:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    
    # 关系类型常量
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    PARENT, CHILD, SEQUENCE, REFERENCE = 4, 5, 6, 7
    
    legal_mask = np.ones(len(src_arr), dtype=bool)
    
    # 1. 空间关系约束（Up/Down, Left/Right 互斥 + 反对称）
    if apply_spatial:
        # Up/Down 互斥
        legal_mask &= _apply_spatial_constraint_numpy(
            src_arr, tgt_arr, rel_arr, scores_arr, UP, DOWN, legal_mask
        )
        # Left/Right 互斥
        legal_mask &= _apply_spatial_constraint_numpy(
            src_arr, tgt_arr, rel_arr, scores_arr, LEFT, RIGHT, legal_mask
        )
    
    # 2. Parent/Child 树约束（每个节点最多一个 Parent）
    if apply_tree:
        legal_mask &= _apply_tree_constraint_numpy(
            src_arr, tgt_arr, rel_arr, scores_arr, PARENT, CHILD, num_nodes, legal_mask
        )
    
    # 3. Sequence DAG 约束（无环）
    if apply_dag:
        legal_mask &= _apply_dag_constraint_numpy(
            src_arr, tgt_arr, rel_arr, scores_arr, SEQUENCE, legal_mask
        )
    
    # 返回合法的边
    return (src_arr[legal_mask].astype(np.int64),
            tgt_arr[legal_mask].astype(np.int64),
            rel_arr[legal_mask].astype(np.int64),
            scores_arr[legal_mask].astype(np.float32))


def _apply_spatial_constraint_numpy(src, tgt, rel, scores, rel_a, rel_b, mask):
    """
    空间关系约束（向量化）：
    - A(i,j) 和 B(i,j) 互斥
    - A(i,j) 和 A(j,i) 反对称
    对于每个节点对 (i,j)，只保留分数最高的一个关系
    """
    new_mask = mask.copy()
    
    # 找出所有 rel_a 和 rel_b 的边
    is_a = (rel == rel_a) & mask
    is_b = (rel == rel_b) & mask
    
    if not (is_a.any() or is_b.any()):
        return new_mask
    
    # 构建节点对到最高分的映射
    # 对于每个无序节点对 (min(i,j), max(i,j))，找出最高分的边
    pair_best = {}  # (i, j) -> (score, index, rel_type)
    
    for idx in np.where(is_a | is_b)[0]:
        s, t, r, sc = src[idx], tgt[idx], rel[idx], scores[idx]
        pair = (min(s, t), max(s, t))
        
        if pair not in pair_best or sc > pair_best[pair][0]:
            pair_best[pair] = (sc, idx, r, s, t)
    
    # 标记非最优的边
    for idx in np.where(is_a | is_b)[0]:
        s, t = src[idx], tgt[idx]
        pair = (min(s, t), max(s, t))
        
        if pair in pair_best and pair_best[pair][1] != idx:
            new_mask[idx] = False
    
    return new_mask


def _apply_tree_constraint_numpy(src, tgt, rel, scores, parent_rel, child_rel, num_nodes, mask):
    """
    树约束（向量化）：每个节点最多一个 Parent
    贪心选择：按分数降序，每个 tgt 只保留第一个
    """
    new_mask = mask.copy()
    
    # 找出所有 Parent 和 Child 边
    is_tree = ((rel == parent_rel) | (rel == child_rel)) & mask
    
    if not is_tree.any():
        return new_mask
    
    # 按分数降序排序
    tree_indices = np.where(is_tree)[0]
    sorted_idx = tree_indices[np.argsort(-scores[tree_indices])]
    
    # 贪心：每个 tgt 只保留第一个
    has_parent = set()
    
    for idx in sorted_idx:
        t = tgt[idx]
        if t in has_parent:
            new_mask[idx] = False
        else:
            has_parent.add(t)
    
    return new_mask


def _apply_dag_constraint_numpy(src, tgt, rel, scores, seq_rel, mask):
    """
    DAG 约束（向量化）：Sequence 关系无环
    贪心选择：按分数降序，跳过会形成环的边
    """
    new_mask = mask.copy()
    
    # 找出所有 Sequence 边
    is_seq = (rel == seq_rel) & mask
    
    if not is_seq.any():
        return new_mask
    
    # 按分数降序排序
    seq_indices = np.where(is_seq)[0]
    sorted_idx = seq_indices[np.argsort(-scores[seq_indices])]
    
    # 贪心 + 环检测
    from collections import deque
    adj = defaultdict(set)
    
    def can_reach(start, target):
        if start == target:
            return True
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
    
    for idx in sorted_idx:
        s, t = src[idx], tgt[idx]
        if can_reach(t, s):
            new_mask[idx] = False
        else:
            adj[s].add(t)
    
    return new_mask


