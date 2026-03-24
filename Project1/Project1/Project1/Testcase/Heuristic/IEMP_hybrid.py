"""
IEMP Hybrid Algorithm: IMRank + Monte Carlo

核心思想：
1. 使用 IMRank 快速生成高质量候选节点池（无需 MC）
2. 使用 Monte Carlo 精确评估从候选池中选择最优组合

优势：
- 比纯 MC 快：IMRank 快速缩小搜索空间
- 比纯 IMRank 准：MC 精确评估最终选择
"""

import argparse
import os
import random
from typing import List, Tuple, Set, Dict


class IEMData:
    """数据类：封装所有IEM问题相关数据"""
    
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph = {}
        self.reverse_graph = {}
        self.I1 = set()
        self.I2 = set()
        self.S1 = set()
        self.S2 = set()
    
    def load_graph(self, filepath):
        """读取图数据文件"""
        self.graph = {}
        self.reverse_graph = {}
        
        with open(filepath, 'r') as f:
            line = f.readline().strip().split()
            self.n_nodes = int(line[0])
            self.n_edges = int(line[1])
            
            for i in range(self.n_nodes):
                self.graph[i] = []
                self.reverse_graph[i] = []
            
            for _ in range(self.n_edges):
                u, v, p1, p2 = f.readline().strip().split()
                u, v = int(u), int(v)
                p1, p2 = float(p1), float(p2)
                self.graph[u].append((v, p1, p2))
                self.reverse_graph[v].append((u, p1, p2))
    
    def load_initial_seeds(self, filepath):
        """读取初始种子文件"""
        with open(filepath, 'r') as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
    
    def save_solution(self, filepath):
        """保存算法求解结果到文件"""
        with open(filepath, 'w') as f:
            f.write(f"{len(self.S1)} {len(self.S2)}\n")
            for node in sorted(self.S1):
                f.write(f"{node}\n")
            for node in sorted(self.S2):
                f.write(f"{node}\n")


class IEMPHybrid:
    """
    IMRank + Monte Carlo 混合算法
    """

    def __init__(self, data, budget=10, 
                 candidate_size=300,  # IMRank 候选池大小
                 mc_simulations=200,   # MC 评估模拟次数（贪心阶段）
                 mc_final_simulations=1000,  # 最终评估模拟次数
                 alpha=0.8, 
                 balance_lambda=0.05,
                 use_2opt=True,        # 是否使用 2-opt 优化
                 max_2opt_iter=50):    # 2-opt 最大迭代次数
        """
        初始化混合算法
        
        参数:
            data: IEMData对象
            budget: 总预算 k
            candidate_size: IMRank 候选池大小
            mc_simulations: MC 模拟次数（贪心选择阶段）
            mc_final_simulations: 最终评估的 MC 模拟次数
            use_2opt: 是否启用 2-opt 局部搜索
            max_2opt_iter: 2-opt 最大迭代次数
        """
        self.data = data
        self.budget = budget
        self.candidate_size = candidate_size
        self.mc_simulations = mc_simulations
        self.mc_final_simulations = mc_final_simulations
        self.alpha = alpha
        self.balance_lambda = balance_lambda
        self.use_2opt = use_2opt
        self.max_2opt_iter = max_2opt_iter
    
    # ============== IMRank 部分 ==============
    
    def compute_weighted_degree_ranking(self, campaign_idx):
        """计算加权度数排名"""
        weighted_degrees = []
        for node in range(self.data.n_nodes):
            score = 0.0
            for _, p1, p2 in self.data.graph[node]:
                score += p1 if campaign_idx == 0 else p2
            weighted_degrees.append((node, score, len(self.data.graph[node])))
        
        weighted_degrees.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)
        return [node for node, _, _ in weighted_degrees]
    
    def lfa_strategy(self, ranking, campaign_idx):
        """LFA (Last-to-First Allocating) 策略"""
        pos = {node: i for i, node in enumerate(ranking)}
        n = len(ranking)
        
        M = {node: 1.0 for node in ranking}
        
        for i in range(n - 1, 0, -1):
            v = ranking[i]
            remaining = M[v]
            if remaining <= 0:
                continue
            
            higher_ranked_parents = []
            for parent, p1, p2 in self.data.reverse_graph.get(v, []):
                parent_pos = pos.get(parent)
                if parent_pos is not None and parent_pos < i:
                    p = p1 if campaign_idx == 0 else p2
                    higher_ranked_parents.append((parent_pos, parent, p))
            
            higher_ranked_parents.sort(key=lambda x: x[0])
            for _, parent, p in higher_ranked_parents:
                influence = remaining * p
                if influence <= 0:
                    continue
                M[parent] += influence
                remaining *= (1 - p)
                if remaining <= 1e-12:
                    break
        
        return M
    
    def imrank_self_consistent(self, campaign_idx, max_iter=20):
        """计算自洽排名"""
        ranking = self.compute_weighted_degree_ranking(campaign_idx)
        
        if not ranking:
            return {}, []
        
        last_ranking = None
        final_M = None
        for _ in range(max_iter):
            M = self.lfa_strategy(ranking, campaign_idx)
            new_ranking = sorted(
                ranking,
                key=lambda node: (M.get(node, 0.0), len(self.data.graph[node]), -node),
                reverse=True,
            )
            final_M = M
            if new_ranking == ranking or new_ranking == last_ranking:
                ranking = new_ranking
                break
            last_ranking = ranking
            ranking = new_ranking
        
        if final_M is None:
            final_M = self.lfa_strategy(ranking, campaign_idx)
        
        return final_M, ranking
    
    def build_candidate_pool(self, available, rank1, rank2):
        """构建候选池"""
        available_set = set(available)
        pool = []
        seen = set()
        
        limit_each = max(1, self.candidate_size // 2)
        for node in rank1[:limit_each]:
            if node in available_set and node not in seen:
                pool.append(node)
                seen.add(node)
        for node in rank2[:limit_each]:
            if node in available_set and node not in seen:
                pool.append(node)
                seen.add(node)
        
        if len(pool) < min(self.candidate_size, len(available_set)):
            fallback = sorted(available_set - seen)
            for node in fallback:
                pool.append(node)
                if len(pool) >= min(self.candidate_size, len(available_set)):
                    break
        
        return pool
    
    # ============== Monte Carlo 部分 ==============
    
    def mc_simulation(self, seeds1: Set[int], seeds2: Set[int]) -> int:
        """
        运行一次 IC 模拟
        
        Returns:
            平衡曝光节点数 = n_nodes - |symmetric_diff|
        """
        full_seeds_1 = self.data.I1 | seeds1
        full_seeds_2 = self.data.I2 | seeds2
        
        # Campaign 1
        active1 = set(full_seeds_1)
        reached1 = set(full_seeds_1)
        newly_active1 = set(full_seeds_1)
        
        while newly_active1:
            current_new = set()
            for node in newly_active1:
                for neighbor, p1, p2 in self.data.graph.get(node, []):
                    if neighbor not in active1:
                        reached1.add(neighbor)
                        if random.random() < p1:
                            current_new.add(neighbor)
            newly_active1 = current_new - active1
            active1.update(newly_active1)
        
        # Campaign 2
        active2 = set(full_seeds_2)
        reached2 = set(full_seeds_2)
        newly_active2 = set(full_seeds_2)
        
        while newly_active2:
            current_new = set()
            for node in newly_active2:
                for neighbor, p1, p2 in self.data.graph.get(node, []):
                    if neighbor not in active2:
                        reached2.add(neighbor)
                        if random.random() < p2:
                            current_new.add(neighbor)
            newly_active2 = current_new - active2
            active2.update(newly_active2)
        
        # 计算平衡曝光
        symmetric_diff = reached1.symmetric_difference(reached2)
        return self.data.n_nodes - len(symmetric_diff)
    
    def mc_evaluate(self, seeds1: Set[int], seeds2: Set[int], n_simulations: int) -> float:
        """使用 Monte Carlo 评估解的质量"""
        if n_simulations <= 0:
            return 0.0
        
        total_score = 0
        for _ in range(n_simulations):
            total_score += self.mc_simulation(seeds1, seeds2)
        
        return total_score / n_simulations
    
    def mc_evaluate_add_node(self, S1: Set[int], S2: Set[int], 
                             node: int, for_s1: bool, n_simulations: int) -> float:
        """评估添加一个节点后的 MC 得分"""
        if for_s1:
            new_S1 = S1 | {node}
            new_S2 = S2
        else:
            new_S1 = S1
            new_S2 = S2 | {node}
        
        return self.mc_evaluate(new_S1, new_S2, n_simulations)
    
    # ============== 混合选择策略 ==============
    
    def greedy_mc_selection(self, candidate_pool: List[int], available: Set[int]) -> Tuple[Set[int], Set[int]]:
        """
        基于 Monte Carlo 的贪心选择
        
        每一步都用 MC 精确评估选择最佳节点
        """
        S1, S2 = set(), set()
        remaining_candidates = [n for n in candidate_pool if n in available]
        
        steps = min(self.budget, len(available))
        
        for i in range(steps):
            best_score = -float('inf')
            best_node = None
            best_for_s1 = True
            
            print(f"  MC-Greedy step {i+1}/{steps}...")
            
            # 评估每个候选节点加入 S1 的效果
            for node in remaining_candidates:
                score_s1 = self.mc_evaluate_add_node(S1, S2, node, True, self.mc_simulations)
                if score_s1 > best_score:
                    best_score = score_s1
                    best_node = node
                    best_for_s1 = True
            
            # 评估每个候选节点加入 S2 的效果
            for node in remaining_candidates:
                score_s2 = self.mc_evaluate_add_node(S1, S2, node, False, self.mc_simulations)
                if score_s2 > best_score:
                    best_score = score_s2
                    best_node = node
                    best_for_s1 = False
            
            if best_node is None:
                break
            
            # 添加最佳节点
            if best_for_s1:
                S1.add(best_node)
                print(f"    -> Add to S1: node {best_node}, score={best_score:.2f}")
            else:
                S2.add(best_node)
                print(f"    -> Add to S2: node {best_node}, score={best_score:.2f}")
            
            remaining_candidates.remove(best_node)
        
        return S1, S2
    
    def two_opt_improvement(self, S1: Set[int], S2: Set[int], 
                           candidate_pool: List[int]) -> Tuple[Set[int], Set[int]]:
        """
        2-opt 局部搜索优化
        
        尝试将已选节点与候选池中的节点交换，看是否能提高 MC 得分
        """
        if not self.use_2opt:
            return S1, S2
        
        print("\n  Starting 2-opt improvement...")
        
        current_score = self.mc_evaluate(S1, S2, self.mc_final_simulations)
        print(f"    Initial score: {current_score:.2f}")
        
        all_nodes = list(S1 | S2)
        improved = True
        iter_count = 0
        
        while improved and iter_count < self.max_2opt_iter:
            improved = False
            iter_count += 1
            
            for node_out in list(all_nodes):
                for node_in in candidate_pool:
                    if node_in in S1 or node_in in S2:
                        continue
                    
                    # 尝试交换
                    new_S1 = set(S1)
                    new_S2 = set(S2)
                    
                    if node_out in S1:
                        new_S1.remove(node_out)
                    else:
                        new_S2.remove(node_out)
                    
                    # 尝试将 node_in 加入 S1 或 S2
                    score_s1 = self.mc_evaluate(new_S1 | {node_in}, new_S2, 
                                                self.mc_simulations // 2)
                    score_s2 = self.mc_evaluate(new_S1, new_S2 | {node_in}, 
                                                self.mc_simulations // 2)
                    
                    if score_s1 > score_s2:
                        new_S1.add(node_in)
                        new_score = score_s1
                    else:
                        new_S2.add(node_in)
                        new_score = score_s2
                    
                    # 如果改进，接受交换
                    if new_score > current_score:
                        S1, S2 = new_S1, new_S2
                        current_score = new_score
                        all_nodes = list(S1 | S2)
                        improved = True
                        print(f"    2-opt iter {iter_count}: swap {node_out} -> {node_in}, "
                              f"score={current_score:.2f}")
                        break
                
                if improved:
                    break
        
        print(f"    Final 2-opt score: {current_score:.2f}")
        return S1, S2
    
    # ============== 主算法 ==============
    
    def run(self) -> Tuple[Set[int], Set[int]]:
        """
        运行混合算法
        
        流程：
        1. IMRank 生成候选池
        2. MC-based 贪心选择
        3. 2-opt 局部优化（可选）
        4. MC 最终评估
        """
        available = set(range(self.data.n_nodes)) - self.data.I1 - self.data.I2
        
        print("=" * 60)
        print("IMRank + Monte Carlo Hybrid Algorithm")
        print("=" * 60)
        
        # Step 1: IMRank 生成候选池
        print(f"\n[Step 1] IMRank generating candidate pool (size={self.candidate_size})...")
        M1, rank1 = self.imrank_self_consistent(campaign_idx=0)
        M2, rank2 = self.imrank_self_consistent(campaign_idx=1)
        
        candidate_pool = self.build_candidate_pool(available, rank1, rank2)
        print(f"  Candidate pool: {len(candidate_pool)} nodes")
        print(f"  Top 10 candidates: {candidate_pool[:10]}")
        
        # Step 2: MC-based 贪心选择
        print(f"\n[Step 2] MC-based greedy selection (simulations={self.mc_simulations})...")
        S1, S2 = self.greedy_mc_selection(candidate_pool, available)
        
        # Step 3: 2-opt 局部优化
        if self.use_2opt:
            S1, S2 = self.two_opt_improvement(S1, S2, candidate_pool)
        
        # Step 4: 最终 MC 评估
        print(f"\n[Step 4] Final MC evaluation (simulations={self.mc_final_simulations})...")
        final_score = self.mc_evaluate(S1, S2, self.mc_final_simulations)
        
        print("\n" + "=" * 60)
        print("Final Result")
        print("=" * 60)
        print(f"  S1 ({len(S1)} nodes): {sorted(S1)}")
        print(f"  S2 ({len(S2)} nodes): {sorted(S2)}")
        print(f"  Total: {len(S1) + len(S2)} / {self.budget}")
        print(f"  MC Score: {final_score:.4f}")
        print("=" * 60)
        
        self.data.S1 = S1
        self.data.S2 = S2
        return S1, S2


def main():
    parser = argparse.ArgumentParser(description="IEMP Hybrid Algorithm (IMRank + Monte Carlo)")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")
    parser.add_argument("--candidate-size", type=int, default=300, 
                       help="IMRank candidate pool size (default: 300)")
    parser.add_argument("--mc-simulations", type=int, default=200,
                       help="MC simulations for greedy selection (default: 200)")
    parser.add_argument("--mc-final", type=int, default=1000,
                       help="MC simulations for final evaluation (default: 1000)")
    parser.add_argument("--no-2opt", action="store_true",
                       help="Disable 2-opt improvement")
    parser.add_argument("--max-2opt-iter", type=int, default=50,
                       help="Max 2-opt iterations (default: 50)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # 加载数据
    data = IEMData()
    data.load_graph(args.network)
    data.load_initial_seeds(args.initial)
    
    print(f"Graph: {data.n_nodes} nodes, {data.n_edges} edges")
    print(f"I1: {len(data.I1)}, I2: {len(data.I2)}")
    print(f"Budget k: {args.budget}")
    
    # 运行混合算法
    hybrid = IEMPHybrid(
        data=data,
        budget=args.budget,
        candidate_size=args.candidate_size,
        mc_simulations=args.mc_simulations,
        mc_final_simulations=args.mc_final,
        use_2opt=not args.no_2opt,
        max_2opt_iter=args.max_2opt_iter,
    )
    
    S1, S2 = hybrid.run()
    
    # 保存结果
    data.save_solution(args.balanced)
    print(f"\nSolution saved to: {args.balanced}")


if __name__ == "__main__":
    main()
