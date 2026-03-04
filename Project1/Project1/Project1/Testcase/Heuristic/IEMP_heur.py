import argparse
import os


class IEMData:
    """
    数据类：封装所有IEM问题相关数据，包括图结构和种子集
    """
    
    def __init__(self):
        self.n_nodes = 0           # 图中节点总数
        self.n_edges = 0           # 图中边总数
        self.graph = {}            # 正向邻接表：graph[u] = [(v, p1, p2), ...]
                                   # 表示u指向v的边，带两个campaign的传播概率
        self.reverse_graph = {}    # 反向邻接表：reverse_graph[v] = [(u, p1, p2), ...]
                                   # 表示指向v的节点，用于LFA策略
        self.I1 = set()            # Campaign 1 的初始种子集 (题目给定，不可更改)
        self.I2 = set()            # Campaign 2 的初始种子集 (题目给定，不可更改)
        self.S1 = set()            # Campaign 1 的平衡种子集 (算法输出，需要求解)
        self.S2 = set()            # Campaign 2 的平衡种子集 (算法输出，需要求解)
    
    def load_graph(self, filepath):
        """
        读取图数据文件 (dataset1 或 dataset2)
        
        文件格式：
            第1行: n_nodes n_edges
            第2行起: u v p1 p2
            
        参数:
            filepath: 图数据文件路径
        """
        self.graph = {}
        self.reverse_graph = {}
        
        with open(filepath, 'r') as f:
            # 读取第1行：节点数和边数
            line = f.readline().strip().split()
            self.n_nodes = int(line[0])
            self.n_edges = int(line[1])
            
            # 初始化所有节点的邻接表（正向和反向）
            for i in range(self.n_nodes):
                self.graph[i] = []
                self.reverse_graph[i] = []
            
            # 读取每条边
            for _ in range(self.n_edges):
                u, v, p1, p2 = f.readline().strip().split()
                u, v = int(u), int(v)
                p1, p2 = float(p1), float(p2)
                
                # p1 = campaign 1的传播概率
                # p2 = campaign 2的传播概率
                self.graph[u].append((v, p1, p2))           # 正向边 u -> v
                self.reverse_graph[v].append((u, p1, p2))   # 反向边 v <- u

    
    def load_initial_seeds(self, filepath):
        """
        读取初始种子文件 (seed)
        
        文件格式：
            第1行: |I1| |I2|  (两个campaign的种子数量)
            接下来 |I1| 行: campaign 1的种子节点ID
            接下来 |I2| 行: campaign 2的种子节点ID
        
        参数:
            filepath: 种子文件路径
        """
        with open(filepath, 'r') as f:
            # 读取种子数量
            n1, n2 = map(int, f.readline().strip().split())
            
            # 读取campaign 1的种子（共n1个）
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            # 读取campaign 2的种子（共n2个）
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
    
    def save_solution(self, filepath):
        """
        保存算法求解结果到文件 (seed_balanced)
        
        文件格式：
            第1行: |S1| |S2|
            接下来 |S1| 行: campaign 1的平衡种子节点ID
            接下来 |S2| 行: campaign 2的平衡种子节点ID
        
        参数:
            filepath: 输出文件路径
        """
        with open(filepath, 'w') as f:
            # 写入种子数量
            f.write(f"{len(self.S1)} {len(self.S2)}\n")
            # 写入S1的节点（升序排列，方便查看）
            for node in sorted(self.S1):
                f.write(f"{node}\n")
            # 写入S2的节点（升序排列）
            for node in sorted(self.S2):
                f.write(f"{node}\n")


class IEMPHeuristic:
    """
    IMRank启发式算法实现类
    
    核心算法：通过迭代寻找"自洽排名"（Self-Consistent Ranking）
    - 自洽排名：节点的排名与它的排名基边际影响力一致
    """

    def __init__(self, data, budget=10, max_iter=20, candidate_size=200, alpha=0.8, balance_lambda=0.05):
        """
        初始化启发式算法
        
        参数:
            data: IEMData对象，包含图数据和种子信息
            budget: 总预算k，即 |S1| + |S2| <= k，默认10
            max_iter: IMRank最大迭代次数，防止无限循环，默认20
        """
        self.data = data
        self.budget = budget       # 总预算 k = |S1| + |S2|
        self.max_iter = max_iter   # 最大迭代次数
        self.candidate_size = candidate_size
        self.alpha = alpha
        self.balance_lambda = balance_lambda
    
    def compute_degree_ranking(self):
        """
        计算按出度降序的初始排名
        
        原理：度数高的节点通常影响力更大，作为IMRank的初始排名
        
        返回:
            节点ID列表，按度数从高到低排序
        """
        degrees = []
        for node in range(self.data.n_nodes):
            # 节点node的出度 = 正向邻接表中邻居数量
            deg = len(self.data.graph[node])
            degrees.append((node, deg))
        
        # 按度数降序排序（reverse=True表示降序，从大到小）
        degrees.sort(key=lambda x: x[1], reverse=True)

        # 只返回节点ID，不返回度数
        return [node for node, deg in degrees]

    def compute_weighted_degree_ranking(self, campaign_idx):
        weighted_degrees = []
        for node in range(self.data.n_nodes):
            score = 0.0
            for _, p1, p2 in self.data.graph[node]:
                score += p1 if campaign_idx == 0 else p2
            weighted_degrees.append((node, score, len(self.data.graph[node])))

        weighted_degrees.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)
        return [node for node, _, _ in weighted_degrees]
    
    def lfa_strategy(self, ranking, campaign_idx):
        """
        LFA (Last-to-First Allocating) 策略
        计算每个节点的排名基边际影响力 M_r(v)
        
        核心思想（论文中的精髓）：
        - 排名低的节点的影响力会传递给排名高的邻居
        - 从排名最后一个节点向前扫描
        - 利用反向图高效计算，无需蒙特卡洛模拟
        
        直观理解：
        - M_r(v) 表示在排名r下，节点v能激活的期望节点数
        - 包括v自己和通过v激活的所有排名更低的节点
        
        参数:
            ranking: 当前排名列表 [node1, node2, ...]，按影响力从高到低排序
            campaign_idx: 0 或 1，选择使用哪个传播概率 (p1 或 p2)
        
        返回:
            M: 字典 {node: 边际影响力值}
        """
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

    def imrank_self_consistent(self, campaign_idx, initial_ranking=None):
        if initial_ranking is None:
            ranking = self.compute_weighted_degree_ranking(campaign_idx)
        else:
            ranking = list(initial_ranking)

        if not ranking:
            return {}, []

        last_ranking = None
        final_M = None
        for _ in range(self.max_iter):
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

    def node_pick_score(self, node, M1, M2, max_m1, max_m2, s1_size, s2_size, for_s1):
        m1 = M1.get(node, 1.0) / max_m1
        m2 = M2.get(node, 1.0) / max_m2
        if for_s1:
            base = m1 - self.alpha * m2
            penalty = self.balance_lambda * max(0, s1_size - s2_size)
        else:
            base = m2 - self.alpha * m1
            penalty = self.balance_lambda * max(0, s2_size - s1_size)
        return base - penalty
    
    def run(self):
        """
        贪心算法：基于对称差最小化选择 S1 和 S2
        目标：最小化 E[|r(S1∪I1) Δ r(S2∪I2)|]
        """
        available = set(range(self.data.n_nodes)) - self.data.I1 - self.data.I2
        S1, S2 = set(), set()

        if not available or self.budget <= 0:
            self.data.S1 = S1
            self.data.S2 = S2
            return S1, S2

        M1, rank1 = self.imrank_self_consistent(campaign_idx=0)
        M2, rank2 = self.imrank_self_consistent(campaign_idx=1)

        max_m1 = max(M1.values()) if M1 else 1.0
        max_m2 = max(M2.values()) if M2 else 1.0
        if max_m1 <= 0:
            max_m1 = 1.0
        if max_m2 <= 0:
            max_m2 = 1.0

        candidate_pool = self.build_candidate_pool(available, rank1, rank2)
        steps = min(self.budget, len(available))

        for i in range(steps):
            best_score = -float('inf')
            best_node = None
            best_for_s1 = True

            print(f"Selecting {i+1}/{steps}...")

            for node in candidate_pool:
                if node not in available:
                    continue
                score_s1 = self.node_pick_score(
                    node=node,
                    M1=M1,
                    M2=M2,
                    max_m1=max_m1,
                    max_m2=max_m2,
                    s1_size=len(S1),
                    s2_size=len(S2),
                    for_s1=True,
                )
                if score_s1 > best_score:
                    best_score = score_s1
                    best_node = node
                    best_for_s1 = True

                score_s2 = self.node_pick_score(
                    node=node,
                    M1=M1,
                    M2=M2,
                    max_m1=max_m1,
                    max_m2=max_m2,
                    s1_size=len(S1),
                    s2_size=len(S2),
                    for_s1=False,
                )
                if score_s2 > best_score:
                    best_score = score_s2
                    best_node = node
                    best_for_s1 = False

            if best_node is None:
                remaining = sorted(available)
                if not remaining:
                    break
                best_node = remaining[0]
                best_for_s1 = len(S1) <= len(S2)

            if best_for_s1:
                S1.add(best_node)
                print(f"  -> S1 adds node {best_node}, score={best_score:.4f}")
            else:
                S2.add(best_node)
                print(f"  -> S2 adds node {best_node}, score={best_score:.4f}")

            available.remove(best_node)

        self.data.S1 = S1
        self.data.S2 = S2
        return S1, S2

    def compute_balance_gain(self, node, S1, S2, I1, I2, for_s1):
        """
        计算将 node 加入 S1 或 S2 对减少对称差的边际贡献（基于LFA估计）
        """
        # 当前对称差（用LFA快速估计）
        curr = self.estimate_sym_diff_lfa(S1, S2, I1, I2)
        
        # 加入 node 后的对称差
        if for_s1:
            new = self.estimate_sym_diff_lfa(S1 | {node}, S2, I1, I2)
        else:
            new = self.estimate_sym_diff_lfa(S1, S2 | {node}, I1, I2)
        
        # 增益 = 减少的对称差大小
        return curr - new

    def estimate_sym_diff_lfa(self, S1, S2, I1, I2):
        """
        用LFA策略快速估计期望对称差 E[|r(S1∪I1) Δ r(S2∪I2)|]
        
        核心思想：
        1. 用LFA估计每个节点被campaign 1和campaign 2激活的期望次数
        2. 期望对称差 = Σ_v [P1(v)(1-P2(v)) + (1-P1(v))P2(v)]
        其中 P1(v), P2(v) 是节点v被两个campaign激活的概率（用期望值近似）
        """
        # 估计两个campaign的激活概率分布
        prob1 = self.lfa_estimate_reach(S1 | I1, campaign=0)
        prob2 = self.lfa_estimate_reach(S2 | I2, campaign=1)
        
        # 计算期望对称差：对于每个节点，计算它只属于一个campaign的概率
        expected_sym_diff = 0.0
        for v in range(self.data.n_nodes):
            p1 = prob1.get(v, 0.0)
            p2 = prob2.get(v, 0.0)
            # 节点v贡献的对称差期望 = P1*(1-P2) + (1-P1)*P2
            expected_sym_diff += p1 * (1 - p2) + (1 - p1) * p2
        
        return expected_sym_diff

    def lfa_estimate_reach(self, seeds, campaign):
        """
        用类似LFA的策略估计每个节点被**到达/暴露**的期望次数（作为概率的近似）
        
        根据IEM问题定义，"暴露节点"包括：
        - 种子节点
        - 被尝试激活的节点（无论成功与否）
        
        算法：
        1. 初始：种子节点的到达期望为1.0（确定被到达）
        2. 传播阶段：
           - 一个节点被"到达"当它至少有一个父节点被激活（尝试激活它）
           - 一个节点被"激活"当它被到达且激活尝试成功
        3. 按BFS顺序传播，同时追踪到达概率和激活概率
        """
        # 初始化：种子节点的到达概率和激活概率都为1.0
        exp_reached = {node: 0.0 for node in range(self.data.n_nodes)}  # 被到达的概率
        exp_active = {node: 0.0 for node in range(self.data.n_nodes)}   # 被激活的概率
        
        for seed in seeds:
            exp_reached[seed] = 1.0
            exp_active[seed] = 1.0
        
        # 按拓扑层次传播
        max_iter = 10  # 最大传播深度/迭代次数
        
        for _ in range(max_iter):
            new_reached = exp_reached.copy()
            new_active = exp_active.copy()
            
            for u in range(self.data.n_nodes):
                if exp_active[u] > 0:  # 只有被激活的节点才会尝试传播
                    # u向其邻居传播
                    for v, p1, p2 in self.data.graph.get(u, []):
                        p = p1 if campaign == 0 else p2
                        
                        # v被u尝试激活的概率 = u被激活的概率
                        # 只要有父节点尝试，v就被视为"reached"
                        if exp_active[u] > 0.001:
                            # v被到达的概率更新（OR公式：至少一个父节点激活它）
                            # P(v reached) = 1 - (1 - P(u active)) * (1 - current P(v reached))
                            new_reached[v] = 1.0 - (1.0 - exp_active[u]) * (1.0 - new_reached[v])
                        
                        # v被成功激活的概率（只有被到达且激活成功才算）
                        # P(v active) 增量 = P(u active) * p(u->v)
                        contribution = exp_active[u] * p
                        if contribution > 0.001:
                            # OR公式：至少一个父节点成功激活它
                            new_active[v] = 1.0 - (1.0 - contribution) * (1.0 - new_active[v])
            
            exp_reached = new_reached
            exp_active = new_active
        
        # 返回到达概率（包含尝试但未激活的节点）
        return exp_reached


def main():
    parser = argparse.ArgumentParser(description="IEM Heuristic Solver")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")
    parser.add_argument("--max-iter", type=int, default=10, help="IMRank max iterations")
    parser.add_argument("--candidate-size", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Cross-campaign penalty weight")
    parser.add_argument("--balance-lambda", type=float, default=0.05, help="S1/S2 size imbalance penalty")

    args = parser.parse_args()

    if args.budget <= 0:
        raise ValueError("Budget k must be a positive integer.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be a positive integer.")
    if args.candidate_size <= 0:
        raise ValueError("--candidate-size must be a positive integer.")
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative.")
    if args.balance_lambda < 0:
        raise ValueError("--balance-lambda must be non-negative.")

    if not os.path.exists(args.network):
        raise FileNotFoundError(f"Network file not found: {args.network}")
    if not os.path.exists(args.initial):
        raise FileNotFoundError(f"Initial seed file not found: {args.initial}")

    data = IEMData()
    data.load_graph(args.network)
    data.load_initial_seeds(args.initial)

    heuristic = IEMPHeuristic(
        data,
        budget=args.budget,
        max_iter=args.max_iter,
        candidate_size=args.candidate_size,
        alpha=args.alpha,
        balance_lambda=args.balance_lambda,
    )
    heuristic.run()

    data.save_solution(args.balanced)


if __name__ == "__main__":
    main()
