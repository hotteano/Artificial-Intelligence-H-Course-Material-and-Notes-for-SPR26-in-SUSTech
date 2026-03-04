"""
IEMP (Information Exposure Maximization Problem) Heuristic Algorithm
基于 IMRank (Influence Maximization via Finding Self-Consistent Ranking) 的启发式算法

核心思想：
1. 初始排名：按节点度数排序
2. LFA策略：从后向前计算每个节点的排名基边际影响力
3. 迭代优化：按边际影响力重新排名，直到收敛
4. 输出：选择top-k节点作为平衡种子集 S1, S2
"""


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

    def __init__(self, data, budget=10, max_iter=20):
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
        # 创建节点到排名的映射：node -> position (0-indexed)
        # 排名越靠前（影响力越大），position越小
        pos = {node: i for i, node in enumerate(ranking)}

        # 初始化边际影响力：每个节点至少能激活自己，所以初始值为1
        M = {node: 1.0 for node in ranking}

        # 从后向前扫描：从排名最低（列表末尾）到排名最高（列表开头）
        # range(start, stop, step): 从 len-1 到 1，步长 -1
        for i in range(len(ranking) - 1, 0, -1): 
            v = ranking[i]   # 当前节点（排名位置为i，排名较低）

            # 遍历v的所有入边邻居（谁指向v，即谁可以激活v）
            # 使用反向图：reverse_graph[v] = [(u, p1, p2), ...] 表示 u -> v
            for u, p1, p2 in self.data.graph.get(v, []):

                # 如果u在排名中且排名比v高（位置数字更小）
                if u in pos and pos[u] < i:

                    # 选择对应campaign的传播概率
                    # campaign_idx=0 用 p1（第一个概率）
                    # campaign_idx=1 用 p2（第二个概率）
                    p = p1 if campaign_idx == 0 else p2

                    # u的边际影响力 += v的边际影响力 * 传播概率
                    # 含义：u可以通过激活v，获得v能激活的所有节点
                    M[u] += M[v] * p 
        
        return M
    
    def run_imrank(self): 
        """
        IMRank主算法
        
        算法流程：
        1. 初始化：按度数排序得到初始排名
        2. 迭代优化（最多max_iter次）：
           a. 用LFA计算当前排名下每个节点的边际影响力M
           b. 按边际影响力M重新排序（M大的排前面）
           c. 如果新排名的top-k与旧排名相同，则收敛，停止迭代
        3. 输出：选择新排名的top-k节点作为S1, S2
        
        返回:
            (S1, S2): 两个平衡种子集合，满足 |S1| + |S2| <= budget
        """
        # 步骤1：初始化排名（按出度降序）
        ranking1 = self.compute_degree_ranking()
        ranking2 = self.compute_degree_ranking()

        # 排除已存在的初始种子（不能重复选择， budgets只用于新选的S1, S2）
        available = set(range(self.data.n_nodes)) - self.data.I1 - self.data.I2
        ranking1 = [n for n in ranking1 if n in available]
        ranking2 = [n for n in ranking2 if n in available]

        # 步骤2：迭代优化
        for iteration in range(self.max_iter): 

            # 用LFA策略计算两个campaign的边际影响力
            # M1[v] = 节点v在campaign 1下的排名基边际影响力
            M1 = self.lfa_strategy(ranking1, 0)
            M2 = self.lfa_strategy(ranking2, 1)

            # 按边际影响力重新排序（降序）
            # key=lambda n: M1[n] 表示按M1值排序
            # reverse=True 表示降序（M大的排前面）
            new_ranking1 = sorted(available, key=lambda n: M1[n], reverse=True)
            new_ranking2 = sorted(available, key=lambda n: M2[n], reverse=True)

            # 预算分配：budget = k1 + k2
            # 整数除法：k1 = budget // 2
            # k2 取剩余部分，处理奇数预算的情况
            k1 = self.budget // 2
            k2 = self.budget - k1

            # 获取新旧排名的top-k节点集合（用于收敛判断）
            topk1_old = set(ranking1[:k1])
            topk2_old = set(ranking2[:k2])
            topk1_new = set(new_ranking1[:k1])
            topk2_new = set(new_ranking2[:k2])

            # 更新排名为新的排名
            ranking1, ranking2 = new_ranking1, new_ranking2

            # 收敛判断：如果top-k节点集合没有变化，说明已找到自洽排名
            if topk1_new == topk1_old and topk2_new == topk2_old:
                break  # 收敛，跳出循环
        
        # 步骤3：选择最终的top-k节点作为解
        k1 = self.budget // 2
        k2 = self.budget - k1

        # S1取ranking1的前k1个，S2取ranking2的前k2个
        self.data.S1 = set(ranking1[:k1])
        self.data.S2 = set(ranking2[:k2])

        return self.data.S1, self.data.S2


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 创建数据对象
    data = IEMData()
    
    # 读取图数据
    data.load_graph("map1/dataset1")
    
    # 读取初始种子
    data.load_initial_seeds("map1/seed")
    
    # 创建启发式算法对象
    # budget=10：总共选10个节点（S1+S2=10）
    # max_iter=10：最多迭代10次
    heuristic = IEMPHeuristic(data, budget=10, max_iter=10)
    
    # 运行IMRank算法
    S1, S2 = heuristic.run_imrank()
    
    print(f"S1: {S1}")
    print(f"S2: {S2}")
    
    # 保存结果到文件
    data.save_solution("map1/seed_balanced")
