"""
IEMP Genetic Algorithm with Simulated Annealing

Problem: Information Exposure Maximization Problem (IEMP)
Objective: Maximize E[|V - (r(I1∪S1) △ r(I2∪S2))|]
           (平衡信息曝光，最小化对称差)

Encoding: Individual is represented as (S1, S2) where
          - S1: set of nodes for campaign 1
          - S2: set of nodes for campaign 2
          - |S1| + |S2| <= budget
          - S1 ∩ S2 = ∅
          - S1, S2 ⊆ V \ (I1 ∪ I2)
"""

import argparse
import os
import random
import copy
import math
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass, field


@dataclass
class Individual:
    """
    个体：表示一个候选解 (S1, S2)
    
    Attributes:
        S1: Campaign 1 的平衡种子集合
        S2: Campaign 2 的平衡种子集合
        fitness: 适应度值 (缓存，避免重复计算)
    """
    S1: Set[int] = field(default_factory=set)
    S2: Set[int] = field(default_factory=set)
    fitness: float = None
    
    def copy(self) -> 'Individual':
        """创建深拷贝"""
        return Individual(
            S1=set(self.S1),
            S2=set(self.S2),
            fitness=self.fitness
        )
    
    def total_size(self) -> int:
        """返回总种子数 |S1| + |S2|"""
        return len(self.S1) + len(self.S2)
    
    def get_all_nodes(self) -> Set[int]:
        """返回所有节点 S1 ∪ S2"""
        return self.S1 | self.S2


class IEMData:
    """数据类：封装所有IEM问题相关数据"""
    
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph = {}  # 正向邻接表
        self.reverse_graph = {}  # 反向邻接表
        self.I1 = set()  # Campaign 1 初始种子
        self.I2 = set()  # Campaign 2 初始种子
    
    def load_graph(self, filepath: str) -> None:
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
    
    def load_initial_seeds(self, filepath: str) -> None:
        """读取初始种子文件"""
        with open(filepath, 'r') as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
    
    def get_available_nodes(self) -> Set[int]:
        """返回可选节点集合 V \ (I1 ∪ I2)"""
        return set(range(self.n_nodes)) - self.I1 - self.I2
    
    def save_solution(self, S1: Set[int], S2: Set[int], filepath: str) -> None:
        """保存解到文件"""
        with open(filepath, 'w') as f:
            f.write(f"{len(S1)} {len(S2)}\n")
            for node in sorted(S1):
                f.write(f"{node}\n")
            for node in sorted(S2):
                f.write(f"{node}\n")


class IEMPEvaluator:
    """
    IEMP 评估器：使用 Monte Carlo 模拟计算适应度
    """
    
    def __init__(self, data: IEMData):
        self.data = data
        self.simulation_count = 0  # 统计模拟次数
    
    def ic_simulation(self, seeds: Set[int], campaign_idx: int) -> Set[int]:
        """
        运行一次 IC (Independent Cascade) 模拟
        
        Returns:
            所有被 reached/exposed 的节点集合
            (包含尝试激活但失败的节点)
        """
        active = set(seeds)
        reached = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            current_new = set()
            for node in newly_active:
                for neighbor, p1, p2 in self.data.graph.get(node, []):
                    if neighbor not in active:
                        reached.add(neighbor)
                        p = p1 if campaign_idx == 0 else p2
                        if random.random() < p:
                            current_new.add(neighbor)
            
            newly_active = current_new - active
            active.update(newly_active)
        
        return reached
    
    def evaluate(self, individual: Individual, n_simulations: int = 100) -> float:
        """
        评估个体的适应度
        
        Fitness = E[|V - (r(I1∪S1) △ r(I2∪S2))|]
               = 平均被两个阵营同时曝光或同时未曝光的节点数
        
        Args:
            individual: 待评估的个体
            n_simulations: Monte Carlo 模拟次数
            
        Returns:
            适应度值 (越高越好)
        """
        if individual.fitness is not None:
            return individual.fitness
        
        full_seeds_1 = self.data.I1 | individual.S1
        full_seeds_2 = self.data.I2 | individual.S2
        
        total_score = 0
        for _ in range(n_simulations):
            reached_1 = self.ic_simulation(full_seeds_1, 0)
            reached_2 = self.ic_simulation(full_seeds_2, 1)
            
            symmetric_diff = reached_1.symmetric_difference(reached_2)
            balanced_exposed = self.data.n_nodes - len(symmetric_diff)
            total_score += balanced_exposed
        
        self.simulation_count += n_simulations
        individual.fitness = total_score / n_simulations
        return individual.fitness
    
    def fast_estimate(self, individual: Individual) -> float:
        """
        快速估计适应度（使用较少的模拟次数）
        用于进化过程中的中间评估
        """
        return self.evaluate(individual, n_simulations=30)


class IEMPGeneticAlgorithm:
    """
    IEMP 遗传算法 + 模拟退火
    """
    
    def __init__(
        self,
        data: IEMData,
        budget: int = 10,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        elite_size: int = 5,
        tournament_size: int = 3,
        # 模拟退火参数
        use_sa: bool = True,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 1.0,
        # 评估参数
        eval_simulations: int = 100,
    ):
        self.data = data
        self.budget = budget
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # 模拟退火参数
        self.use_sa = use_sa
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_temp = initial_temp
        
        # 评估参数
        self.eval_simulations = eval_simulations
        
        # 初始化
        self.evaluator = IEMPEvaluator(data)
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.available_nodes = list(data.get_available_nodes())
        
    def create_random_individual(self) -> Individual:
        """
        创建随机个体
        
        随机选择 0 到 budget 个节点，随机分配到 S1 和 S2
        """
        individual = Individual()
        n_nodes = random.randint(0, self.budget)
        
        if n_nodes > 0 and self.available_nodes:
            selected = random.sample(self.available_nodes, min(n_nodes, len(self.available_nodes)))
            for node in selected:
                # 随机决定分配到 S1 还是 S2
                if random.random() < 0.5:
                    individual.S1.add(node)
                else:
                    individual.S2.add(node)
        
        return individual
    
    def create_greedy_individual(self) -> Individual:
        """
        创建基于度数的贪心个体
        
        优先选择度数高的节点
        """
        individual = Individual()
        
        # 计算节点度数
        degrees = [(node, len(self.data.graph[node])) for node in self.available_nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前 budget 个节点中的随机一部分
        top_nodes = [node for node, _ in degrees[:self.budget * 2]]
        n_select = random.randint(1, self.budget)
        
        if top_nodes and n_select > 0:
            selected = random.sample(top_nodes, min(n_select, len(top_nodes)))
            for node in selected:
                if random.random() < 0.5:
                    individual.S1.add(node)
                else:
                    individual.S2.add(node)
        
        return individual
    
    def initialize_population(self) -> None:
        """初始化种群"""
        print(f"Initializing population (size={self.population_size})...")
        
        self.population = []
        
        # 混合随机个体和贪心个体
        n_greedy = self.population_size // 4
        n_random = self.population_size - n_greedy
        
        for _ in range(n_greedy):
            self.population.append(self.create_greedy_individual())
        
        for _ in range(n_random):
            self.population.append(self.create_random_individual())
        
        # 评估初始种群
        for i, ind in enumerate(self.population):
            self.evaluator.fast_estimate(ind)
            print(f"  Individual {i+1}: fitness={ind.fitness:.2f}, |S1|={len(ind.S1)}, |S2|={len(ind.S2)}")
        
        # 初始化最优个体
        self.best_individual = max(self.population, key=lambda x: x.fitness).copy()
        print(f"  Best initial fitness: {self.best_individual.fitness:.2f}")
    
    def tournament_selection(self) -> Individual:
        """
        锦标赛选择
        
        从种群中随机选择 tournament_size 个个体，返回最优者
        """
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        交叉操作 (Uniform Crossover)
        
        对 S1 和 S2 分别进行均匀交叉
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = Individual()
        child2 = Individual()
        
        # S1 的交叉
        all_s1 = list(parent1.S1 | parent2.S1)
        for node in all_s1:
            in_p1 = node in parent1.S1
            in_p2 = node in parent2.S1
            
            if in_p1 and in_p2:
                # 两个父代都有，子代都有
                child1.S1.add(node)
                child2.S1.add(node)
            elif in_p1 and not in_p2:
                # 随机分配
                if random.random() < 0.5:
                    child1.S1.add(node)
                else:
                    child2.S1.add(node)
            elif in_p2 and not in_p1:
                if random.random() < 0.5:
                    child1.S1.add(node)
                else:
                    child2.S1.add(node)
        
        # S2 的交叉
        all_s2 = list(parent1.S2 | parent2.S2)
        for node in all_s2:
            in_p1 = node in parent1.S2
            in_p2 = node in parent2.S2
            
            if in_p1 and in_p2:
                child1.S2.add(node)
                child2.S2.add(node)
            elif in_p1 and not in_p2:
                if random.random() < 0.5:
                    child1.S2.add(node)
                else:
                    child2.S2.add(node)
            elif in_p2 and not in_p1:
                if random.random() < 0.5:
                    child1.S2.add(node)
                else:
                    child2.S2.add(node)
        
        # 确保 S1 和 S2 不相交
        self._enforce_disjoint(child1)
        self._enforce_disjoint(child2)
        
        # 确保预算约束
        self._enforce_budget(child1)
        self._enforce_budget(child2)
        
        return child1, child2
    
    def _enforce_disjoint(self, individual: Individual) -> None:
        """确保 S1 和 S2 不相交，从交集中随机选择一个集合删除"""
        intersection = individual.S1 & individual.S2
        for node in intersection:
            # 随机从 S1 或 S2 中删除
            if random.random() < 0.5:
                individual.S1.discard(node)
            else:
                individual.S2.discard(node)
    
    def _enforce_budget(self, individual: Individual) -> None:
        """确保个体满足预算约束 |S1| + |S2| <= budget"""
        while individual.total_size() > self.budget:
            # 随机删除一个节点
            all_nodes = list(individual.get_all_nodes())
            if all_nodes:
                node_to_remove = random.choice(all_nodes)
                individual.S1.discard(node_to_remove)
                individual.S2.discard(node_to_remove)
        individual.fitness = None  # 重置适应度
    
    def mutate(self, individual: Individual) -> Individual:
        """
        变异操作
        
        包含多种变异算子：
        1. Add Mutation: 添加新节点
        2. Remove Mutation: 删除节点
        3. Swap Mutation: 在 S1 和 S2 之间交换
        4. Transfer Mutation: 将节点从 S1 移到 S2 或反之
        5. Replace Mutation: 替换为新的节点
        """
        if random.random() > self.mutation_rate:
            return individual
        
        mutant = individual.copy()
        mutation_type = random.choice(['add', 'remove', 'swap', 'transfer', 'replace'])
        
        if mutation_type == 'add':
            self._mutate_add(mutant)
        elif mutation_type == 'remove':
            self._mutate_remove(mutant)
        elif mutation_type == 'swap':
            self._mutate_swap(mutant)
        elif mutation_type == 'transfer':
            self._mutate_transfer(mutant)
        elif mutation_type == 'replace':
            self._mutate_replace(mutant)
        
        # 确保 S1 和 S2 不相交
        self._enforce_disjoint(mutant)
        
        mutant.fitness = None  # 重置适应度
        return mutant
    
    def _mutate_add(self, individual: Individual) -> None:
        """添加变异：向 S1 或 S2 添加一个新节点"""
        if individual.total_size() >= self.budget:
            return
        
        available = set(self.available_nodes) - individual.get_all_nodes()
        if not available:
            return
        
        new_node = random.choice(list(available))
        if random.random() < 0.5:
            individual.S1.add(new_node)
        else:
            individual.S2.add(new_node)
    
    def _mutate_remove(self, individual: Individual) -> None:
        """删除变异：从 S1 或 S2 删除一个节点"""
        all_nodes = list(individual.get_all_nodes())
        if not all_nodes:
            return
        
        node_to_remove = random.choice(all_nodes)
        individual.S1.discard(node_to_remove)
        individual.S2.discard(node_to_remove)
    
    def _mutate_swap(self, individual: Individual) -> None:
        """交换变异：交换 S1 和 S2 中的各一个节点"""
        if not individual.S1 or not individual.S2:
            return
        
        node1 = random.choice(list(individual.S1))
        node2 = random.choice(list(individual.S2))
        
        individual.S1.remove(node1)
        individual.S1.add(node2)
        individual.S2.remove(node2)
        individual.S2.add(node1)
    
    def _mutate_transfer(self, individual: Individual) -> None:
        """转移变异：将节点从 S1 移到 S2 或反之"""
        if random.random() < 0.5 and individual.S1:
            node = random.choice(list(individual.S1))
            individual.S1.remove(node)
            individual.S2.add(node)
        elif individual.S2:
            node = random.choice(list(individual.S2))
            individual.S2.remove(node)
            individual.S1.add(node)
    
    def _mutate_replace(self, individual: Individual) -> None:
        """替换变异：用一个新节点替换现有节点"""
        all_nodes = list(individual.get_all_nodes())
        if not all_nodes:
            return
        
        available = set(self.available_nodes) - individual.get_all_nodes()
        if not available:
            return
        
        old_node = random.choice(all_nodes)
        new_node = random.choice(list(available))
        
        if old_node in individual.S1:
            individual.S1.remove(old_node)
            individual.S1.add(new_node)
        else:
            individual.S2.remove(old_node)
            individual.S2.add(new_node)
    
    def simulated_annealing_accept(self, current_fitness: float, new_fitness: float) -> bool:
        """
        模拟退火接受准则
        
        - 如果新解更好，总是接受
        - 如果新解更差，以概率 exp((new - current) / temp) 接受
        """
        if new_fitness >= current_fitness:
            return True
        
        if not self.use_sa or self.current_temp <= 0:
            return False
        
        delta = new_fitness - current_fitness
        prob = math.exp(delta / self.current_temp)
        return random.random() < prob
    
    def evolve(self) -> Individual:
        """
        主进化循环
        
        Returns:
            找到的最优个体
        """
        print("\n" + "="*60)
        print("Starting Evolution")
        print("="*60)
        
        self.initialize_population()
        
        no_improvement_count = 0
        best_fitness_history = []
        
        for generation in range(self.generations):
            print(f"\n--- Generation {generation + 1}/{self.generations} (Temp={self.current_temp:.2f}) ---")
            
            # 创建新种群
            new_population = []
            
            # 保留精英
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            elites = [ind.copy() for ind in sorted_pop[:self.elite_size]]
            new_population.extend(elites)
            
            # 生成后代
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 评估
                self.evaluator.fast_estimate(child1)
                self.evaluator.fast_estimate(child2)
                
                # 模拟退火接受
                if self.simulated_annealing_accept(parent1.fitness, child1.fitness):
                    new_population.append(child1)
                else:
                    new_population.append(parent1.copy())
                
                if len(new_population) < self.population_size:
                    if self.simulated_annealing_accept(parent2.fitness, child2.fitness):
                        new_population.append(child2)
                    else:
                        new_population.append(parent2.copy())
            
            self.population = new_population[:self.population_size]
            
            # 更新最优个体
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_individual.fitness:
                improvement = current_best.fitness - self.best_individual.fitness
                self.best_individual = current_best.copy()
                no_improvement_count = 0
                print(f"  *** New best fitness: {self.best_individual.fitness:.4f} (+{improvement:.4f})")
            else:
                no_improvement_count += 1
                print(f"  Best fitness: {self.best_individual.fitness:.4f} (no improvement for {no_improvement_count} gen)")
            
            best_fitness_history.append(self.best_individual.fitness)
            
            # 降温
            if self.use_sa:
                self.current_temp = max(self.min_temp, self.current_temp * self.cooling_rate)
            
            # 早停检查
            if no_improvement_count >= 20:
                print(f"\nEarly stopping: No improvement for 20 generations")
                break
        
        # 最终精确评估
        print("\n" + "="*60)
        print("Final Evaluation")
        print("="*60)
        final_fitness = self.evaluator.evaluate(self.best_individual, n_simulations=self.eval_simulations)
        print(f"Best Individual:")
        print(f"  S1 ({len(self.best_individual.S1)} nodes): {sorted(self.best_individual.S1)}")
        print(f"  S2 ({len(self.best_individual.S2)} nodes): {sorted(self.best_individual.S2)}")
        print(f"  Total: {self.best_individual.total_size()} / {self.budget}")
        print(f"  Fitness: {final_fitness:.4f}")
        print(f"  Total simulations: {self.evaluator.simulation_count}")
        
        return self.best_individual
    
    def run(self) -> Tuple[Set[int], Set[int]]:
        """运行遗传算法并返回最优解"""
        best = self.evolve()
        return best.S1, best.S2


def main():
    parser = argparse.ArgumentParser(description="IEM Evolutionary Algorithm Solver")
    parser.add_argument("-n", required=True, help="Path to social network file")
    parser.add_argument("-i", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", type=int, required=True, help="Budget k")
    # 可选参数（用于调优，但评分时可能不使用）
    parser.add_argument("--pop-size", type=int, default=40, help="Population size (default: 40)")
    parser.add_argument("--generations", type=int, default=80, help="Max generations (default: 80)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
    
    # 参数验证
    if args.k <= 0:
        raise ValueError("Budget k must be positive")
    if not os.path.exists(args.n):
        raise FileNotFoundError(f"Network file not found: {args.n}")
    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Initial seed file not found: {args.i}")
    
    # 加载数据
    print("Loading data...")
    data = IEMData()
    data.load_graph(args.n)
    data.load_initial_seeds(args.i)
    print(f"  Nodes: {data.n_nodes}, Edges: {data.n_edges}")
    print(f"  I1: {len(data.I1)}, I2: {len(data.I2)}")
    print(f"  Available: {len(data.get_available_nodes())}")
    
    # 运行遗传算法（使用默认参数）
    ga = IEMPGeneticAlgorithm(
        data=data,
        budget=args.k,
        population_size=args.pop_size,
        generations=args.generations,
    )
    
    S1, S2 = ga.run()
    
    # 保存结果
    data.save_solution(S1, S2, args.b)
    print(f"\nSolution saved to: {args.b}")


if __name__ == "__main__":
    main()
