"""
IEMP Advanced Evolutionary Algorithm
=====================================
Features:
1. Hierarchical Search: Coarse (low MC) → Fine (high MC)
2. Knowledge-Guided Crossover: Use fitness landscape information
3. Adaptive Mutation: Dynamically select mutation operators
4. Diversity Maintenance: Prevent premature convergence
"""

import argparse
import os
import random
import copy
import math
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Individual:
    """个体：表示一个候选解 (S1, S2)"""
    S1: Set[int] = field(default_factory=set)
    S2: Set[int] = field(default_factory=set)
    fitness: float = None
    # 记录评估历史，用于自适应策略
    eval_history: List[Tuple[str, float]] = field(default_factory=list)
    
    def copy(self) -> 'Individual':
        return Individual(
            S1=set(self.S1),
            S2=set(self.S2),
            fitness=self.fitness,
            eval_history=list(self.eval_history)
        )
    
    def total_size(self) -> int:
        return len(self.S1) + len(self.S2)
    
    def get_all_nodes(self) -> Set[int]:
        return self.S1 | self.S2
    
    def hash_key(self) -> str:
        """生成唯一标识，用于缓存"""
        return f"{sorted(self.S1)}|{sorted(self.S2)}"


class IEMData:
    """数据类"""
    
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph = {}
        self.I1 = set()
        self.I2 = set()
    
    def load_graph(self, filepath: str):
        self.graph = {}
        with open(filepath, 'r') as f:
            line = f.readline().strip().split()
            self.n_nodes = int(line[0])
            self.n_edges = int(line[1])
            
            for i in range(self.n_nodes):
                self.graph[i] = []
            
            for _ in range(self.n_edges):
                u, v, p1, p2 = f.readline().strip().split()
                u, v = int(u), int(v)
                p1, p2 = float(p1), float(p2)
                self.graph[u].append((v, p1, p2))
    
    def load_initial_seeds(self, filepath: str):
        with open(filepath, 'r') as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
    
    def save_solution(self, S1: Set[int], S2: Set[int], filepath: str):
        with open(filepath, 'w') as f:
            f.write(f"{len(S1)} {len(S2)}\n")
            for node in sorted(S1):
                f.write(f"{node}\n")
            for node in sorted(S2):
                f.write(f"{node}\n")


class IEMPEvaluator:
    """评估器：Monte Carlo Simulation"""
    
    def __init__(self, data: IEMData):
        self.data = data
        self.cache = {}  # 适应度缓存
    
    def clear_cache(self):
        """清除缓存（当MC次数变化时）"""
        self.cache = {}
    
    def evaluate(self, individual: Individual, n_simulations: int) -> float:
        """评估个体"""
        if individual.fitness is not None and n_simulations == individual.last_n_sim:
            return individual.fitness
        
        # 检查缓存
        cache_key = (individual.hash_key(), n_simulations)
        if cache_key in self.cache:
            individual.fitness = self.cache[cache_key]
            individual.last_n_sim = n_simulations
            return individual.fitness
        
        full_seeds_1 = self.data.I1 | individual.S1
        full_seeds_2 = self.data.I2 | individual.S2
        
        total_score = 0
        for _ in range(n_simulations):
            score = self._single_simulation(full_seeds_1, full_seeds_2)
            total_score += score
        
        individual.fitness = total_score / n_simulations
        individual.last_n_sim = n_simulations
        self.cache[cache_key] = individual.fitness
        
        return individual.fitness
    
    def _single_simulation(self, seeds1: Set[int], seeds2: Set[int]) -> int:
        """单次IC模拟"""
        # Campaign 1
        active1 = set(seeds1)
        reached1 = set(seeds1)
        newly_active1 = set(seeds1)
        
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
        active2 = set(seeds2)
        reached2 = set(seeds2)
        newly_active2 = set(seeds2)
        
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
        
        symmetric_diff = reached1.symmetric_difference(reached2)
        return self.data.n_nodes - len(symmetric_diff)


class IEMPAdvancedEvolutionary:
    """
    高级进化算法：分层搜索 + 改进交叉变异
    """
    
    def __init__(self, data: IEMData, budget: int = 10,
                 # 分层搜索参数
                 coarse_generations: int = 100,
                 coarse_mc: int = 30,
                 fine_generations: int = 50,
                 fine_mc: int = 300,
                 n_elites: int = 10,
                 # 种群参数
                 population_size: int = 40,
                 # 交叉变异参数
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.3,
                 # 多样性维持
                 diversity_threshold: float = 0.3,
                 # 模拟退火
                 use_sa: bool = True,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.95):
        
        self.data = data
        self.budget = budget
        self.available_nodes = list(set(range(data.n_nodes)) - data.I1 - data.I2)
        
        # 分层参数
        self.coarse_generations = coarse_generations
        self.coarse_mc = coarse_mc
        self.fine_generations = fine_generations
        self.fine_mc = fine_mc
        self.n_elites = n_elites
        
        # 种群参数
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 多样性维持
        self.diversity_threshold = diversity_threshold
        
        # 模拟退火
        self.use_sa = use_sa
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.current_temp = initial_temp
        
        # 评估器
        self.evaluator = IEMPEvaluator(data)
        
        # 自适应变异统计
        self.mutation_stats = {
            'add': {'success': 0, 'total': 0},
            'remove': {'success': 0, 'total': 0},
            'swap': {'success': 0, 'total': 0},
            'transfer': {'success': 0, 'total': 0},
            'replace': {'success': 0, 'total': 0},
        }
        
        # 种群
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.generation = 0
    
    # ==================== 初始化 ====================
    
    def create_random_individual(self) -> Individual:
        """创建随机个体"""
        ind = Individual()
        n_nodes = random.randint(1, self.budget)
        
        if n_nodes > 0 and self.available_nodes:
            selected = random.sample(self.available_nodes, 
                                   min(n_nodes, len(self.available_nodes)))
            for node in selected:
                if random.random() < 0.5:
                    ind.S1.add(node)
                else:
                    ind.S2.add(node)
        return ind
    
    def initialize_population(self) -> None:
        """初始化种群"""
        print(f"Initializing population (size={self.population_size})...")
        self.population = []
        
        # 50% 随机个体
        for _ in range(self.population_size // 2):
            self.population.append(self.create_random_individual())
        
        # 50% 贪心初始（基于度数）
        degrees = [(n, len(self.data.graph[n])) for n in self.available_nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in degrees[:self.budget * 3]]
        
        for _ in range(self.population_size - len(self.population)):
            ind = Individual()
            n_select = random.randint(1, self.budget)
            if top_nodes and n_select > 0:
                selected = random.sample(top_nodes, min(n_select, len(top_nodes)))
                for node in selected:
                    if random.random() < 0.5:
                        ind.S1.add(node)
                    else:
                        ind.S2.add(node)
            self.population.append(ind)
        
        # 评估初始种群
        for i, ind in enumerate(self.population):
            self.evaluator.evaluate(ind, self.coarse_mc)
            print(f"  Individual {i+1}: fitness={ind.fitness:.2f}, "
                  f"|S1|={len(ind.S1)}, |S2|={len(ind.S2)}")
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"  Best initial: {self.best_individual.fitness:.2f}")
    
    def initialize_from_elites(self, elites: List[Individual]) -> None:
        """从精英个体初始化"""
        print(f"\nInitializing from {len(elites)} elites...")
        self.population = [e.copy() for e in elites]
        
        # 对精英进行变异产生新个体
        while len(self.population) < self.population_size:
            parent = random.choice(elites)
            child = self.mutate(parent.copy(), force=True)
            self.evaluator.evaluate(child, self.fine_mc)
            self.population.append(child)
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"  Best initial (fine): {self.best_individual.fitness:.2f}")
    
    # ==================== 改进交叉 ====================
    
    def knowledge_guided_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        知识引导交叉：优先继承高适应度父代的节点
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = Individual(), Individual()
        
        # 确定哪个父代更好
        if parent1.fitness > parent2.fitness:
            better, worse = parent1, parent2
            bias = 0.7  # 70%概率选更好的
        else:
            better, worse = parent2, parent1
            bias = 0.7
        
        # 对 S1 进行交叉
        all_s1 = list(parent1.S1 | parent2.S1)
        for node in all_s1:
            in_better = node in better.S1
            in_worse = node in worse.S1
            
            if in_better and in_worse:
                child1.S1.add(node)
                child2.S1.add(node)
            elif in_better:
                if random.random() < bias:
                    child1.S1.add(node)
                else:
                    child2.S1.add(node)
            else:  # only in worse
                if random.random() < (1 - bias):
                    child1.S1.add(node)
                else:
                    child2.S1.add(node)
        
        # 对 S2 进行交叉
        all_s2 = list(parent1.S2 | parent2.S2)
        for node in all_s2:
            in_better = node in better.S2
            in_worse = node in worse.S2
            
            if in_better and in_worse:
                child1.S2.add(node)
                child2.S2.add(node)
            elif in_better:
                if random.random() < bias:
                    child1.S2.add(node)
                else:
                    child2.S2.add(node)
            else:
                if random.random() < (1 - bias):
                    child1.S2.add(node)
                else:
                    child2.S2.add(node)
        
        # 修复约束
        self._enforce_constraints(child1)
        self._enforce_constraints(child2)
        
        return child1, child2
    
    def _enforce_constraints(self, ind: Individual) -> None:
        """确保约束满足"""
        # S1 和 S2 不相交
        intersection = ind.S1 & ind.S2
        for node in intersection:
            if random.random() < 0.5:
                ind.S1.discard(node)
            else:
                ind.S2.discard(node)
        
        # 预算约束
        while ind.total_size() > self.budget:
            all_nodes = list(ind.get_all_nodes())
            if all_nodes:
                node = random.choice(all_nodes)
                ind.S1.discard(node)
                ind.S2.discard(node)
        
        ind.fitness = None
    
    # ==================== 自适应变异 ====================
    
    def adaptive_mutation(self, individual: Individual) -> Individual:
        """
        自适应变异：根据历史成功率选择变异算子
        """
        if random.random() > self.mutation_rate:
            return individual
        
        mutant = individual.copy()
        
        # 计算各算子的成功率
        success_rates = {}
        for op, stats in self.mutation_stats.items():
            if stats['total'] > 0:
                success_rates[op] = stats['success'] / stats['total']
            else:
                success_rates[op] = 0.2  # 默认值
        
        # 添加小概率探索新算子
        if random.random() < 0.1:  # 10%随机选择
            mutation_type = random.choice(list(self.mutation_stats.keys()))
        else:
            # 按成功率加权选择
            total = sum(success_rates.values())
            if total > 0:
                probs = [success_rates[op]/total for op in self.mutation_stats.keys()]
            else:
                probs = [0.2] * 5
            
            mutation_type = random.choices(
                list(self.mutation_stats.keys()),
                weights=probs
            )[0]
        
        # 执行变异
        old_fitness = individual.fitness
        
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
        
        self._enforce_constraints(mutant)
        
        # 更新统计（稍后评估后更新）
        mutant.mutation_type = mutation_type
        mutant.parent_fitness = old_fitness
        
        return mutant
    
    def mutate(self, individual: Individual, force: bool = False) -> Individual:
        """标准变异接口"""
        if not force and random.random() > self.mutation_rate:
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
        
        self._enforce_constraints(mutant)
        return mutant
    
    def _mutate_add(self, ind: Individual) -> None:
        if ind.total_size() >= self.budget:
            return
        available = set(self.available_nodes) - ind.get_all_nodes()
        if available:
            node = random.choice(list(available))
            if random.random() < 0.5:
                ind.S1.add(node)
            else:
                ind.S2.add(node)
    
    def _mutate_remove(self, ind: Individual) -> None:
        all_nodes = list(ind.get_all_nodes())
        if all_nodes:
            node = random.choice(all_nodes)
            ind.S1.discard(node)
            ind.S2.discard(node)
    
    def _mutate_swap(self, ind: Individual) -> None:
        if ind.S1 and ind.S2:
            n1 = random.choice(list(ind.S1))
            n2 = random.choice(list(ind.S2))
            ind.S1.remove(n1)
            ind.S1.add(n2)
            ind.S2.remove(n2)
            ind.S2.add(n1)
    
    def _mutate_transfer(self, ind: Individual) -> None:
        if random.random() < 0.5 and ind.S1:
            node = random.choice(list(ind.S1))
            ind.S1.remove(node)
            ind.S2.add(node)
        elif ind.S2:
            node = random.choice(list(ind.S2))
            ind.S2.remove(node)
            ind.S1.add(node)
    
    def _mutate_replace(self, ind: Individual) -> None:
        all_nodes = list(ind.get_all_nodes())
        available = set(self.available_nodes) - ind.get_all_nodes()
        if all_nodes and available:
            old_node = random.choice(all_nodes)
            new_node = random.choice(list(available))
            if old_node in ind.S1:
                ind.S1.remove(old_node)
                ind.S1.add(new_node)
            else:
                ind.S2.remove(old_node)
                ind.S2.add(new_node)
    
    # ==================== 选择 ====================
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(self.population, 
                                 min(tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)
    
    def diversity_score(self, ind1: Individual, ind2: Individual) -> float:
        """计算两个个体的差异度"""
        nodes1 = ind1.get_all_nodes()
        nodes2 = ind2.get_all_nodes()
        if not nodes1 and not nodes2:
            return 0.0
        intersection = len(nodes1 & nodes2)
        union = len(nodes1 | nodes2)
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    # ==================== 进化 ====================
    
    def evolve_one_generation(self, mc_simulations: int, track_stats: bool = False) -> None:
        """执行一代进化"""
        new_population = []
        
        # 精英保留
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = [ind.copy() for ind in sorted_pop[:3]]  # 保留Top 3
        new_population.extend(elites)
        
        # 生成后代
        while len(new_population) < self.population_size:
            # 选择
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # 交叉
            child1, child2 = self.knowledge_guided_crossover(parent1, parent2)
            
            # 变异
            if track_stats:
                child1 = self.adaptive_mutation(child1)
                child2 = self.adaptive_mutation(child2)
            else:
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
            
            # 评估
            self.evaluator.evaluate(child1, mc_simulations)
            self.evaluator.evaluate(child2, mc_simulations)
            
            # SA 接受
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
        
        # 更新最优
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        # 降温
        self.current_temp *= self.cooling_rate
        
        # 更新变异统计
        if track_stats:
            for ind in self.population:
                if hasattr(ind, 'mutation_type'):
                    op = ind.mutation_type
                    self.mutation_stats[op]['total'] += 1
                    if ind.fitness > ind.parent_fitness:
                        self.mutation_stats[op]['success'] += 1
    
    def simulated_annealing_accept(self, current: float, new: float) -> bool:
        """SA 接受准则"""
        if new >= current:
            return True
        if not self.use_sa or self.current_temp <= 0:
            return False
        prob = math.exp((new - current) / self.current_temp)
        return random.random() < prob
    
    # ==================== 分层搜索主流程 ====================
    
    def run(self) -> Tuple[Set[int], Set[int]]:
        """
        分层搜索主流程
        """
        print("\n" + "="*70)
        print("Advanced Evolutionary Algorithm with Hierarchical Search")
        print("="*70)
        print(f"Parameters:")
        print(f"  Budget k: {self.budget}")
        print(f"  Coarse: {self.coarse_generations} gens, MC={self.coarse_mc}")
        print(f"  Fine: {self.fine_generations} gens, MC={self.fine_mc}")
        print(f"  Population: {self.population_size}")
        print(f"  Crossover: {self.crossover_rate}, Mutation: {self.mutation_rate}")
        
        # ========== Phase 1: Coarse Evolution ==========
        print("\n" + "="*70)
        print("PHASE 1: Coarse Evolution (Exploration)")
        print("="*70)
        
        self.evaluator.clear_cache()
        self.initialize_population()
        
        for gen in range(self.coarse_generations):
            self.evolve_one_generation(self.coarse_mc, track_stats=True)
            
            if (gen + 1) % 20 == 0:
                print(f"Gen {gen+1}/{self.coarse_generations}: "
                      f"best={self.best_individual.fitness:.2f}, "
                      f"temp={self.current_temp:.2f}")
        
        # 选择精英
        print("\nSelecting elites from coarse phase...")
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # 用中等MC重新评估Top候选
        candidates = sorted_pop[:self.n_elites * 2]
        for ind in candidates:
            ind.fitness = None
            self.evaluator.evaluate(ind, 100)
        
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.copy() for ind in candidates[:self.n_elites]]
        
        print(f"Top {len(elites)} elites selected:")
        for i, e in enumerate(elites):
            print(f"  Elite {i+1}: {e.fitness:.2f}, |S1|={len(e.S1)}, |S2|={len(e.S2)}")
        
        # ========== Phase 2: Fine Evolution ==========
        print("\n" + "="*70)
        print("PHASE 2: Fine Evolution (Exploitation)")
        print("="*70)
        
        self.evaluator.clear_cache()
        self.current_temp = self.initial_temp / 2  # 降低初始温度
        self.initialize_from_elites(elites)
        
        for gen in range(self.fine_generations):
            self.evolve_one_generation(self.fine_mc, track_stats=False)
            
            if (gen + 1) % 10 == 0:
                print(f"Gen {gen+1}/{self.fine_generations}: "
                      f"best={self.best_individual.fitness:.2f}, "
                      f"temp={self.current_temp:.2f}")
        
        # ========== Final Evaluation ==========
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        final_fitness = self.evaluator.evaluate(self.best_individual, 1000)
        
        print(f"Final Fitness: {final_fitness:.4f}")
        print(f"S1 ({len(self.best_individual.S1)} nodes): {sorted(self.best_individual.S1)}")
        print(f"S2 ({len(self.best_individual.S2)} nodes): {sorted(self.best_individual.S2)}")
        print(f"Total: {self.best_individual.total_size()} / {self.budget}")
        print("="*70)
        
        return self.best_individual.S1, self.best_individual.S2


def main():
    parser = argparse.ArgumentParser(description="IEMP Advanced Evolutionary Algorithm")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")
    
    # 分层搜索参数
    parser.add_argument("--coarse-gens", type=int, default=100, help="Coarse phase generations (default: 100)")
    parser.add_argument("--coarse-mc", type=int, default=30, help="Coarse phase MC simulations (default: 30)")
    parser.add_argument("--fine-gens", type=int, default=50, help="Fine phase generations (default: 50)")
    parser.add_argument("--fine-mc", type=int, default=300, help="Fine phase MC simulations (default: 300)")
    parser.add_argument("--n-elites", type=int, default=10, help="Number of elites to keep (default: 10)")
    
    # 种群参数
    parser.add_argument("--pop-size", type=int, default=40, help="Population size (default: 40)")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate (default: 0.8)")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation rate (default: 0.3)")
    
    # SA参数
    parser.add_argument("--initial-temp", type=float, default=100.0, help="Initial temperature (default: 100)")
    parser.add_argument("--cooling-rate", type=float, default=0.95, help="Cooling rate (default: 0.95)")
    
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
    
    # 运行算法
    ea = IEMPAdvancedEvolutionary(
        data=data,
        budget=args.budget,
        coarse_generations=args.coarse_gens,
        coarse_mc=args.coarse_mc,
        fine_generations=args.fine_gens,
        fine_mc=args.fine_mc,
        n_elites=args.n_elites,
        population_size=args.pop_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
    )
    
    S1, S2 = ea.run()
    
    # 保存结果
    data.save_solution(S1, S2, args.balanced)
    print(f"\nSolution saved to: {args.balanced}")


if __name__ == "__main__":
    main()
