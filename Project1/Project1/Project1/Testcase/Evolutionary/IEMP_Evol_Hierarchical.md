# 分层进化算法 (Hierarchical Evolutionary Algorithm)

## 核心思想

将进化过程分为两个阶段：
1. **粗粒度阶段**: 使用少量MC模拟快速评估，充分探索搜索空间
2. **细粒度阶段**: 在粗阶段找到的精英个体上，使用大量MC模拟精细优化

## 算法流程

```python
class IEMPHierarchicalEvolutionary:
    """
    分层进化算法
    """
    
    def run(self):
        """
        主流程
        """
        print("="*60)
        print("Hierarchical Evolutionary Algorithm")
        print("="*60)
        
        # ========== Phase 1: 粗粒度进化 ==========
        print("\n[Phase 1] Coarse Evolution")
        print("-"*60)
        
        # 配置粗阶段参数
        coarse_ga = IEMPGeneticAlgorithm(
            data=self.data,
            budget=self.budget,
            population_size=40,           # 较大种群
            generations=120,              # 较多代数
            eval_simulations=30,          # 少量MC (快速评估)
            use_sa=True,
            initial_temp=100.0,
            cooling_rate=0.97,
        )
        
        # 运行粗阶段进化
        coarse_ga.evolve()
        
        # 获取粗阶段Top N精英
        elite_individuals = self.select_elites(
            coarse_ga.population, 
            n_elites=10,                  # 保留10个精英
            eval_simulations=100          # 用更多MC重新评估精英
        )
        
        print(f"\nPhase 1 Complete. Top {len(elite_individuals)} elites selected.")
        for i, elite in enumerate(elite_individuals):
            print(f"  Elite {i+1}: fitness={elite.fitness:.2f}, "
                  f"|S1|={len(elite.S1)}, |S2|={len(elite.S2)}")
        
        # ========== Phase 2: 细粒度进化 ==========
        print("\n[Phase 2] Fine Evolution")
        print("-"*60)
        
        # 配置细阶段参数
        fine_ga = IEMPGeneticAlgorithm(
            data=self.data,
            budget=self.budget,
            population_size=20,           # 较小种群
            generations=50,               # 较少代数
            eval_simulations=300,         # 大量MC (精确评估)
            use_sa=True,
            initial_temp=50.0,            # 较低初始温度
            cooling_rate=0.95,
        )
        
        # 用精英个体初始化细阶段种群
        fine_ga.initialize_from_elites(elite_individuals)
        
        # 运行细阶段进化
        best_individual = fine_ga.evolve()
        
        # ========== 最终评估 ==========
        print("\n[Final Evaluation]")
        print("-"*60)
        final_fitness = fine_ga.evaluator.evaluate(
            best_individual, 
            n_simulations=1000            # 最终精确评估
        )
        
        print(f"Final Best Fitness: {final_fitness:.4f}")
        print(f"S1: {sorted(best_individual.S1)}")
        print(f"S2: {sorted(best_individual.S2)}")
        
        return best_individual.S1, best_individual.S2
    
    def select_elites(self, population, n_elites, eval_simulations):
        """
        从粗阶段种群中选择精英个体
        
        策略:
        1. 先用粗阶段的适应度排序
        2. Top 20% 用更多MC重新精确评估
        3. 按精确评估结果选择最终精英
        """
        # 按粗阶段适应度排序
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Top 20% 候选
        n_candidates = max(n_elites * 2, len(population) // 5)
        candidates = sorted_pop[:n_candidates]
        
        # 用更多MC重新评估
        print(f"\nRe-evaluating top {len(candidates)} candidates "
              f"with {eval_simulations} simulations...")
        
        for ind in candidates:
            # 清除缓存的适应度，强制重新评估
            ind.fitness = None
            self.evaluator.evaluate(ind, n_simulations=eval_simulations)
        
        # 按精确评估排序，选择最终精英
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.copy() for ind in candidates[:n_elites]]
        
        return elites


# ========== 关键修改：支持精英初始化的 GeneticAlgorithm ==========

class IEMPGeneticAlgorithm:
    """
    修改后的遗传算法，支持从精英初始化
    """
    
    def initialize_from_elites(self, elites):
        """
        从精英个体初始化种群
        
        策略:
        1. 保留所有精英
        2. 对精英进行变异产生新个体填充剩余种群
        """
        self.population = [elite.copy() for elite in elites]
        
        # 用变异产生剩余个体
        while len(self.population) < self.population_size:
            # 随机选择一个精英进行变异
            parent = random.choice(elites)
            child = self.mutate(parent.copy())
            
            # 评估新个体
            self.evaluator.evaluate(child, n_simulations=self.eval_simulations)
            self.population.append(child)
        
        # 更新最优个体
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        
        print(f"\nInitialized population from {len(elites)} elites.")
        print(f"Population size: {len(self.population)}")
        print(f"Best initial fitness: {self.best_individual.fitness:.2f}")


# ========== 另一种分层策略：动态评估精度 ==========

class IEMPAdaptiveEvaluation:
    """
    自适应评估精度：根据代数动态调整MC模拟次数
    """
    
    def get_eval_simulations(self, generation, total_generations):
        """
        动态计算当前代应使用的MC模拟次数
        
        策略: 前期少，后期多
        """
        # 线性增加
        progress = generation / total_generations
        
        if progress < 0.3:           # 前30%代: 探索阶段
            return 30
        elif progress < 0.7:         # 中间40%代: 过渡阶段
            return 100
        else:                        # 后30%代: 收敛阶段
            return 300
    
    def evolve_adaptive(self):
        """
        使用自适应评估精度的进化
        """
        for generation in range(self.generations):
            # 动态确定当前代的MC模拟次数
            n_sim = self.get_eval_simulations(generation, self.generations)
            self.eval_simulations = n_sim
            
            print(f"\nGeneration {generation+1}, MC simulations: {n_sim}")
            
            # 执行一代进化
            self.evolve_one_generation()


# ========== 分层搜索的另一种实现：多分辨率评估 ==========

class IEMPMultiResolution:
    """
    多分辨率评估：同时使用不同精度的评估
    """
    
    def __init__(self):
        # 三个不同精度的评估器
        self.fast_evaluator = IEMPEvaluator(mc_simulations=30)
        self.medium_evaluator = IEMPEvaluator(mc_simulations=100)
        self.accurate_evaluator = IEMPEvaluator(mc_simulations=500)
    
    def hierarchical_selection(self, population):
        """
        分层选择：先用快速评估筛选，再用精确评估确认
        """
        # Stage 1: 快速评估所有个体
        for ind in population:
            ind.fast_fitness = self.fast_evaluator.evaluate(ind)
        
        # 按快速评估排序，保留Top 50%
        sorted_by_fast = sorted(population, key=lambda x: x.fast_fitness, reverse=True)
        survivors = sorted_by_fast[:len(population)//2]
        
        # Stage 2: 中等评估幸存者
        for ind in survivors:
            ind.medium_fitness = self.medium_evaluator.evaluate(ind)
        
        # 按中等评估排序，保留Top 30%
        sorted_by_medium = sorted(survivors, key=lambda x: x.medium_fitness, reverse=True)
        finalists = sorted_by_medium[:len(survivors)//2]
        
        # Stage 3: 精确评估最终候选
        for ind in finalists:
            ind.accurate_fitness = self.accurate_evaluator.evaluate(ind)
        
        # 按精确评估排序
        sorted_by_accurate = sorted(finalists, key=lambda x: x.accurate_fitness, reverse=True)
        
        return sorted_by_accurate
