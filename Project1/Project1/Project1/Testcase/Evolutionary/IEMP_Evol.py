"""
IEMP Evolutionary Algorithm - Binary Encoding with Penalty-based Constraint Handling
====================================================================================

This is a PURE evolutionary algorithm implementation WITHOUT any heuristic guidance.

Key features:
1. Binary encoding: x = [x_1, ..., x_n, x_{n+1}, ..., x_{2n}]
   - First n bits: nodes in S1
   - Last n bits: nodes in S2
   - Only available nodes (not in I1 or I2) are considered
   
2. Constraint handling: Penalty-based fitness function
   - If |S1| + |S2| <= k: fitness = MC_score
   - Else: fitness = -(|S1| + |S2|)
   
3. Genetic operators:
   - Crossover: Single-point or Uniform
   - Mutation: Bit-flip
   
4. Selection: Tournament selection with elitism

Reference: Standard GA for Influence Maximization (Particle Swarm Optimization paper)
"""

import argparse
import os
import random
import copy
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass, field


@dataclass
class Individual:
    """
    Binary encoding for IEM problem
    
    chromosome: List of 0/1 values, length = 2 * n_available
    - indices [0, n-1]: represent nodes in S1
    - indices [n, 2n-1]: represent nodes in S2
    """
    chromosome: List[int] = field(default_factory=list)
    fitness: float = None
    # Decoded solution (filled during evaluation)
    S1: Set[int] = field(default_factory=set)
    S2: Set[int] = field(default_factory=set)
    
    def copy(self) -> 'Individual':
        return Individual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness,
            S1=set(self.S1),
            S2=set(self.S2)
        )


class IEMData:
    """Data class for IEM problem"""
    
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph = {}
        self.I1 = set()
        self.I2 = set()
        self.available_nodes = []  # Nodes that can be selected (not in I1 or I2)
        self.node_to_idx = {}  # Map node ID to index in chromosome
    
    def load_graph(self, filepath: str):
        """Load graph from file"""
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
        """Load initial seed sets and identify available nodes"""
        with open(filepath, 'r') as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
        
        # Available nodes: not in I1 and not in I2
        self.available_nodes = [n for n in range(self.n_nodes) 
                                if n not in self.I1 and n not in self.I2]
        self.node_to_idx = {node: idx for idx, node in enumerate(self.available_nodes)}
    
    def save_solution(self, S1: Set[int], S2: Set[int], filepath: str):
        """Save solution to file"""
        with open(filepath, 'w') as f:
            f.write(f"{len(S1)} {len(S2)}\n")
            for node in sorted(S1):
                f.write(f"{node}\n")
            for node in sorted(S2):
                f.write(f"{node}\n")


class IEMPEvaluator:
    """Monte Carlo evaluator for fitness computation"""
    
    def __init__(self, data: IEMData):
        self.data = data
        self.cache = {}
    
    def clear_cache(self):
        self.cache = {}
    
    def evaluate(self, S1: Set[int], S2: Set[int], n_simulations: int) -> float:
        """Evaluate solution using MC simulation"""
        cache_key = (frozenset(S1), frozenset(S2), n_simulations)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        full_seeds_1 = self.data.I1 | S1
        full_seeds_2 = self.data.I2 | S2
        
        total = 0
        for _ in range(n_simulations):
            total += self._single_simulation(full_seeds_1, full_seeds_2)
        
        score = total / n_simulations
        self.cache[cache_key] = score
        return score
    
    def _single_simulation(self, seeds1: Set[int], seeds2: Set[int]) -> int:
        """Single IC simulation"""
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


class IEMPEvolutionary:
    """
    Pure Evolutionary Algorithm with Binary Encoding
    
    NO heuristic guidance (IMRank, LFA, etc.) is used.
    This is a completely different approach from the heuristic algorithm.
    """

    def __init__(self, data: IEMData, budget: int = 10,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.05,
                 elitism: int = 2,
                 mc_coarse: int = 30,
                 mc_fine: int = 200):
        """
        Initialize evolutionary algorithm
        
        Args:
            data: IEMData object
            budget: Total budget k (|S1| + |S2| <= k)
            population_size: Population size
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Bit-flip mutation probability per bit
            elitism: Number of elite individuals to preserve
            mc_coarse: MC simulations for fitness evaluation during evolution
            mc_fine: MC simulations for final evaluation
        """
        self.data = data
        self.budget = budget
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.mc_coarse = mc_coarse
        self.mc_fine = mc_fine
        
        # Chromosome length = 2 * number of available nodes
        self.n_available = len(data.available_nodes)
        self.chromosome_length = 2 * self.n_available
        
        # Evaluator
        self.evaluator = IEMPEvaluator(data)
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Individual = None
    
    # ==================== Encoding & Decoding ====================
    
    def decode(self, individual: Individual) -> Tuple[Set[int], Set[int]]:
        """
        Decode chromosome to S1 and S2
        
        chromosome[0:n] -> S1
        chromosome[n:2n] -> S2
        """
        S1 = set()
        S2 = set()
        
        n = self.n_available
        for i in range(n):
            if individual.chromosome[i] == 1:
                S1.add(self.data.available_nodes[i])
            if individual.chromosome[n + i] == 1:
                S2.add(self.data.available_nodes[i])
        
        individual.S1 = S1
        individual.S2 = S2
        return S1, S2
    
    def compute_fitness(self, individual: Individual, mc_simulations: int) -> float:
        """
        Compute fitness with penalty for constraint violation
        
        Fitness function:
        - If |S1| + |S2| <= k: fitness = MC_score + budget_usage_bonus
        - Else: fitness = -(|S1| + |S2|)
        
        Budget usage bonus encourages using more of the budget.
        """
        S1, S2 = self.decode(individual)
        total_size = len(S1) + len(S2)
        
        # Check constraint
        if total_size > self.budget:
            # Penalty for infeasible solution
            individual.fitness = -total_size
            return individual.fitness
        
        # Feasible solution: evaluate with MC
        score = self.evaluator.evaluate(S1, S2, mc_simulations)
        
        # Budget usage bonus: encourage using more budget (large reward to prioritize budget usage)
        budget_usage_ratio = total_size / self.budget if self.budget > 0 else 0
        budget_bonus = 200.0 * budget_usage_ratio  # Very large bonus to force full budget usage
        
        individual.fitness = score + budget_bonus
        return individual.fitness
    
    # ==================== Initialization ====================
    
    def create_random_individual(self) -> Individual:
        """Create random individual with biased bit probability"""
        # Each bit has low probability of being 1 (sparse solution)
        # Target: expected budget/2 nodes in each of S1 and S2
        # But we need to be very conservative to ensure feasibility
        sparsity = (self.budget / 2) / self.n_available if self.n_available > 0 else 0.05
        sparsity = min(sparsity * 0.5, 0.05)  # Very conservative: max 5%
        
        chromosome = [1 if random.random() < sparsity else 0 
                     for _ in range(self.chromosome_length)]
        return Individual(chromosome=chromosome)
    
    def create_degree_biased_individual(self) -> Individual:
        """
        Create individual biased by node degree (simple heuristic, not IMRank)
        Higher degree nodes have higher probability of being selected
        """
        # Compute degrees
        degrees = [len(self.data.graph[node]) for node in self.data.available_nodes]
        max_degree = max(degrees) if degrees else 1
        
        n = self.n_available
        chromosome = [0] * self.chromosome_length
        
        # Probability proportional to degree
        for i in range(n):
            prob = 0.3 * degrees[i] / max_degree  # Max 30% probability
            if random.random() < prob:
                chromosome[i] = 1  # S1
            if random.random() < prob:
                chromosome[n + i] = 1  # S2
        
        return Individual(chromosome=chromosome)
    
    def initialize_population(self):
        """Initialize population with random and degree-biased individuals, all repaired to use full budget"""
        print(f"\nInitializing population (size={self.population_size})...")
        self.population = []
        
        # 50% degree-biased (repaired to full budget)
        for _ in range(self.population_size // 2):
            ind = self.create_degree_biased_individual()
            ind = self.repair(ind)  # Repair to use full budget
            self.population.append(ind)
        
        # 50% random (repaired to full budget)
        for _ in range(self.population_size - len(self.population)):
            ind = self.create_random_individual()
            ind = self.repair(ind)  # Repair to use full budget
            self.population.append(ind)
        
        # Evaluate initial population
        budget_count = 0
        for i, ind in enumerate(self.population):
            self.compute_fitness(ind, self.mc_coarse)
            total_size = len(ind.S1) + len(ind.S2)
            if total_size == self.budget:
                budget_count += 1
            print(f"  Individual {i+1}: fitness={ind.fitness:.2f}, "
                  f"|S1|={len(ind.S1)}, |S2|={len(ind.S2)}, total={total_size}")
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"  Best initial: {self.best_individual.fitness:.2f}")
        print(f"  Full budget solutions: {budget_count}/{self.population_size}")
    
    # ==================== Crossover Operators ====================
    
    def single_point_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if len(p1.chromosome) <= 1:
            return p1.copy(), p2.copy()
        
        point = random.randint(1, len(p1.chromosome) - 1)
        
        c1 = Individual(chromosome=p1.chromosome[:point] + p2.chromosome[point:])
        c2 = Individual(chromosome=p2.chromosome[:point] + p1.chromosome[point:])
        
        return c1, c2
    
    def uniform_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        c1_chrom = []
        c2_chrom = []
        
        for bit1, bit2 in zip(p1.chromosome, p2.chromosome):
            if random.random() < 0.5:
                c1_chrom.append(bit1)
                c2_chrom.append(bit2)
            else:
                c1_chrom.append(bit2)
                c2_chrom.append(bit1)
        
        return Individual(chromosome=c1_chrom), Individual(chromosome=c2_chrom)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Apply crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Use uniform crossover (generally better for binary encoding)
        return self.uniform_crossover(parent1, parent2)
    
    # ==================== Mutation Operator ====================
    
    def bit_flip_mutation(self, individual: Individual) -> Individual:
        """
        Bit-flip mutation
        Each bit has mutation_rate probability of being flipped
        """
        mutant = individual.copy()
        
        for i in range(len(mutant.chromosome)):
            if random.random() < self.mutation_rate:
                mutant.chromosome[i] = 1 - mutant.chromosome[i]
        
        return mutant
    
    def repair(self, individual: Individual) -> Individual:
        """
        Repair operator: ensure solution uses exactly budget nodes
        - If over budget: randomly remove nodes
        - If under budget: randomly add nodes from available set
        """
        S1, S2 = self.decode(individual)
        total_size = len(S1) + len(S2)
        
        if total_size > self.budget:
            # Over budget: randomly remove nodes
            all_nodes = list(S1) + list(S2)
            nodes_to_remove = random.sample(all_nodes, total_size - self.budget)
            for node in nodes_to_remove:
                idx = self.data.available_nodes.index(node)
                if node in S1:
                    individual.chromosome[idx] = 0
                    S1.remove(node)
                elif node in S2:
                    individual.chromosome[self.n_available + idx] = 0
                    S2.remove(node)
        
        elif total_size < self.budget:
            # Under budget: randomly add nodes
            available_for_S1 = [n for n in self.data.available_nodes if n not in S1 and n not in S2]
            available_for_S2 = available_for_S1.copy()
            
            nodes_needed = self.budget - total_size
            for _ in range(nodes_needed):
                if random.random() < 0.5 and available_for_S1:
                    # Add to S1
                    node = random.choice(available_for_S1)
                    idx = self.data.available_nodes.index(node)
                    individual.chromosome[idx] = 1
                    S1.add(node)
                    available_for_S1.remove(node)
                    if node in available_for_S2:
                        available_for_S2.remove(node)
                elif available_for_S2:
                    # Add to S2
                    node = random.choice(available_for_S2)
                    idx = self.data.available_nodes.index(node)
                    individual.chromosome[self.n_available + idx] = 1
                    S2.add(node)
                    available_for_S2.remove(node)
                    if node in available_for_S1:
                        available_for_S1.remove(node)
                elif available_for_S1:
                    # Fallback: add to S1 if S2 empty
                    node = random.choice(available_for_S1)
                    idx = self.data.available_nodes.index(node)
                    individual.chromosome[idx] = 1
                    S1.add(node)
                    available_for_S1.remove(node)
        
        return individual
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation and repair"""
        mutant = self.bit_flip_mutation(individual)
        return self.repair(mutant)
    
    # ==================== Selection ====================
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        candidates = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)
    
    # ==================== Evolution Loop ====================
    
    def evolve(self):
        """Main evolution loop"""
        print("\n" + "=" * 60)
        print("Binary Encoding Evolutionary Algorithm")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Population: {self.population_size}")
        print(f"  Generations: {self.generations}")
        print(f"  Early stopping: DISABLED (will run all generations)")
        print(f"  Budget: {self.budget}")
        print(f"  Available nodes: {self.n_available}")
        print(f"  Chromosome length: {self.chromosome_length}")
        print(f"  Crossover rate: {self.crossover_rate}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  MC (coarse): {self.mc_coarse}, MC (fine): {self.mc_fine}")
        
        # Initialize
        self.initialize_population()
        
        # Statistics
        best_fitness_history = [self.best_individual.fitness]
        full_budget_count_history = [sum(1 for ind in self.population 
                                         if len(ind.S1) + len(ind.S2) == self.budget)]
        
        # Evolution loop
        no_improvement_count = 0
        
        for gen in range(self.generations):
            new_population = []
            
            # Elitism: preserve best individuals
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            elites = [ind.copy() for ind in sorted_pop[:self.elitism]]
            new_population.extend(elites)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                # Evaluation
                self.compute_fitness(c1, self.mc_coarse)
                self.compute_fitness(c2, self.mc_coarse)
                
                new_population.extend([c1, c2])
            
            self.population = new_population[:self.population_size]
            
            # Statistics
            current_best = max(self.population, key=lambda x: x.fitness)
            full_budget_count = sum(1 for ind in self.population 
                                   if len(ind.S1) + len(ind.S2) == self.budget)
            full_budget_count_history.append(full_budget_count)
            
            # Update best
            if current_best.fitness > self.best_individual.fitness:
                improvement = current_best.fitness - self.best_individual.fitness
                self.best_individual = current_best.copy()
                no_improvement_count = 0
                if gen % 10 == 0 or improvement > 10:
                    print(f"Gen {gen+1}: New best = {self.best_individual.fitness:.2f}, "
                          f"full_budget={full_budget_count}/{self.population_size}")
            else:
                no_improvement_count += 1
            
            best_fitness_history.append(self.best_individual.fitness)
            
            # Progress report
            if (gen + 1) % 20 == 0:
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                print(f"Gen {gen+1}: Best={self.best_individual.fitness:.2f}, "
                      f"Avg={avg_fitness:.2f}, FullBudget={full_budget_count}/{self.population_size}, "
                      f"NoImprove={no_improvement_count}")
            
            # Early stopping - DISABLED to run full generations
            # if no_improvement_count >= 40:
            #     print(f"  Early stopping at generation {gen+1} (no improvement for 40 gens)")
            #     break
        
        # Final fine evaluation
        print("\n" + "=" * 60)
        print("Final Fine Evaluation")
        print("=" * 60)
        self.evaluator.clear_cache()
        
        # Re-evaluate best with fine MC
        final_fitness = self.compute_fitness(self.best_individual, self.mc_fine)
        
        print(f"Final Fitness: {final_fitness:.4f}")
        print(f"S1 ({len(self.best_individual.S1)} nodes): {sorted(self.best_individual.S1)}")
        print(f"S2 ({len(self.best_individual.S2)} nodes): {sorted(self.best_individual.S2)}")
        print(f"Total: {len(self.best_individual.S1) + len(self.best_individual.S2)} / {self.budget}")
        
        # Check feasibility
        if len(self.best_individual.S1) + len(self.best_individual.S2) <= self.budget:
            print("Solution is FEASIBLE")
        else:
            print("Solution is INFEASIBLE (this shouldn't happen for best individual)")
        
        print("=" * 60)
        
        return self.best_individual.S1, self.best_individual.S2


def main():
    parser = argparse.ArgumentParser(description="IEM Evolutionary Algorithm (Binary Encoding)")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")
    
    # EA parameters
    parser.add_argument("--pop-size", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--generations", type=int, default=300, help="Number of generations (default: 300)")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate (default: 0.8)")
    parser.add_argument("--mutation-rate", type=float, default=0.05, 
                       help="Bit-flip mutation rate (default: 0.05)")
    parser.add_argument("--elitism", type=int, default=2, help="Number of elites (default: 2)")
    parser.add_argument("--mc-coarse", type=int, default=30,
                       help="MC simulations for evolution (default: 30)")
    parser.add_argument("--mc-fine", type=int, default=200,
                       help="MC simulations for final eval (default: 200)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    if not os.path.exists(args.network):
        raise FileNotFoundError(f"Network file not found: {args.network}")
    if not os.path.exists(args.initial):
        raise FileNotFoundError(f"Initial seed file not found: {args.initial}")
    
    # Load data
    data = IEMData()
    data.load_graph(args.network)
    data.load_initial_seeds(args.initial)
    
    print(f"Graph: {data.n_nodes} nodes, {data.n_edges} edges")
    print(f"Initial seeds: I1={len(data.I1)}, I2={len(data.I2)}")
    print(f"Available nodes: {len(data.available_nodes)}")
    print(f"Budget k: {args.budget}")
    
    # Run evolutionary algorithm
    ea = IEMPEvolutionary(
        data=data,
        budget=args.budget,
        population_size=args.pop_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        mc_coarse=args.mc_coarse,
        mc_fine=args.mc_fine,
    )
    
    S1, S2 = ea.evolve()
    
    # Save solution
    data.save_solution(S1, S2, args.balanced)
    print(f"\nSolution saved to: {args.balanced}")


if __name__ == "__main__":
    main()
