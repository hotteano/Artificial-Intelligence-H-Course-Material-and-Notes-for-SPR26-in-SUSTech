import os
import random
from typing import List, Tuple, Set, Dict


class IEMEvaluator:
    """
    Information Exposure Maximization (IEM) Evaluator
    
    This class handles:
    1. Reading graph data (nodes, edges, propagation probabilities)
    2. Reading initial seed sets for both campaigns
    3. Reading balanced seed sets (solution)
    4. Computing the objective function using Monte Carlo simulation
    """
    
    def __init__(self):
        self.n_nodes: int = 0
        self.n_edges: int = 0
        self.graph: Dict[int, List[Tuple[int, float, float]]] = {}  # node -> [(neighbor, p1, p2), ...]
        self.initial_seeds: List[Set[int]] = [set(), set()]  # I1, I2
        self.balanced_seeds: List[Set[int]] = [set(), set()]  # S1, S2
        
    def read_graph(self, filepath: str) -> None:
        """
        Read graph file (dataset1 or dataset2)
        
        File format:
        n_nodes n_edges
        u v p1 p2  (for each edge)
        
        where p1 = probability for campaign 1, p2 = probability for campaign 2
        """
        self.graph = {}
        with open(filepath, 'r') as f:
            # Read first line: number of nodes and edges
            first_line = f.readline().strip().split()
            self.n_nodes = int(first_line[0])
            self.n_edges = int(first_line[1])
            
            # Initialize graph
            for i in range(self.n_nodes):
                self.graph[i] = []
            
            # Read edges
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                u = int(parts[0])
                v = int(parts[1])
                p1 = float(parts[2])
                p2 = float(parts[3])
                self.graph[u].append((v, p1, p2))
    
    def read_initial_seeds(self, filepath: str) -> None:
        """
        Read initial seed sets for both campaigns
        
        File format:
        |I1| |I2|
        (I1 nodes, one per line)
        (I2 nodes, one per line)
        """
        self.initial_seeds = [set(), set()]
        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            n1 = int(first_line[0])
            n2 = int(first_line[1])
            
            # Read campaign 1 seeds
            for _ in range(n1):
                node = int(f.readline().strip())
                self.initial_seeds[0].add(node)
            
            # Read campaign 2 seeds
            for _ in range(n2):
                node = int(f.readline().strip())
                self.initial_seeds[1].add(node)
    
    def read_balanced_seeds(self, filepath: str) -> None:
        """
        Read balanced seed sets (solution to evaluate)
        
        File format:
        |S1| |S2|
        (S1 nodes, one per line)
        (S2 nodes, one per line)
        """
        self.balanced_seeds = [set(), set()]
        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            n1 = int(first_line[0])
            n2 = int(first_line[1])
            
            # Read campaign 1 balanced seeds
            for _ in range(n1):
                node = int(f.readline().strip())
                self.balanced_seeds[0].add(node)
            
            # Read campaign 2 balanced seeds
            for _ in range(n2):
                node = int(f.readline().strip())
                self.balanced_seeds[1].add(node)
    
    def read_data(self, dataset_path: str, seed_path: str, seed_balanced_path: str) -> None:
        """
        Read all data files
        
        Args:
            dataset_path: Path to dataset file (e.g., dataset1 or dataset2)
            seed_path: Path to initial seed file
            seed_balanced_path: Path to balanced seed file (solution)
        """
        self.read_graph(dataset_path)
        self.read_initial_seeds(seed_path)
        self.read_balanced_seeds(seed_balanced_path)
        
    def ic_simulation(self, seeds: Set[int], campaign_idx: int) -> Set[int]:
        """
        Run one IC (Independent Cascade) simulation
        
        Args:
            seeds: Set of seed nodes
            campaign_idx: 0 for campaign 1, 1 for campaign 2
            
        Returns:
            Set of all activated (exposed) nodes
        """
        active = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            current_new = set()
            for node in newly_active:
                for neighbor, p1, p2 in self.graph.get(node, []):
                    if neighbor not in active:
                        # Get the appropriate probability for this campaign
                        p = p1 if campaign_idx == 0 else p2
                        if random.random() < p:
                            current_new.add(neighbor)
            
            newly_active = current_new - active
            active.update(newly_active)
        
        return active
    
    def evaluate(self, n_simulations: int = 1000) -> float:
        """
        Evaluate the objective function (balanced information exposure)
        
        The objective is: |V - (r1 △ r2)| = number of nodes that are either
        reached by both campaigns or reached by neither campaign.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Expected value of balanced information exposure
        """
        total_score = 0
        
        # Full seed sets for both campaigns
        full_seeds_1 = self.initial_seeds[0] | self.balanced_seeds[0]
        full_seeds_2 = self.initial_seeds[1] | self.balanced_seeds[1]
        
        for _ in range(n_simulations):
            # Run IC simulation for both campaigns
            exposed_1 = self.ic_simulation(full_seeds_1, 0)
            exposed_2 = self.ic_simulation(full_seeds_2, 1)
            
            # Calculate symmetric difference: nodes exposed to exactly one campaign
            symmetric_diff = exposed_1.symmetric_difference(exposed_2)
            
            # Balanced exposure: nodes NOT in symmetric difference
            balanced_exposed = self.n_nodes - len(symmetric_diff)
            total_score += balanced_exposed
        
        return total_score / n_simulations


def main():
    """
    Main function for command-line usage
    """
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python Evaluator.py <dataset_path> <seed_path> <seed_balanced_path> [n_simulations]")
        print("Example: python Evaluator.py map1/dataset1 map1/seed map1/seed_balanced 1000")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    seed_path = sys.argv[2]
    seed_balanced_path = sys.argv[3]
    n_simulations = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    
    # Create evaluator and read data
    evaluator = IEMEvaluator()
    evaluator.read_data(dataset_path, seed_path, seed_balanced_path)
    
    print(f"Graph: {evaluator.n_nodes} nodes, {evaluator.n_edges} edges")
    print(f"Initial seeds: Campaign 1: {len(evaluator.initial_seeds[0])}, Campaign 2: {len(evaluator.initial_seeds[1])}")
    print(f"Balanced seeds: Campaign 1: {len(evaluator.balanced_seeds[0])}, Campaign 2: {len(evaluator.balanced_seeds[1])}")
    print(f"Running {n_simulations} simulations...")
    
    # Evaluate
    score = evaluator.evaluate(n_simulations)
    print(f"\nBalanced Information Exposure: {score:.4f}")


if __name__ == "__main__":
    main()
