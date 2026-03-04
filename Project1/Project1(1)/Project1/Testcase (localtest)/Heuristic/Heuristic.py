"""
IMRank Algorithm for Information Exposure Maximization (IEM)

Based on: "IMRank: Influence Maximization via Finding Self-Consistent Ranking"
(SIGIR 2014)

This implementation adapts IMRank for the IEM problem with two campaigns.
"""

import os
import random
import sys
from typing import List, Tuple, Set, Dict
from collections import defaultdict


class IMRankIEM:
    """
    IMRank algorithm for IEM problem
    
    Finds balanced seed sets S1 and S2 that maximize:
    Φ(S1, S2) = |V - (r(I1∪S1) △ r(I2∪S2))|
    """
    
    def __init__(self, max_iter: int = 10, l: int = 1):
        """
        Args:
            max_iter: Maximum number of iterations for IMRank
            l: Influence path length (1 or 2, default 1 for speed)
        """
        self.max_iter = max_iter
        self.l = l
        
        # Graph data
        self.n_nodes: int = 0
        self.n_edges: int = 0
        self.graph: Dict[int, List[Tuple[int, float, float]]] = {}  # node -> [(neighbor, p1, p2)]
        self.reverse_graph: Dict[int, List[Tuple[int, float, float]]] = {}  # for reverse lookup
        
        # Seed sets
        self.I1: Set[int] = set()  # Initial seeds for campaign 1
        self.I2: Set[int] = set()  # Initial seeds for campaign 2
        
        # Results
        self.S1: Set[int] = set()  # Balanced seeds for campaign 1
        self.S2: Set[int] = set()  # Balanced seeds for campaign 2
        
    def read_graph(self, filepath: str) -> None:
        """Read graph file"""
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        
        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            self.n_nodes = int(first_line[0])
            self.n_edges = int(first_line[1])
            
            for i in range(self.n_nodes):
                self.graph[i] = []
                self.reverse_graph[i] = []
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                u = int(parts[0])
                v = int(parts[1])
                p1 = float(parts[2])
                p2 = float(parts[3])
                self.graph[u].append((v, p1, p2))
                self.reverse_graph[v].append((u, p1, p2))
    
    def read_initial_seeds(self, filepath: str) -> None:
        """Read initial seed sets"""
        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            n1 = int(first_line[0])
            n2 = int(first_line[1])
            
            self.I1 = set()
            self.I2 = set()
            
            for _ in range(n1):
                node = int(f.readline().strip())
                self.I1.add(node)
            
            for _ in range(n2):
                node = int(f.readline().strip())
                self.I2.add(node)
    
    def compute_degree_ranking(self) -> List[int]:
        """
        Compute initial ranking by out-degree
        Returns: List of node IDs sorted by degree (highest first)
        """
        degrees = [(node, len(self.graph[node])) for node in range(self.n_nodes)]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in degrees]
    
    def compute_strength_ranking(self) -> List[int]:
        """
        Compute initial ranking by node strength (sum of propagation probabilities)
        Returns: List of node IDs sorted by strength (highest first)
        """
        strengths = []
        for node in range(self.n_nodes):
            # Sum of p1 and p2 for all outgoing edges
            strength = sum(p1 + p2 for _, p1, p2 in self.graph[node])
            strengths.append((node, strength))
        strengths.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in strengths]
    
    def lfa_strategy(self, ranking: List[int], campaign_idx: int) -> Dict[int, float]:
        """
        Last-to-First Allocating (LFA) strategy
        
        Compute ranking-based marginal influence spread for all nodes
        
        Args:
            ranking: Current ranking of nodes (list of node IDs, best first)
            campaign_idx: 0 for campaign 1, 1 for campaign 2
            
        Returns:
            Dictionary mapping node ID to marginal influence M_r(v)
        """
        # Create position mapping: node -> position in ranking (0-indexed)
        # Only for nodes in the ranking
        pos = {node: i for i, node in enumerate(ranking)}
        
        # Initialize marginal influence: each node in ranking starts with 1 (itself)
        M = {node: 1.0 for node in ranking}
        
        # Scan from last to first
        for i in range(len(ranking) - 1, 0, -1):
            v_ri = ranking[i]  # Current node (rank i)
            
            # Look at all neighbors that are ranked HIGHER (j < i)
            # These are the nodes that can activate v_ri
            for neighbor, p1, p2 in self.reverse_graph[v_ri]:
                if neighbor in pos and pos[neighbor] < i:  # Higher ranked neighbor in ranking
                    p = p1 if campaign_idx == 0 else p2
                    if p > 0:
                        # Add influence from v_ri to neighbor
                        # Simple version: M[neighbor] += p * M[v_ri]
                        # LFA version: account for competition among higher-ranked nodes
                        M[neighbor] += p * M[v_ri]
        
        return M
    
    def compute_balanced_marginal(self, M1: Dict[int, float], M2: Dict[int, float], 
                                   ranking1: List[int], ranking2: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Compute marginal gain for balanced information exposure
        
        For IEM, we need to consider the interaction between two campaigns.
        A node is "good" if it helps make the two exposed sets more similar.
        
        Args:
            M1: Marginal influence for campaign 1
            M2: Marginal influence for campaign 2
            ranking1: Current ranking for campaign 1
            ranking2: Current ranking for campaign 2
            
        Returns:
            Balanced marginal scores for both campaigns
        """
        # For simplicity, we use the original marginal influence
        # A more sophisticated approach would consider the overlap between M1 and M2
        # Nodes that can cover "gaps" in the other campaign are preferred
        
        balanced_M1 = {}
        balanced_M2 = {}
        
        # Create position mappings
        pos1 = {node: i for i, node in enumerate(ranking1)}
        pos2 = {node: i for i, node in enumerate(ranking2)}
        
        for node in range(self.n_nodes):
            # Base score
            score1 = M1[node]
            score2 = M2[node]
            
            # Bonus for nodes that are highly ranked in both campaigns
            # (helps synchronize the two campaigns)
            rank1 = pos1.get(node, len(ranking1))
            rank2 = pos2.get(node, len(ranking2))
            
            # Normalize ranks to [0, 1]
            norm_rank1 = 1 - (rank1 / len(ranking1)) if len(ranking1) > 0 else 0
            norm_rank2 = 1 - (rank2 / len(ranking2)) if len(ranking2) > 0 else 0
            
            # Combined score: balance between individual influence and synchronization
            balanced_M1[node] = score1 * (1 + 0.5 * norm_rank2)
            balanced_M2[node] = score2 * (1 + 0.5 * norm_rank1)
        
        return balanced_M1, balanced_M2
    
    def imrank(self, budget: int, initial_ranking_type: str = "degree") -> Tuple[Set[int], Set[int]]:
        """
        IMRank algorithm for IEM
        
        Args:
            budget: Total budget k (|S1| + |S2| <= k)
            initial_ranking_type: "degree" or "strength"
            
        Returns:
            Tuple of (S1, S2) - balanced seed sets
        """
        # Initialize rankings
        if initial_ranking_type == "degree":
            ranking1 = self.compute_degree_ranking()
            ranking2 = self.compute_degree_ranking()
        else:
            ranking1 = self.compute_strength_ranking()
            ranking2 = self.compute_strength_ranking()
        
        # Exclude initial seeds from consideration
        available_nodes = set(range(self.n_nodes)) - self.I1 - self.I2
        
        # Iteratively improve rankings
        for iteration in range(self.max_iter):
            # Compute marginal influence for both campaigns
            M1 = self.lfa_strategy(ranking1, 0)
            M2 = self.lfa_strategy(ranking2, 1)
            
            # Compute balanced marginal scores
            balanced_M1, balanced_M2 = self.compute_balanced_marginal(M1, M2, ranking1, ranking2)
            
            # Filter to available nodes only
            available_balanced_M1 = {k: v for k, v in balanced_M1.items() if k in available_nodes}
            available_balanced_M2 = {k: v for k, v in balanced_M2.items() if k in available_nodes}
            
            # Create new rankings by sorting
            new_ranking1 = sorted(available_balanced_M1.keys(), 
                                 key=lambda x: available_balanced_M1[x], reverse=True)
            new_ranking2 = sorted(available_balanced_M2.keys(), 
                                 key=lambda x: available_balanced_M2[x], reverse=True)
            
            # Check convergence (top-k nodes unchanged)
            k1 = budget // 2
            k2 = budget - k1
            
            topk1_old = set(ranking1[:k1])
            topk1_new = set(new_ranking1[:k1])
            topk2_old = set(ranking2[:k2])
            topk2_new = set(new_ranking2[:k2])
            
            ranking1 = new_ranking1
            ranking2 = new_ranking2
            
            if topk1_old == topk1_new and topk2_old == topk2_new:
                print(f"IMRank converged at iteration {iteration + 1}")
                break
        
        # Select top nodes as balanced seeds
        k1 = budget // 2
        k2 = budget - k1
        
        self.S1 = set(ranking1[:k1])
        self.S2 = set(ranking2[:k2])
        
        return self.S1, self.S2
    
    def save_balanced_seeds(self, filepath: str) -> None:
        """Save balanced seed sets to file"""
        with open(filepath, 'w') as f:
            f.write(f"{len(self.S1)} {len(self.S2)}\n")
            for node in sorted(self.S1):
                f.write(f"{node}\n")
            for node in sorted(self.S2):
                f.write(f"{node}\n")
    
    def run(self, dataset_path: str, seed_path: str, output_path: str, budget: int) -> None:
        """
        Run IMRank algorithm
        
        Args:
            dataset_path: Path to graph dataset
            seed_path: Path to initial seed file
            output_path: Path to output balanced seed file
            budget: Budget k for balanced seeds
        """
        print(f"Reading graph from {dataset_path}...")
        self.read_graph(dataset_path)
        print(f"Graph: {self.n_nodes} nodes, {self.n_edges} edges")
        
        print(f"Reading initial seeds from {seed_path}...")
        self.read_initial_seeds(seed_path)
        print(f"Initial seeds: |I1|={len(self.I1)}, |I2|={len(self.I2)}")
        
        print(f"Running IMRank with budget k={budget}...")
        S1, S2 = self.imrank(budget)
        print(f"Selected: |S1|={len(S1)}, |S2|={len(S2)}")
        print(f"S1: {sorted(S1)}")
        print(f"S2: {sorted(S2)}")
        
        print(f"Saving results to {output_path}...")
        self.save_balanced_seeds(output_path)
        print("Done!")


def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) < 4:
        print("Usage: python Heuristic.py <dataset_path> <seed_path> <output_path> [budget] [max_iter]")
        print("Example: python Heuristic.py map1/dataset1 map1/seed map1/seed_balanced 10")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    seed_path = sys.argv[2]
    output_path = sys.argv[3]
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    max_iter = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    
    imrank = IMRankIEM(max_iter=max_iter)
    imrank.run(dataset_path, seed_path, output_path, budget)


if __name__ == "__main__":
    main()
