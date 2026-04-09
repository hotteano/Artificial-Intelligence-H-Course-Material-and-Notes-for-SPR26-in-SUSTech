"""
IEMP Heuristic Algorithm - MC SAA Greedy
=======================================

This implementation follows the requested algorithmic template:

1. Keep two seed sets S1 and S2 (initially empty).
2. In each budget step, generate N Monte Carlo scenarios based on current
   (I1 U S1) and (I2 U S2).
3. For every candidate vertex v, evaluate h1(v) and h2(v) under the same N
   scenarios using incremental diffusion from the sampled base states.
4. Use average h1/h2 to get v1* and v2*, then add the better option.

Objective is consistent with evaluator:
Phi = |V - (r1 XOR r2)|, where r is the reached/exposed set under IC.
"""

import argparse
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def simulate_ic_with_exposure(
    seeds: Set[int],
    neighbors_by_node: List[np.ndarray],
    probs_by_node: List[np.ndarray],
    n_nodes: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one IC simulation and return (active_set, reached_set)."""
    active = np.zeros(n_nodes, dtype=bool)
    reached = np.zeros(n_nodes, dtype=bool)

    if not seeds:
        return active, reached

    seed_nodes = np.fromiter(seeds, dtype=np.int32, count=len(seeds))
    active[seed_nodes] = True
    reached[seed_nodes] = True
    frontier = seed_nodes.tolist()

    while frontier:
        next_frontier: List[int] = []
        for node in frontier:
            neighbors = neighbors_by_node[node]
            if neighbors.size == 0:
                continue

            inactive_mask = ~active[neighbors]
            if not inactive_mask.any():
                continue

            attempted_nodes = neighbors[inactive_mask]
            attempted_probs = probs_by_node[node][inactive_mask]

            unseen_mask = ~reached[attempted_nodes]
            if unseen_mask.any():
                reached[attempted_nodes[unseen_mask]] = True

            success_mask = rng.random(attempted_nodes.size) < attempted_probs
            if not success_mask.any():
                continue

            for v in attempted_nodes[success_mask]:
                if not active[v]:
                    active[v] = True
                    next_frontier.append(int(v))

        frontier = next_frontier

    return active, reached


def simulate_incremental_ic(
    start_node: int,
    base_active: np.ndarray,
    base_reached: np.ndarray,
    neighbors_by_node: List[np.ndarray],
    probs_by_node: List[np.ndarray],
    rng: np.random.Generator,
) -> List[int]:
    """Run incremental IC from one added seed on top of sampled base state.

    Returns newly reached node IDs compared with base_reached.
    """
    if base_active[start_node]:
        return []

    active = base_active.copy()
    reached = base_reached.copy()
    incremental_reached: List[int] = []

    active[start_node] = True
    frontier = [start_node]

    if not reached[start_node]:
        reached[start_node] = True
        incremental_reached.append(start_node)

    while frontier:
        next_frontier: List[int] = []
        for node in frontier:
            neighbors = neighbors_by_node[node]
            if neighbors.size == 0:
                continue

            inactive_mask = ~active[neighbors]
            if not inactive_mask.any():
                continue

            attempted_nodes = neighbors[inactive_mask]
            attempted_probs = probs_by_node[node][inactive_mask]

            unseen_mask = ~reached[attempted_nodes]
            if unseen_mask.any():
                newly_seen = attempted_nodes[unseen_mask]
                reached[newly_seen] = True
                incremental_reached.extend(newly_seen.tolist())

            success_mask = rng.random(attempted_nodes.size) < attempted_probs
            if not success_mask.any():
                continue

            for v in attempted_nodes[success_mask]:
                if not active[v]:
                    active[v] = True
                    next_frontier.append(int(v))

        frontier = next_frontier

    return incremental_reached


class IEMData:
    """Data class for IEM problem."""

    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph: Dict[int, List[Tuple[int, float, float]]] = {}
        self.reverse_graph: Dict[int, List[Tuple[int, float, float]]] = {}
        self.I1: Set[int] = set()
        self.I2: Set[int] = set()
        self.S1: Set[int] = set()
        self.S2: Set[int] = set()

    def load_graph(self, filepath: str):
        """Load graph from file."""
        self.graph = {}
        self.reverse_graph = {}

        with open(filepath, "r") as f:
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

    def load_initial_seeds(self, filepath: str):
        """Load initial seed sets."""
        with open(filepath, "r") as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))

    def check_high_probability(self, threshold: float = 0.99) -> bool:
        """Check if all edge probabilities equal or exceed a high threshold."""
        for u in self.graph:
            for v, p1, p2 in self.graph[u]:
                if p1 < threshold or p2 < threshold:
                    return False
        return True

    def save_solution(self, filepath: str):
        """Save solution to file."""
        with open(filepath, "w") as f:
            f.write(f"{len(self.S1)} {len(self.S2)}\n")
            for node in sorted(self.S1):
                f.write(f"{node}\n")
            for node in sorted(self.S2):
                f.write(f"{node}\n")


class IEMPMCHeuristic:
    """Monte Carlo SAA greedy heuristic with optional IMRank screening."""

    def __init__(
        self,
        data: IEMData,
        budget: int = 10,
        max_iter: int = 20,
        mc_simulations: int = 100,
        candidate_pool_size: int = 0,
        seed: Optional[int] = None,
        workers: int = 1,
    ):
        self.data = data
        self.budget = budget
        self.max_iter = max_iter
        self.mc_simulations = mc_simulations
        self.candidate_pool_size = candidate_pool_size
        self.workers = workers
        self.rng = np.random.default_rng(seed)

        if self.data.check_high_probability(0.95):
            print("[Warning] High probability graph detected (p>=0.95). Reducing MC simulations to 3.")
            self.mc_simulations = 3

        # Cache adjacency as NumPy arrays for faster repeated simulation.
        self._mc_neighbors: List[np.ndarray] = []
        self._mc_prob1: List[np.ndarray] = []
        self._mc_prob2: List[np.ndarray] = []

        for node in range(self.data.n_nodes):
            edges = self.data.graph[node]
            if not edges:
                self._mc_neighbors.append(np.empty(0, dtype=np.int32))
                self._mc_prob1.append(np.empty(0, dtype=np.float64))
                self._mc_prob2.append(np.empty(0, dtype=np.float64))
                continue

            self._mc_neighbors.append(
                np.fromiter((v for v, _, _ in edges), dtype=np.int32, count=len(edges))
            )
            self._mc_prob1.append(
                np.fromiter((p1 for _, p1, _ in edges), dtype=np.float64, count=len(edges))
            )
            self._mc_prob2.append(
                np.fromiter((p2 for _, _, p2 in edges), dtype=np.float64, count=len(edges))
            )

    def compute_weighted_degree_ranking(self, campaign_idx: int) -> List[int]:
        """Initialize ranking by weighted out-degree for one campaign."""
        weighted_degrees: List[Tuple[int, float, int]] = []
        for node in range(self.data.n_nodes):
            score = sum(
                p1 if campaign_idx == 0 else p2
                for _, p1, p2 in self.data.graph[node]
            )
            weighted_degrees.append((node, score, len(self.data.graph[node])))

        weighted_degrees.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)
        return [node for node, _, _ in weighted_degrees]

    def lfa_strategy(self, ranking: List[int], campaign_idx: int) -> np.ndarray:
        """LFA pass for one ranking, used in IMRank self-consistency."""
        if not ranking:
            return np.zeros(self.data.n_nodes, dtype=np.float64)

        n_nodes = self.data.n_nodes
        pos = np.full(n_nodes, -1, dtype=np.int32)
        pos[ranking] = np.arange(len(ranking), dtype=np.int32)

        M = np.zeros(n_nodes, dtype=np.float64)
        M[ranking] = 1.0

        for i in range(len(ranking) - 1, 0, -1):
            v = ranking[i]
            remaining = M[v]
            if remaining <= 1e-15:
                continue

            higher_parents: List[Tuple[int, int, float]] = []
            for parent, p1, p2 in self.data.reverse_graph.get(v, []):
                parent_pos = pos[parent]
                if parent_pos != -1 and parent_pos < i:
                    p = p1 if campaign_idx == 0 else p2
                    higher_parents.append((parent_pos, parent, p))

            higher_parents.sort(key=lambda x: x[0])
            for _, parent, p in higher_parents:
                influence = remaining * p
                if influence <= 0:
                    continue
                M[parent] += influence
                remaining *= (1 - p)
                if remaining <= 1e-12:
                    break

        return M

    def imrank_self_consistent(self, campaign_idx: int) -> List[int]:
        """Compute IMRank self-consistent ranking for one campaign."""
        ranking = self.compute_weighted_degree_ranking(campaign_idx)
        if not ranking:
            return []

        last_ranking: Optional[List[int]] = None
        for _ in range(self.max_iter):
            M = self.lfa_strategy(ranking, campaign_idx)
            new_ranking = sorted(
                ranking,
                key=lambda node: (M[node], len(self.data.graph[node]), -node),
                reverse=True,
            )
            if new_ranking == ranking or new_ranking == last_ranking:
                ranking = new_ranking
                break
            last_ranking = ranking
            ranking = new_ranking

        return ranking

    def build_candidate_pool(
        self,
        available: Set[int],
        rank1: List[int],
        rank2: List[int],
    ) -> List[int]:
        """Build candidate pool from top IMRank nodes of two campaigns."""
        if self.candidate_pool_size <= 0 or self.candidate_pool_size >= len(available):
            return sorted(available)

        limit_total = min(self.candidate_pool_size, len(available))
        limit_each = max(1, limit_total // 2)

        pool: List[int] = []
        seen: Set[int] = set()

        for node in rank1:
            if node in available and node not in seen:
                pool.append(node)
                seen.add(node)
            if len(pool) >= limit_each:
                break

        for node in rank2:
            if node in available and node not in seen:
                pool.append(node)
                seen.add(node)
            if len(pool) >= limit_total:
                break

        if len(pool) < limit_total:
            for node in sorted(available):
                if node not in seen:
                    pool.append(node)
                    if len(pool) >= limit_total:
                        break

        return pool

    @staticmethod
    def _delta_from_increment(
        incremental_reached: List[int],
        opposite_reached: np.ndarray,
    ) -> int:
        """Compute Phi gain from newly reached nodes on one side."""
        if not incremental_reached:
            return 0

        nodes = np.asarray(incremental_reached, dtype=np.int32)
        overlap = int(np.count_nonzero(opposite_reached[nodes]))
        return (2 * overlap) - int(nodes.size)

    def mc_simulation(self, full_seeds_1: Set[int], full_seeds_2: Set[int]) -> int:
        """Single IC simulation scored by balanced exposure objective."""
        _, reached1 = simulate_ic_with_exposure(
            full_seeds_1,
            self._mc_neighbors,
            self._mc_prob1,
            self.data.n_nodes,
            self.rng,
        )
        _, reached2 = simulate_ic_with_exposure(
            full_seeds_2,
            self._mc_neighbors,
            self._mc_prob2,
            self.data.n_nodes,
            self.rng,
        )
        symmetric_diff_size = int(np.count_nonzero(np.logical_xor(reached1, reached2)))
        return self.data.n_nodes - symmetric_diff_size

    def mc_evaluate(self, seeds1: Set[int], seeds2: Set[int]) -> float:
        """Evaluate final solution by Monte Carlo."""
        full_seeds_1 = self.data.I1 | seeds1
        full_seeds_2 = self.data.I2 | seeds2

        total = 0
        for _ in range(self.mc_simulations):
            total += self.mc_simulation(full_seeds_1, full_seeds_2)
        return total / self.mc_simulations

    def run(self) -> Tuple[Set[int], Set[int]]:
        """Main algorithm: MC SAA greedy with IMRank candidate screening."""
        available = set(range(self.data.n_nodes)) - self.data.I1 - self.data.I2
        S1: Set[int] = set()
        S2: Set[int] = set()

        if not available or self.budget <= 0:
            self.data.S1, self.data.S2 = S1, S2
            return S1, S2

        use_imrank = 0 < self.candidate_pool_size < len(available)
        rank1: List[int] = []
        rank2: List[int] = []

        if use_imrank:
            rank1 = self.imrank_self_consistent(campaign_idx=0)
            rank2 = self.imrank_self_consistent(campaign_idx=1)

        steps = min(self.budget, len(available))

        for step in range(steps):
            if use_imrank:
                valid_candidates = self.build_candidate_pool(available, rank1, rank2)
            else:
                valid_candidates = sorted(available)

            if not valid_candidates:
                break

            base_a1_list: List[np.ndarray] = []
            base_r1_list: List[np.ndarray] = []
            base_a2_list: List[np.ndarray] = []
            base_r2_list: List[np.ndarray] = []

            seeds_1 = self.data.I1 | S1
            seeds_2 = self.data.I2 | S2

            for _ in range(self.mc_simulations):
                a1, r1 = simulate_ic_with_exposure(
                    seeds_1,
                    self._mc_neighbors,
                    self._mc_prob1,
                    self.data.n_nodes,
                    self.rng,
                )
                a2, r2 = simulate_ic_with_exposure(
                    seeds_2,
                    self._mc_neighbors,
                    self._mc_prob2,
                    self.data.n_nodes,
                    self.rng,
                )
                base_a1_list.append(a1)
                base_r1_list.append(r1)
                base_a2_list.append(a2)
                base_r2_list.append(r2)

            best_h1 = -float("inf")
            best_h2 = -float("inf")
            best_v1: Optional[int] = None
            best_v2: Optional[int] = None

            for node in valid_candidates:
                h1_total = 0.0
                h2_total = 0.0

                for j in range(self.mc_simulations):
                    inc_r1 = simulate_incremental_ic(
                        node,
                        base_a1_list[j],
                        base_r1_list[j],
                        self._mc_neighbors,
                        self._mc_prob1,
                        self.rng,
                    )
                    h1_total += self._delta_from_increment(inc_r1, base_r2_list[j])

                    inc_r2 = simulate_incremental_ic(
                        node,
                        base_a2_list[j],
                        base_r2_list[j],
                        self._mc_neighbors,
                        self._mc_prob2,
                        self.rng,
                    )
                    h2_total += self._delta_from_increment(inc_r2, base_r1_list[j])

                h1_avg = h1_total / self.mc_simulations
                h2_avg = h2_total / self.mc_simulations

                if h1_avg > best_h1 or (h1_avg == best_h1 and (best_v1 is None or node < best_v1)):
                    best_h1 = h1_avg
                    best_v1 = node

                if h2_avg > best_h2 or (h2_avg == best_h2 and (best_v2 is None or node < best_v2)):
                    best_h2 = h2_avg
                    best_v2 = node

            if best_v1 is None and best_v2 is None:
                break

            choose_s1 = False
            if best_v2 is None:
                choose_s1 = True
            elif best_v1 is None:
                choose_s1 = False
            elif best_h1 > best_h2:
                choose_s1 = True
            elif best_h2 > best_h1:
                choose_s1 = False
            else:
                if len(S1) < len(S2):
                    choose_s1 = True
                elif len(S2) < len(S1):
                    choose_s1 = False
                else:
                    choose_s1 = best_v1 < best_v2

            if choose_s1:
                assert best_v1 is not None
                S1.add(best_v1)
                available.remove(best_v1)
            else:
                assert best_v2 is not None
                S2.add(best_v2)
                available.remove(best_v2)

        final_score = self.mc_evaluate(S1, S2)

        self.data.S1, self.data.S2 = S1, S2
        return S1, S2


def main():
    parser = argparse.ArgumentParser(description="IEMP MC SAA Greedy Heuristic")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")
    parser.add_argument(
        "--mc-sim",
        type=int,
        default=70,
        help="MC scenarios per step and final evaluation (default: 70)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum IMRank self-consistency iterations (default: 200)",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=500,
        help="IMRank candidate pool size; <=0 means evaluate all vertices (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.budget <= 0:
        raise ValueError("Budget k must be a positive integer.")
    if args.mc_sim <= 0:
        raise ValueError("MC simulations must be a positive integer.")
    if not os.path.exists(args.network):
        raise FileNotFoundError(f"Network file not found: {args.network}")
    if not os.path.exists(args.initial):
        raise FileNotFoundError(f"Initial seed file not found: {args.initial}")

    data = IEMData()
    data.load_graph(args.network)
    data.load_initial_seeds(args.initial)

    pass

    heuristic = IEMPMCHeuristic(
        data,
        budget=args.budget,
        max_iter=args.max_iter,
        mc_simulations=args.mc_sim,
        candidate_pool_size=args.candidate_size,
        seed=args.seed,
    )
    heuristic.run()

    data.save_solution(args.balanced)


if __name__ == "__main__":
    main()
