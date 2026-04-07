"""
IEMP Evolutionary Algorithm - pymoo 0.6.0.1 Implementation
===========================================================

This version upgrades the evolutionary solver with pymoo's GA pipeline:

1. Binary chromosome encoding x = [S1 bits | S2 bits], length = 2 * n_available.
2. Custom repair keeps solutions feasible and sparse-search friendly.
3. pymoo GA handles population evolution, crossover, mutation, and selection.
4. Fitness is still Monte Carlo balanced exposure (same evaluator objective).
5. Optional simulated annealing refines the best GA chromosome locally.

Primary objective:
maximize E[ |V - (r1 XOR r2)| ]
"""

import argparse
import os
import random
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.callback import Callback
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.repair import Repair
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


class IEMData:
    """Data class for IEM problem."""

    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph: Dict[int, List[Tuple[int, float, float]]] = {}
        self.I1: Set[int] = set()
        self.I2: Set[int] = set()
        self.available_nodes: List[int] = []
        self.node_to_idx: Dict[int, int] = {}

    def load_graph(self, filepath: str):
        """Load graph from file."""
        self.graph = {}

        with open(filepath, "r") as f:
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
        """Load initial seed sets and identify available nodes."""
        with open(filepath, "r") as f:
            n1, n2 = map(int, f.readline().strip().split())
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))

        self.available_nodes = [
            node for node in range(self.n_nodes) if node not in self.I1 and node not in self.I2
        ]
        self.node_to_idx = {node: idx for idx, node in enumerate(self.available_nodes)}

    def save_solution(self, S1: Set[int], S2: Set[int], filepath: str):
        """Save solution to file."""
        with open(filepath, "w") as f:
            f.write(f"{len(S1)} {len(S2)}\n")
            for node in sorted(S1):
                f.write(f"{node}\n")
            for node in sorted(S2):
                f.write(f"{node}\n")


class IEMPEvaluator:
    """Monte Carlo evaluator for fitness computation."""

    def __init__(self, data: IEMData):
        self.data = data
        self.cache: Dict[Tuple[frozenset, frozenset, int], float] = {}

    def clear_cache(self):
        self.cache = {}

    def evaluate(self, S1: Set[int], S2: Set[int], n_simulations: int) -> float:
        """Evaluate solution using MC simulation."""
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
        """Single IC simulation."""
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
    """pymoo-based evolutionary optimization for IEMP."""

    def __init__(
        self,
        data: IEMData,
        budget: int = 10,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        elitism: int = 2,
        mc_coarse: int = 30,
        mc_fine: int = 200,
        use_sa: bool = True,
        sa_steps: int = 200,
        sa_t0: float = 1.0,
        sa_alpha: float = 0.985,
        seed: int | None = None,
    ):
        self.data = data
        self.budget = budget
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.mc_coarse = mc_coarse
        self.mc_fine = mc_fine
        self.use_sa = use_sa
        self.sa_steps = sa_steps
        self.sa_t0 = sa_t0
        self.sa_alpha = sa_alpha
        self.seed = seed

        self.n_available = len(data.available_nodes)
        self.chromosome_length = 2 * self.n_available

        self.evaluator = IEMPEvaluator(data)
        self.rng = np.random.default_rng(seed)

    def decode_chromosome(self, chromosome: np.ndarray) -> Tuple[Set[int], Set[int]]:
        """Decode binary chromosome to S1 and S2."""
        x = np.asarray(chromosome).astype(np.int8)
        x = (x > 0).astype(np.int8)

        n = self.n_available
        s1_idx = np.flatnonzero(x[:n])
        s2_idx = np.flatnonzero(x[n:])

        S1 = {self.data.available_nodes[i] for i in s1_idx}
        S2 = {self.data.available_nodes[i] for i in s2_idx}
        return S1, S2

    def repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """Repair to keep chromosomes feasible and budget-aware."""
        x = np.asarray(chromosome).astype(np.int8).copy()
        x = (x > 0).astype(np.int8)

        if self.n_available == 0:
            return x

        n = self.n_available

        # Resolve same-node dual assignment conflicts first.
        both_selected = np.where((x[:n] == 1) & (x[n:] == 1))[0]
        for idx in both_selected:
            if self.rng.random() < 0.5:
                x[idx] = 0
            else:
                x[n + idx] = 0

        target_budget = min(self.budget, self.n_available)
        current = int(np.count_nonzero(x))

        if current > target_budget:
            one_positions = np.flatnonzero(x == 1)
            remove_count = current - target_budget
            if remove_count > 0 and one_positions.size > 0:
                remove_pos = self.rng.choice(one_positions, size=remove_count, replace=False)
                x[remove_pos] = 0

        elif current < target_budget:
            need = target_budget - current
            free_nodes = np.where((x[:n] == 0) & (x[n:] == 0))[0]
            if free_nodes.size > 0:
                pick_size = min(need, free_nodes.size)
                picked_nodes = self.rng.choice(free_nodes, size=pick_size, replace=False)
                side_choices = self.rng.integers(0, 2, size=pick_size)
                for node_idx, side in zip(picked_nodes, side_choices):
                    if side == 0:
                        x[node_idx] = 1
                    else:
                        x[n + node_idx] = 1

        return x

    def coarse_fitness(self, chromosome: np.ndarray) -> float:
        """Compute coarse fitness for one chromosome."""
        repaired = self.repair_chromosome(chromosome)
        S1, S2 = self.decode_chromosome(repaired)

        total_size = len(S1) + len(S2)

        score = self.evaluator.evaluate(S1, S2, self.mc_coarse)

        # Small budget-usage bonus to break ties toward fuller usage.
        usage_ratio = total_size / self.budget if self.budget > 0 else 0.0
        return score + 20.0 * usage_ratio

    def _opposite_position(self, pos: int) -> int:
        """Return the mirrored campaign-bit position for the same available-node index."""
        return pos + self.n_available if pos < self.n_available else pos - self.n_available

    def _random_neighbor(self, chromosome: np.ndarray) -> np.ndarray:
        """Generate one feasible neighbor with fixed budget usage."""
        x = self.repair_chromosome(chromosome)

        if self.n_available == 0:
            return x

        ones = np.flatnonzero(x == 1)
        if ones.size == 0:
            return x

        # Move type A: switch one selected node to the opposite campaign.
        if self.rng.random() < 0.35:
            pos = int(self.rng.choice(ones))
            opp = self._opposite_position(pos)
            if x[opp] == 0:
                x[pos] = 0
                x[opp] = 1
                return self.repair_chromosome(x)

        # Move type B: replace one selected bit with another feasible zero bit.
        pos_off = int(self.rng.choice(ones))
        x[pos_off] = 0

        zero_positions = np.flatnonzero(x == 0)
        candidates = [
            int(p)
            for p in zero_positions
            if p != pos_off and x[self._opposite_position(int(p))] == 0
        ]

        if not candidates:
            x[pos_off] = 1
            return self.repair_chromosome(x)

        pos_on = int(self.rng.choice(candidates))
        x[pos_on] = 1
        return self.repair_chromosome(x)

    def simulated_annealing_refine(self, start: np.ndarray) -> Tuple[np.ndarray, float]:
        """Refine a GA solution using simulated annealing on coarse fitness."""
        current = self.repair_chromosome(start)
        current_f = self.coarse_fitness(current)

        best = current.copy()
        best_f = current_f

        temperature = max(float(self.sa_t0), 1e-12)

        for _ in range(self.sa_steps):
            candidate = self._random_neighbor(current)
            candidate_f = self.coarse_fitness(candidate)

            delta = candidate_f - current_f
            if delta >= 0:
                accept = True
            else:
                accept_prob = np.exp(delta / max(temperature, 1e-12))
                accept = self.rng.random() < accept_prob

            if accept:
                current = candidate
                current_f = candidate_f
                if current_f > best_f:
                    best = current.copy()
                    best_f = current_f

            temperature *= self.sa_alpha

        return best, best_f

    def evolve(self) -> Tuple[Set[int], Set[int]]:
        """Run pymoo GA and return best solution."""
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo is required. Please install pymoo==0.6.0.1 in the active Python environment."
            )

        if self.n_available == 0:
            print("No available nodes to optimize.")
            return set(), set()

        print("\n" + "=" * 60)
        print("pymoo Binary GA for IEMP")
        print("=" * 60)
        print("Parameters:")
        print(f"  Population: {self.population_size}")
        print(f"  Generations: {self.generations}")
        print(f"  Budget: {self.budget}")
        print(f"  Available nodes: {self.n_available}")
        print(f"  Chromosome length: {self.chromosome_length}")
        print(f"  Crossover rate: {self.crossover_rate}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  Elitism parameter (legacy): {self.elitism}")
        print(f"  MC (coarse): {self.mc_coarse}, MC (fine): {self.mc_fine}")
        print(
            f"  Simulated Annealing: {'ON' if self.use_sa else 'OFF'}"
            f" (steps={self.sa_steps}, T0={self.sa_t0}, alpha={self.sa_alpha})"
        )

        class _IEMPRepair(Repair):
            def __init__(self, solver: "IEMPEvolutionary"):
                super().__init__()
                self.solver = solver

            def _do(self, problem, X, **kwargs):
                X_arr = np.asarray(X).copy()
                if X_arr.ndim == 1:
                    X_arr = X_arr[None, :]
                for i in range(X_arr.shape[0]):
                    X_arr[i] = self.solver.repair_chromosome(X_arr[i])
                return X_arr

        class _IEMPProblem(ElementwiseProblem):
            def __init__(self, solver: "IEMPEvolutionary"):
                self.solver = solver
                super().__init__(
                    n_var=solver.chromosome_length,
                    n_obj=1,
                    n_ieq_constr=2,
                    xl=0,
                    xu=1,
                    vtype=bool,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                repaired = self.solver.repair_chromosome(x)
                S1, S2 = self.solver.decode_chromosome(repaired)
                total_size = len(S1) + len(S2)
                overlap = len(S1 & S2)

                fitness = self.solver.coarse_fitness(repaired)
                out["F"] = -fitness
                # Constraints: G <= 0 is feasible
                # G[0]: total_size <= budget
                # G[1]: overlap <= 0 (S1 and S2 must be disjoint)
                out["G"] = np.array([
                    total_size - self.solver.budget,
                    float(overlap),
                ])

        class _Progress(Callback):
            def __init__(self):
                super().__init__()

            def notify(self, algorithm):
                gen = algorithm.n_gen
                if gen == 1 or gen % 20 == 0:
                    F = algorithm.pop.get("F")
                    best = float(np.min(F))
                    print(f"Gen {gen}: Best coarse fitness = {-best:.4f}")

        problem = _IEMPProblem(self)
        repair = _IEMPRepair(self)

        algorithm = GA(
            pop_size=self.population_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=self.crossover_rate),
            mutation=BitflipMutation(prob=self.mutation_rate),
            repair=repair,
            eliminate_duplicates=True,
        )

        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", self.generations),
            seed=self.seed,
            callback=_Progress(),
            verbose=False,
            save_history=False,
        )

        if result.X is not None:
            best_x = result.X
        else:
            pop_x = result.pop.get("X")
            pop_f = result.pop.get("F")
            best_idx = int(np.argmin(pop_f))
            best_x = pop_x[best_idx]

        best_x = self.repair_chromosome(best_x)

        if self.use_sa and self.sa_steps > 0:
            print("\n" + "=" * 60)
            print("Simulated Annealing Refinement")
            print("=" * 60)
            ga_coarse = self.coarse_fitness(best_x)
            sa_x, sa_coarse = self.simulated_annealing_refine(best_x)

            if sa_coarse > ga_coarse:
                print(f"SA improved coarse fitness: {ga_coarse:.4f} -> {sa_coarse:.4f}")
                best_x = sa_x
            else:
                print(f"SA kept GA solution (best coarse: {ga_coarse:.4f})")

        S1, S2 = self.decode_chromosome(best_x)

        print("\n" + "=" * 60)
        print("Final Fine Evaluation")
        print("=" * 60)
        self.evaluator.clear_cache()

        coarse_score = self.evaluator.evaluate(S1, S2, self.mc_coarse)
        fine_score = self.evaluator.evaluate(S1, S2, self.mc_fine)

        print(f"Final Coarse Fitness (score only): {coarse_score:.4f}")
        print(f"Final Fine Fitness (score only): {fine_score:.4f}")
        print(f"S1 ({len(S1)} nodes): {sorted(S1)}")
        print(f"S2 ({len(S2)} nodes): {sorted(S2)}")
        print(f"Total: {len(S1) + len(S2)} / {self.budget}")
        print("=" * 60)

        return S1, S2


def main():
    parser = argparse.ArgumentParser(description="IEM Evolutionary Algorithm (pymoo binary GA)")
    parser.add_argument("-n", "--network", required=True, help="Path to social network file")
    parser.add_argument("-i", "--initial", required=True, help="Path to initial seed set file")
    parser.add_argument("-b", "--balanced", required=True, help="Path to output balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, required=True, help="Budget k")

    parser.add_argument("--pop-size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--generations", type=int, default=120, help="Number of generations (default: 100)")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate (default: 0.8)")
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.05,
        help="Bit-flip mutation rate (default: 0.07)",
    )
    parser.add_argument("--elitism", type=int, default=2, help="Legacy parameter kept for compatibility")
    parser.add_argument(
        "--mc-coarse",
        type=int,
        default=90,
        help="MC simulations for evolution (default: 80)",
    )
    parser.add_argument(
        "--mc-fine",
        type=int,
        default=100,
        help="MC simulations for final eval (default: 100)",
    )
    parser.add_argument(
        "--no-sa",
        action="store_true",
        help="Disable simulated annealing refinement after GA",
    )
    parser.add_argument(
        "--sa-steps",
        type=int,
        default=200,
        help="Simulated annealing steps (default: 200)",
    )
    parser.add_argument(
        "--sa-t0",
        type=float,
        default=1.0,
        help="Initial temperature for simulated annealing (default: 1.0)",
    )
    parser.add_argument(
        "--sa-alpha",
        type=float,
        default=0.985,
        help="Temperature decay factor for simulated annealing (default: 0.985)",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed (default: 3407)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.budget <= 0:
        raise ValueError("Budget k must be a positive integer.")
    if args.mc_coarse <= 0 or args.mc_fine <= 0:
        raise ValueError("mc-coarse and mc-fine must be positive integers.")
    if args.sa_steps < 0:
        raise ValueError("sa-steps must be non-negative.")
    if args.sa_t0 <= 0:
        raise ValueError("sa-t0 must be positive.")
    if not (0 < args.sa_alpha < 1):
        raise ValueError("sa-alpha must be in (0, 1).")
    if not os.path.exists(args.network):
        raise FileNotFoundError(f"Network file not found: {args.network}")
    if not os.path.exists(args.initial):
        raise FileNotFoundError(f"Initial seed file not found: {args.initial}")

    data = IEMData()
    data.load_graph(args.network)
    data.load_initial_seeds(args.initial)

    print(f"Graph: {data.n_nodes} nodes, {data.n_edges} edges")
    print(f"Initial seeds: I1={len(data.I1)}, I2={len(data.I2)}")
    print(f"Available nodes: {len(data.available_nodes)}")
    print(f"Budget k: {args.budget}")

    if args.budget > len(data.available_nodes):
        print(
            "Warning: budget is larger than available nodes; with disjoint assignment, "
            "effective maximum is number of available nodes."
        )

    solver = IEMPEvolutionary(
        data=data,
        budget=args.budget,
        population_size=args.pop_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        mc_coarse=args.mc_coarse,
        mc_fine=args.mc_fine,
        use_sa=not args.no_sa,
        sa_steps=args.sa_steps,
        sa_t0=args.sa_t0,
        sa_alpha=args.sa_alpha,
        seed=args.seed,
    )

    S1, S2 = solver.evolve()
    data.save_solution(S1, S2, args.balanced)
    print(f"\nSolution saved to: {args.balanced}")


if __name__ == "__main__":
    main()
