# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the course material repository for **SUSTech CS311H: Artificial Intelligence (Spring 2026)**. It contains lab assignments and Project 1.

- **Labs**: Lab1 through Lab6 directories contain weekly assignments
- **Project 1**: Information Exposure Maximization Problem (IEMP) - located in `Project1/Project1/Project1/`

## Project 1: IEMP (Information Exposure Maximization Problem)

### Problem Summary
Given a social network with two competing campaigns, select additional seed nodes (S1, S2) to maximize balanced information exposure. A node is "balanced exposed" if reached by exactly one campaign (XOR). Maximize E[|V - (r1 XOR r2)|] where r1, r2 are reached node sets.

### Directory Structure

```
Project1/Project1/Project1/
├── Testcase/
│   ├── Heuristic/IEMP_Heur.py          # MC SAA Greedy + IMRank screening
│   ├── Evolutionary/IEMP_Evol.py       # Binary GA with pymoo
│   ├── Evaluator/Evaluator.py          # MC evaluation with objective scoring
│   ├── map1/                           # Small graph (475 nodes, 13289 edges)
│   ├── map2/                           # Large sparse graph (36742 nodes, 49248 edges)
│   ├── map3/                           # Same network as map2, different seeds
│   ├── map4/
│   ├── map5/
│   └── run_all_methods_summary.py      # Unified runner for all methods
├── Ref/                                # Reference papers
├── Project1.pdf                        # Assignment description
└── project1_text.txt                   # Plain text problem description
```

### Input/Output File Formats

**Graph file** (e.g., `map1/dataset1`):
```
n_nodes n_edges
u v p1 p2      # for each edge, p1=prob campaign 1, p2=prob campaign 2
```

**Seed file** (e.g., `map1/seed`):
```
|I1| |I2|
(I1 nodes, one per line)
(I2 nodes, one per line)
```

**Solution file** (output):
```
|S1| |S2|
(S1 nodes, one per line)
(S2 nodes, one per line)
```

### Key CLI Commands

**Run Heuristic**:
```bash
cd Project1/Project1/Project1/Testcase/Heuristic
python IEMP_Heur.py -n map1/dataset1 -i map1/seed -b output.txt -k 10
```

**Run Evolutionary**:
```bash
cd Project1/Project1/Project1/Testcase/Evolutionary
python IEMP_Evol.py -n map2/dataset2 -i map2/seed -b output.txt -k 15
```

**Run Evaluator** (score a solution):
```bash
cd Project1/Project1/Project1/Testcase/Evaluator
python Evaluator.py -n map1/dataset1 -i map1/seed -b solution.txt -k 10 -o score.txt
```

**Run all methods on all maps**:
```bash
cd Project1/Project1/Project1/Testcase
python run_all_methods_summary.py
```

**Run heuristic on all maps**:
```bash
cd Project1/Project1/Project1/Testcase/Heuristic
python run_all_maps.py
```

### Algorithm Parameters

**IEMP_Heur.py**:
- `-k, --budget`: Budget (required)
- `-n, --network`: Graph file path (required)
- `-i, --initial`: Initial seed file path (required)
- `-b, --balanced`: Output balanced seed file path (required)
- `--mc-sim`: MC simulations per step (default: 70)
- `--candidate-size`: IMRank candidate pool size (default: 400, 0 means all nodes)
- `--max-iter`: IMRank self-consistency iterations (default: 250)
- `--seed`: Random seed (optional)

**IEMP_Evol.py**:
- `-k, --budget`: Budget (required)
- `-n, --network`: Graph file path (required)
- `-i, --initial`: Initial seed file path (required)
- `-b, --balanced`: Output balanced seed file path (required)
- `--pop-size`: Population size (default: 50)
- `--generations`: Number of generations (default: 100)
- `--crossover-rate`: Crossover probability (default: 0.8)
- `--mutation-rate`: Mutation rate (default: 0.05)
- `--mc-coarse`: MC simulations during evolution (default: 30)
- `--mc-fine`: MC simulations for final evaluation (default: 200)

### Test Case Specifications

| Map | Nodes | Edges | Avg Degree | Budget | Initial Seeds |
|-----|-------|-------|------------|--------|---------------|
| map1 | 475 | 13,289 | 28.0 | 10 | I1=8, I2=12 |
| map2 | 36,742 | 49,248 | 1.34 | 15 | I1=14, I2=10 |
| map3 | 36,742 | 49,248 | 1.34 | 15 | I1=8, I2=6 (seed2) |
| map4 | varies | varies | - | 25 | varies |
| map5 | 3,454 | 32,140 | 9.3 | 25 | varies |

### Key Implementation Details

**Heuristic Algorithm** (IEMP_Heur.py):
- Uses IMRank self-consistent ranking for candidate screening
- Monte Carlo Sample Average Approximation (SAA) for greedy selection
- Incremental IC diffusion simulation for efficiency
- Candidate pool size adapts based on graph density: `size = max(2*k, 400/(1+(avg_degree/18)^3))`

**Evolutionary Algorithm** (IEMP_Evol.py):
- Binary chromosome encoding: [S1_bits | S2_bits]
- Uses pymoo library for GA pipeline (if available)
- Repair operator ensures budget constraint satisfaction
- Two-phase MC evaluation (coarse during evolution, fine at end)

**Evaluator** (Evaluator.py):
- Monte Carlo simulation with IC model
- Computes objective: |V| - |r1 XOR r2|
- Adaptive simulation count based on graph size and time constraints

### Common Pitfalls

1. **File paths differ by map**:
   - map1 uses `dataset1`, others use `dataset2`, `dataset3`, or `dataset4`
   - map3 uses `seed2` instead of `seed`

2. **Candidate pool sizing**: The current formula uses cubic decay for dense graphs. At avg_degree=1.34 (map2/map3), candidate ≈ 397. At avg_degree=28 (map1), candidate ≈ 33.

3. **Dependencies**: Project uses numpy and (optionally) pymoo for evolutionary algorithm.

### Reference Materials

- `Testcase/AGENTS.md`: Detailed agent guide with algorithm explanations and achieved scores
- `Project1.pdf`: Full problem description with examples
- `Ref/`: Academic papers on IMRank and influence maximization
