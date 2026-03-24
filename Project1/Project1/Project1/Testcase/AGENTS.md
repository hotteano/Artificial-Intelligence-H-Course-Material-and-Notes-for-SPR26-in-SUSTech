# IEMP (Information Exposure Maximization Problem) - Agent Guide

## Problem Description

### Objective
Given a social network with two competing information campaigns (Campaign 1 and Campaign 2), select additional seed nodes (S1 for Campaign 1, S2 for Campaign 2) such that the **Balanced Information Exposure** is maximized.

### Balanced Information Exposure
A node is informed by a campaign if it is either:
- A seed node of that campaign, OR
- Activated through influence diffusion from seed nodes

The Balanced Information Exposure counts nodes that are informed by **exactly one** campaign (not both, not neither):
```
Balanced Exposure = |{v : v ∈ R1 XOR v ∈ R2}|
```
where R1 and R2 are the sets of nodes reached by Campaign 1 and Campaign 2 respectively.

### Constraints
- |S1| + |S2| ≤ k (budget constraint)
- S1 ∩ S2 = ∅ (mutually exclusive seeds)
- S1 ∩ I1 = ∅, S2 ∩ I2 = ∅ (cannot select initial seeds)

### Influence Diffusion Model
- **Independent Cascade (IC) Model**
- Each edge (u, v) has two probabilities: p1 (for Campaign 1), p2 (for Campaign 2)
- When node u is activated, it attempts to activate neighbor v with probability p1 or p2

---

## Test Cases & Scoring Standards

### Case 0 (Small Graph)
- **Graph**: map1, 475 nodes, 13,289 edges
- **Budget**: k = 10
- **Initial Seeds**: I1 = 8 nodes, I2 = 12 nodes
- **Baseline**: 430
- **Higher**: 450
- **Status**: ✅ Achieved 445.42 with MC-Guided Heuristic

### Case 1 (Large Graph - Variant 1)
- **Graph**: map2, 36,742 nodes, 49,248 edges
- **Budget**: k = 15
- **Initial Seeds**: I1 = 14 nodes, I2 = 10 nodes
- **Baseline**: 35,900
- **Higher**: 36,035
- **Status**: ✅ Achieved 35,934.41 with MC-Guided Heuristic

### Case 2 (Large Graph - Variant 2)
- **Graph**: map3 (same network as map2), 36,742 nodes, 49,248 edges
- **Budget**: k = 15
- **Initial Seeds**: I1 = 8 nodes, I2 = 6 nodes (different from Case 1)
- **Baseline**: 36,000
- **Higher**: 36,200
- **Status**: ✅ Achieved 36,037.76 with MC-Guided Heuristic

---

## Algorithms

### 1. Heuristic Algorithm: MC-Guided Heuristic
**File**: `Heuristic/IEMP_Heur.py`

**Core Idea**: Use IMRank for candidate screening + Monte Carlo for precise evaluation

**Key Steps**:
1. **IMRank Self-Consistent Ranking**: Generate high-quality candidate pool (fast O(E) screening)
2. **MC-Guided Greedy Selection**: At each step, evaluate adding candidate to S1 or S2 using MC simulation
3. **Budget Satisfaction**: Stop when budget is exhausted

**Key Parameters**:
- `--mc-sim`: MC simulations per evaluation (default: 50)
- `--candidate-size`: IMRank candidate pool size (default: 100)
- `--max-iter`: IMRank max iterations (default: 20)

**Why It Works**:
- Pure IMRank with LFA estimation has ~1.5-2% error on large graphs
- MC simulation corrects this bias while maintaining heuristic nature
- Candidate pool reduces MC evaluations from O(n) to O(candidate_size)

**Usage**:
```bash
cd Heuristic
python IEMP_Heur.py -n .\map1\dataset1 -i .\map1\seed -b output.txt -k 10
```

---

### 2. Evolutionary Algorithm: Binary GA with MC Evaluation
**File**: `Evolutionary/IEMP_Evol.py`

**Core Idea**: Population-based search with binary encoding and MC fitness evaluation

**Key Features**:
1. **Binary Encoding**: Chromosome = [S1_bits | S2_bits], length = 2 × n_available
2. **Constraint Handling**: Penalty-based fitness (infeasible solutions get negative fitness)
3. **Genetic Operators**: Uniform crossover + bit-flip mutation
4. **Selection**: Tournament selection with elitism
5. **Two-phase MC**: Coarse evaluation (30 sims) during evolution, fine evaluation (200 sims) at end

**Key Parameters**:
- `--pop-size`: Population size (default: 50)
- `--generations`: Number of generations (default: 100)
- `--crossover-rate`: Crossover probability (default: 0.8)
- `--mutation-rate`: Bit-flip mutation rate (default: 0.05)
- `--mc-coarse`: MC simulations during evolution (default: 30)
- `--mc-fine`: MC simulations for final evaluation (default: 200)

**Why It's Different from Heuristic**:
- Heuristic: Constructive greedy algorithm (sequential node selection)
- Evolutionary: Population-based search with recombination and mutation

**Usage**:
```bash
cd Evolutionary
python IEMP_Evol.py -n .\map2\dataset2 -i .\map2\seed -b output.txt -k 15
```

---

## Directory Structure

```
Testcase/
├── Heuristic/
│   └── IEMP_Heur.py          # MC-Guided Heuristic (Optimal)
├── Evolutionary/
│   └── IEMP_Evol.py          # Binary GA with MC Evaluation
├── Evaluator/
│   └── Evaluator.py          # MC Evaluation with adaptive sim count
├── map1/                     # Case 0 data
├── map2/                     # Case 1 data
├── map3/                     # Case 2 data (same network as map2)
├── Results_Case012/          # Output directory (cleaned)
│   ├── Heuristic/
│   ├── Evolutionary/
│   └── Evaluator/
└── AGENTS.md                 # This file
```

---

## Evaluation

### Evaluator Script
**File**: `Evaluator/Evaluator.py`

Evaluates a solution using Monte Carlo simulation with:
- Adaptive simulation count based on time limit
- Default: 1,000 simulations for accurate evaluation
- Progress tracking with tqdm

**Usage**:
```bash
cd Evaluator
python Evaluator.py -n .\map1\dataset1 -i .\map1\seed -b ..\Results_Case012\Heuristic\output.txt -k 10
```

### Baseline Scoring
To achieve baseline score:
- **Heuristic**: Use MC-Guided Heuristic with default parameters
- **Case 0**: Should achieve ~445 (Baseline: 430)
- **Case 1**: Should achieve ~35,934 (Baseline: 35,900)
- **Case 2**: Should achieve ~36,038 (Baseline: 36,000)

---

## Implementation Notes

### Performance Considerations
1. **Large Graphs** (map2/map3): MC simulation is expensive
   - Use candidate pool to limit evaluations
   - Reduce MC simulations for faster execution (e.g., 30 instead of 50)
   
2. **Small Graphs** (map1): MC simulation is fast
   - Can use more MC simulations for better accuracy
   - Larger candidate pools are acceptable

### Key Insights
1. **LFA Estimation Bias**: Pure IMRank with LFA has systematic error on large graphs
2. **MC Correction**: MC-guided selection fixes this while keeping heuristic efficiency
3. **Balance Matters**: Unequal |S1| and |S2| sizes can hurt performance
4. **Candidate Quality**: IMRank produces better candidates than random or degree-based selection

### Common Pitfalls
1. **File Paths**: 
   - map1: dataset1
   - map2: dataset2
   - map3: dataset2 (shares network with map2), seed2
   
2. **Seed Naming**: map3 uses `seed2` not `seed`

3. **Budget Constraint**: Must satisfy |S1| + |S2| ≤ k exactly in output format

---

## References

1. **IMRank**: "Influence Maximization via Finding Self-Consistent Ranking" (SIGIR 2014)
2. **LFA Strategy**: Last-to-First Allocating for influence estimation
3. **IC Model**: Independent Cascade model for influence diffusion
4. **Binary GA**: Standard genetic algorithm with penalty-based constraint handling

---

*Last Updated*: After achieving all baselines with MC-Guided Heuristic
*Best Results*: Case 0: 445.42, Case 1: 35934.41, Case 2: 36037.76
