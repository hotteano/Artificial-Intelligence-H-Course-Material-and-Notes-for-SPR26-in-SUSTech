# 🤖 Artificial Intelligence (H) Notes

---

## 📚 Table of Contents

1. [An Introduction to AI](#-an-introduction-to-ai)
2. [Agents](#-agents)
3. [Search Algorithms](#-search-algorithms)
   - [Uninformed Search](#-uninformed-search)
   - [Informed Search](#-informed-search)
   - [Local Search](#-local-search)
   - [Adversarial Search](#-adversarial-search)

---

## 🌟 An Introduction to AI

### 💡 What is Artificial Intelligence?

> **定义** | Artificial Intelligence (AI) is not formally defined, but it generally refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.

AI can be categorized into two main types:

| Type | Description | Examples |
|:----:|:------------|:---------|
| 🎯 **Narrow AI** | Designed for specific tasks | Siri, AlphaGo, Recommendation systems |
| 🌐 **General AI** | Can perform any intellectual task that a human can do | *Still theoretical* |

---

### 📜 History of AI

```
Timeline of AI Development
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1956 🎓  │ Dartmouth Conference: "AI" coined
1960s-70s │ ELIZA, SHRDLU
1980s    │ 📈 Expert Systems boom
1990s    │ 🤖 Machine Learning + Big Data
2000s    │ 🧠 Deep Learning revolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **Key Milestones**:
> - 🏛️ **1956**: The term "Artificial Intelligence" was coined at the Dartmouth Conference
> - 💬 **1960s-1970s**: Early AI programs (ELIZA, SHRDLU)
> - 👔 **1980s**: Rise of expert systems
> - 📊 **1990s**: Machine learning algorithms + Big data
> - 🧠 **2000s**: Deep learning and neural networks breakthrough

---

## 🔧 Agents

### 🎭 What is an Agent?

> **定义** | An **agent** is an entity that perceives its environment through **sensors** and acts upon that environment through **actuators**.

Agents can be:
- 🔹 **Simple**: Thermostat
- 🔹 **Complex**: Self-driving car, Robot assistant

---

### 🏗️ Agent Architecture

```
┌─────────────────────────────────────────┐
│              🎯 AGENT                   │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    │
│  │ 📥 Sensors  │◄───│ 🌍 Environment│   │
│  └──────┬──────┘    └─────────────┘    │
│         │                               │
│         ▼                               │
│  ┌─────────────────┐                    │
│  │  🧠 Percept     │                    │
│  │  ⚙️ Decision    │                    │
│  │  🎬 Action      │                    │
│  └────────┬────────┘                    │
│           │                             │
│  ┌────────┴────────┐                    │
│  │ 📤 Actuators    │────────────────────┘
│  └─────────────────┘
└─────────────────────────────────────────┘
```

---

## 🔍 Search Algorithms

---

### 🔵 Uninformed Search

> 📌 **Uninformed Search** (also known as **Blind Search**) refers to search strategies that have no additional information about states beyond that provided in the problem definition.

#### 📊 Complexity Comparison

| Algorithm | Complete? | Optimal? | ⏱️ Time | 💾 Space |
|:---------:|:---------:|:--------:|:-------:|:--------:|
| **BFS** | ✅ Yes | ✅ Yes (if cost=1) | $O(b^d)$ | $O(b^d)$ |
| **UCS** | ✅ Yes | ✅ Yes | $O(b^{1+\lfloor C^*/\epsilon \rfloor})$ | $O(b^{1+\lfloor C^*/\epsilon \rfloor})$ |
| **DFS** | ❌ No | ❌ No | $O(b^m)$ | $O(bm)$ |
| **DLS** | ❌ No | ❌ No | $O(b^\ell)$ | $O(b\ell)$ |
| **IDS** | ✅ Yes | ✅ Yes (if cost=1) | $O(b^d)$ | $O(bd)$ |
| **BDS** | ✅ Yes | ✅ Yes | $O(b^{d/2})$ | $O(b^{d/2})$ |

> 📝 **Notation**: $b$ = branching factor, $d$ = depth of shallowest solution, $m$ = maximum depth, $\ell$ = depth limit, $C^*$ = cost of optimal solution

---

#### 🌊 Breadth-First Search (BFS)

> 💡 Expands the **shallowest** unexpanded node first using a **FIFO queue**.

```python
function BFS(problem):
    node ← Node(problem.INITIAL_STATE)
    if problem.IS_GOAL(node.state) then return node
    frontier ← FIFO queue with node
    reached ← {problem.INITIAL_STATE}
    
    while frontier is not empty:
        node ← POP(frontier)
        for child in EXPAND(problem, node):
            s ← child.state
            if problem.IS_GOAL(s) then return child
            if s not in reached then:
                add s to reached
                add child to frontier
    return failure
```

> ✅ **Properties**: Complete if $b$ is finite; Optimal if step costs are equal; High space complexity is the main drawback.

---

#### ⚖️ Uniform Cost Search (UCS)

> 💡 Expands the node with the **lowest path cost** $g(n)$ using a **priority queue**.

- 🔄 Equivalent to Dijkstra's algorithm
- 🎯 Always finds the least-cost path
- 📈 Explores nodes in order of increasing path cost

---

#### 🔽 Depth-First Search (DFS)

> 💡 Expands the **deepest** unexpanded node first using a **LIFO stack**.

```python
function DFS(problem):
    return RECURSIVE_DFS(problem, Node(problem.INITIAL_STATE))

function RECURSIVE_DFS(problem, node):
    if problem.IS_GOAL(node.state) then return node
    for child in EXPAND(problem, node):
        result ← RECURSIVE_DFS(problem, child)
        if result ≠ failure then return result
    return failure
```

> ⚠️ **Properties**: Not complete (may loop infinitely); Not optimal; Linear space complexity $O(bm)$ is the main advantage.

---

#### 🔢 Depth-Limited Search (DLS)

> 💡 DFS with a predetermined **depth limit** $\ell$ to avoid infinite paths.

---

#### 🔄 Iterative Deepening Search (IDS)

> 💡 Performs DLS with increasing depth limits: `0, 1, 2, 3, ...`

| ✅ Advantages | ⚠️ Considerations |
|:-------------|:------------------|
| Combines BFS completeness + DFS low space | Overhead of regenerated nodes |
| Preferred for large state spaces | Usually small overhead in practice |

---

#### ↔️ Bidirectional Search

> 💡 Runs two simultaneous searches: forward from initial state and backward from goal.

- 🛑 Stops when the two searches meet
- 🚀 Reduces time and space to $O(b^{d/2})$
- ⚠️ Requires ability to compute predecessors

---

### 🟢 Informed Search

> 📌 **Informed Search** (also known as **Heuristic Search**) uses problem-specific knowledge to find solutions more efficiently via a **heuristic function** $h(n)$.

#### 📊 Algorithm Comparison

| Algorithm | Complete? | Optimal? | ⏱️ Time | 💾 Space |
|:---------:|:---------:|:--------:|:-------:|:--------:|
| **Greedy Best-First** | ❌ No | ❌ No | $O(b^m)$ | $O(b^m)$ |
| **A\*** | ✅ Yes | ✅ Yes | Exponential | Exponential |
| **IDA\*** | ✅ Yes | ✅ Yes | Exponential | $O(bd)$ |
| **RBFS** | ✅ Yes | ✅ Yes | Exponential | $O(bd)$ |
| **SMA\*** | ✅ Yes | ✅ Yes | Exponential | Limited |

---

#### 🎯 Heuristic Function Properties

> 📐 **Admissible**: $h(n) \leq h^*(n)$ for all $n$
> - ✅ Never overestimates the cost to reach the goal
> - 🔑 Required for A* optimality

> 📐 **Consistent (Monotonic)**: $h(n) \leq c(n, a, n') + h(n')$
> - 🔺 Triangle inequality: estimated cost never decreases faster than actual step cost
> - ✅ Every consistent heuristic is admissible

---

#### 🏃 Greedy Best-First Search

> 💡 Expands the node that **appears closest** to the goal according to $h(n)$.

- 📊 Evaluation: $f(n) = h(n)$
- 🐆 Greedy approach: minimize estimated cost to goal
- ❌ Not complete and not optimal
- 🔄 Can get stuck in loops

---

#### ⭐ A* Search

> 💡 Expands the node with the lowest **combined cost**: $f(n) = g(n) + h(n)$

| Component | Meaning |
|:----------|:--------|
| $g(n)$ | 📍 Actual cost from **start** to $n$ |
| $h(n)$ | 🎯 Estimated cost from $n$ to **goal** |
| $f(n)$ | 💰 Estimated **total cost** of cheapest solution through $n$ |

> 🏆 **Optimality**: A* is optimal if $h(n)$ is **admissible** (tree search) or **consistent** (graph search).

```python
function A_STAR(problem):
    node ← Node(problem.INITIAL_STATE)
    frontier ← priority queue ordered by f = g + h, with node
    reached ← {problem.INITIAL_STATE: node}
    
    while frontier is not empty:
        node ← POP(frontier)
        if problem.IS_GOAL(node.state) then return node
        for child in EXPAND(problem, node):
            s ← child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] ← child
                add child to frontier
    return failure
```

---

#### 🔄 Iterative Deepening A* (IDA*)

> 💡 Combines iterative deepening with A* evaluation.

- 📏 Uses $f$-cost limit instead of depth limit
- 📊 Threshold is the smallest $f$-cost that exceeded the previous threshold
- 💾 Memory-efficient: $O(bd)$ space
- 🎮 Suitable for problems with large state spaces

---

#### 🌲 Recursive Best-First Search (RBFS)

> 💡 Recursive algorithm that mimics best-first search with **linear space**.

- 📝 Keeps track of the best alternative path available
- ↩️ Backtracks when current path exceeds this alternative
- 💾 Memory efficient but may re-expand nodes

---

#### 💾 Simplified Memory-Bounded A* (SMA*)

> 💡 A* with **memory limit**; when memory is full, drops the worst node.

- ✂️ Prunes nodes with highest $f$-cost
- 🧠 Remembers best descendant's cost in parent
- ✅ Complete if any solution fits in memory
- 🏆 Optimal if optimal solution fits in memory

---

### 🟡 Local Search

> 📌 **Local Search** algorithms operate by searching from a current state to neighboring states, without keeping track of paths. Suitable for **optimization problems**.

#### ✨ Characteristics

| Feature | Description |
|:--------|:------------|
| 💾 Space | Constant $O(1)$ — only keep current state |
| 🗺️ State Space | Large or infinite spaces |
| 🎯 Goal | Find best state by **objective function** |
| ⚠️ Trade-off | Not systematic — may miss optimal solutions |

#### 📊 Algorithm Comparison

| Algorithm | Complete? | Optimal? | 🔄 Escape Local Optima? |
|:---------:|:---------:|:--------:|:-----------------------:|
| **Hill Climbing** | ❌ No | ❌ No | ❌ No |
| **Random Restart HC** | ✅ Yes (prob.) | ❌ No | 🔄 Restart-based |
| **Simulated Annealing** | ✅ Yes (prob.) | ✅ Yes (prob.) | ✅ Yes |
| **Local Beam Search** | ❌ No | ❌ No | 👥 Parallel exploration |
| **Genetic Algorithm** | ❌ No | ❌ No | 🧬 Crossover/Mutation |

---

#### 🏔️ State Space Landscape

```
        🏔️ Global Max
           /\\
          /  \\
         /    \\      🏔️ Local Max
   _____/      \\____    /\\
  /              \\  \\  /  \\
 /                \\--\\/    \\____
/       🏔️ Local Max             \\____
```

| Terrain Feature | Description |
|:----------------|:------------|
| 🏔️ **Global Maximum** | Best possible state |
| 🏔️ **Local Maximum** | Better than neighbors, but not the best |
| 📏 **Plateau** | Flat area where neighbors have equal value |
| 🏔️ **Ridge** | Sequence of local maxima, difficult to navigate |

---

#### 🧗 Hill Climbing (Steepest-Ascent)

> 💡 Greedy local search that always moves to the **best neighboring state**.

```python
function HILL_CLIMBING(problem):
    current ← problem.INITIAL_STATE
    while True:
        neighbor ← highest-valued successor of current
        if VALUE(neighbor) ≤ VALUE(current) then return current
        current ← neighbor
```

**Variants**:
- 🎲 **Stochastic HC**: chooses randomly among uphill moves
- ⏩ **First-Choice HC**: generates successors randomly, picks first improvement
- 🔄 **Random-Restart HC**: multiple searches from random initial states

**Problems** ⚠️:
- 🏔️ **Local maxima**: stuck at peaks that aren't the global maximum
- 📏 **Ridges**: cause slow progress or getting stuck
- 📏 **Plateaus**: flat areas with no uphill direction

---

#### 🌡️ Simulated Annealing

> 💡 Combines hill climbing with **random walk** to escape local maxima.

> 🔬 Inspired by **metallurgical annealing**: heat metal then cool slowly to reach low-energy crystalline state.

```python
function SIMULATED_ANNEALING(problem, schedule):
    current ← problem.INITIAL_STATE
    for t = 1 to ∞:
        T ← schedule(t)                    # 🌡️ temperature
        if T = 0 then return current
        next ← randomly selected successor of current
        ΔE ← VALUE(next) - VALUE(current)
        if ΔE > 0 then current ← next      # ⬆️ uphill: always accept
        else current ← next with probability e^(ΔE/T)  # ⬇️ downhill: probabilistic
```

| Phase | Behavior |
|:------|:---------|
| 🔥 **Early** (High T) | More random exploration |
| ❄️ **Late** (Low T) | Greedy hill climbing |
| 🎯 **Theory** | With slow enough cooling, probability of optimal solution → 1 |

---

#### 👥 Local Beam Search

> 💡 Keeps track of **$k$ states** instead of just one.

```python
function LOCAL_BEAM_SEARCH(problem, k):
    states ← k randomly generated states
    while True:
        successors ← []
        for each state in states:
            successors ← successors ∪ ALL_SUCCESSORS(state)
        states ← k best successors
        if all states have same value then return best(states)
```

- 💬 Information is shared among parallel searches
- 🚀 If one search finds a good path, others follow
- ⚠️ Can suffer from **lack of diversity** (all states cluster)
- 🎲 **Stochastic Beam Search**: chooses $k$ successors probabilistically

---

#### 🧬 Genetic Algorithms (GA)

> 💡 Population-based search inspired by **biological evolution**.

```python
function GENETIC_ALGORITHM(population, fitness_fn):
    repeat:
        new_population ← empty set
        for i = 1 to SIZE(population):
            x ← RANDOM_SELECTION(population, fitness_fn)
            y ← RANDOM_SELECTION(population, fitness_fn)
            child ← REPRODUCE(x, y)
            if small random probability then child ← MUTATE(child)
            add child to new_population
        population ← new_population
    until some individual is fit enough or time expired
    return best individual

function REPRODUCE(x, y):
    n ← LENGTH(x)
    c ← random number from 1 to n
    return APPEND(SUBSTRING(x, 1, c), SUBSTRING(y, c+1, n))
```

**Key Operations** 🔑:
- 🎯 **Selection**: probabilistically choose parents based on fitness
- 🔄 **Crossover**: combine two parents to create offspring
- 🎲 **Mutation**: random alteration with small probability

**State Representation**: Usually encoded as strings (binary, real-valued, etc.)

**Applications** 🎯: Function optimization, scheduling, design problems, neural network training

---

#### 📊 Comparison Summary

| Aspect | Systematic Search | Local Search |
|:-------|:-----------------|:-------------|
| 💾 Memory | High ($O(b^d)$) | Low ($O(1)$) |
| ✅ Complete | Often yes | Usually no |
| 🏆 Optimal | Often yes | Usually no |
| 🎯 Best for | Finding paths | Optimization |
| 🗺️ State space | Any | Large or continuous |

---

### 🔴 Adversarial Search

> 📌 **Adversarial Search** deals with **multi-agent environments** where agents have conflicting goals (competitive games). The agent must consider the opponent's actions, assuming optimal play.

#### 🎮 Key Characteristics

| Property | Description |
|:---------|:------------|
| ⚖️ **Zero-sum games** | One player's gain is another's loss |
| 👁️ **Perfect information** | All players know complete game state (e.g., chess) |
| 🎯 **Deterministic** | No random elements in state transitions |
| 🔄 **Turn-taking** | Players alternate moves |

#### 🎲 Game Formulation

| Component | Description |
|:----------|:------------|
| 🚀 **Initial State** | Starting position and player to move |
| 📋 **Actions(s)** | Legal moves in state $s$ |
| 🔄 **Result(s, a)** | Transition model |
| 🏁 **Terminal-Test(s)** | Is the game over? |
| 💯 **Utility(s, p)** | Final value for player $p$ in terminal state $s$ |

---

#### ⚫⚪ Minimax Algorithm

> 💡 Assumes the opponent plays optimally to **minimize** our utility.

```python
function MINIMAX_DECISION(state):
    return argmax_{a ∈ ACTIONS(state)} MIN_VALUE(RESULT(state, a))

function MAX_VALUE(state):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← -∞
    for each a in ACTIONS(state):
        v ← MAX(v, MIN_VALUE(RESULT(state, a)))
    return v

function MIN_VALUE(state):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← +∞
    for each a in ACTIONS(state):
        v ← MIN(v, MAX_VALUE(RESULT(state, a)))
    return v
```

| Property | Value |
|:---------|:------|
| ✅ **Complete** | Yes (if game tree is finite) |
| 🏆 **Optimal** | Yes (against optimal opponent) |
| ⏱️ **Time** | $O(b^m)$ — exponential in depth |
| 💾 **Space** | $O(bm)$ — depth-first exploration |

> 📝 **Notation**: $b$ = branching factor, $m$ = maximum depth

---

#### ✂️ Alpha-Beta Pruning

> 💡 Optimization of Minimax that **prunes branches** that cannot influence the final decision.

| Parameter | Meaning |
|:----------|:--------|
| **α (alpha)** | Best value that **MAX** can guarantee at current path |
| **β (beta)** | Best value that **MIN** can guarantee at current path |

```python
function ALPHA_BETA_SEARCH(state):
    return argmax_{a} MIN_VALUE(RESULT(state, a), -∞, +∞)

function MAX_VALUE(state, α, β):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← -∞
    for each a in ACTIONS(state):
        v ← MAX(v, MIN_VALUE(RESULT(state, a), α, β))
        if v ≥ β then return v          # ✂️ β cutoff
        α ← MAX(α, v)
    return v

function MIN_VALUE(state, α, β):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← +∞
    for each a in ACTIONS(state):
        v ← MIN(v, MAX_VALUE(RESULT(state, a), α, β))
        if v ≤ α then return v          # ✂️ α cutoff
        β ← MIN(β, v)
    return v
```

| Case | Complexity |
|:-----|:-----------|
| 🏆 **Best-case** | $O(b^{m/2})$ — with perfect move ordering |
| 📊 **Average-case** | $O(b^{3m/4})$ |
| 💾 **Space** | $O(bm)$ — same as Minimax |

> 🎯 **Move Ordering Heuristics**:
> - ⚔️ Try captures before quiet moves
> - 📈 Try moves with good historical scores
> - 🗡️ Killer heuristic: moves that caused cutoffs before

---

#### ⏱️ Cutting Off Search & Evaluation Functions

For most games, exploring to terminal states is **impossible** (chess has ~$10^{40}$ nodes).

> 💡 **Approach**: Cut off search early and apply **evaluation function**:

```python
function CUTOFF_TEST(state, depth):
    return depth > limit or TERMINAL_TEST(state)

function EVAL(state):
    return estimated utility of state
```

**Evaluation Function Design** 📐:
- 🔹 **Features**: Material, position, mobility, king safety, etc.
- 📊 **Weighted linear**: $\text{Eval}(s) = w_1f_1(s) + w_2f_2(s) + ... + w_nf_n(s)$
- ✅ Must preserve **transitivity**: if $A > B$, then $\text{Eval}(A) > \text{Eval}(B)$

> 🌊 **Quiescence Search**: Extend search in "unquiet" positions (e.g., captures) to avoid horizon effect.

---

#### 🎲 Expectimax (Stochastic Games)

> 💡 For games with **chance elements** (dice rolls, card draws).

Add **chance nodes** to the game tree:

```python
function EXPECTIMAX_DECISION(state):
    return argmax_{a} EXPECT_VALUE(RESULT(state, a))

function MAX_VALUE(state):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← -∞
    for each a in ACTIONS(state):
        v ← MAX(v, EXPECT_VALUE(RESULT(state, a)))
    return v

function EXPECT_VALUE(state):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← 0
    for each outcome with probability p:
        v ← v + p × MAX_VALUE(RESULT(state, outcome))
    return v
```

| Feature | Description |
|:--------|:------------|
| 📊 **Expected value** | Calculation weighted by probabilities |
| 🎮 **Applications** | Backgammon, poker, etc. |
| ✂️ **Pruning** | Can combine with alpha-beta (expectiminimax) |

---

#### 🌳 Monte Carlo Tree Search (MCTS)

> 💡 Widely used in modern game AI (**AlphaGo**, **Leela Chess**).

Four steps repeated iteratively:

```
┌─────────────────────────────────────────────┐
│              🔄 MCTS Loop                   │
├─────────────────────────────────────────────┤
│                                             │
│  1️⃣  SELECTION                              │
│      Select child using UCB1 formula        │
│      UCB1 = (wins/visits) + C × √(ln(parent_visits)/visits) │
│                                             │
│  2️⃣  EXPANSION                              │
│      Expand one child of the selected node  │
│                                             │
│  3️⃣  SIMULATION                             │
│      Play random rollout from new node      │
│                                             │
│  4️⃣  BACKPROPAGATION                        │
│      Update statistics up the tree          │
│                                             │
└─────────────────────────────────────────────┘
```

> 📊 **Upper Confidence Bound (UCB1)**: Balances exploitation (high win rate) vs exploration (few visits).

**Advantages** ✅:
- 🎲 No domain knowledge required (pure simulation)
- ⏱️ **Anytime algorithm**: can stop at any point
- 🌳 Handles large branching factors well
- 🔄 Parallelizable

---

#### 📊 Summary Table

| Algorithm | Perfect Info | Deterministic | Optimal | Complexity |
|:---------:|:------------:|:-------------:|:-------:|:----------:|
| **Minimax** | ✅ Yes | ✅ Yes | ✅ Yes | $O(b^m)$ |
| **Alpha-Beta** | ✅ Yes | ✅ Yes | ✅ Yes | $O(b^{m/2})$ |
| **Expectimax** | ✅ Yes | ❌ No | ✅ Yes | $O(b^m)$ |
| **MCTS** | ✅/❌ | Both | ❌ No | Polynomial |

---

#### 🌍 Applications Beyond Games

| Domain | Application |
|:-------|:------------|
| 💰 **Auction design** | Bidding strategies |
| 🔒 **Network security** | Attacker-defender models |
| 📈 **Economic modeling** | Competitive markets |
| 🎖️ **Military planning** | Adversarial scenarios |
| 🤖 **Robotics** | Multi-agent coordination/competition |

## 🧮 Mathematical Logic & Knowledge Representation

> 📌 **Mathematical Logic** provides a formal foundation for representing knowledge and reasoning. It enables AI agents to draw valid conclusions from known facts.

---

### 🧩 Knowledge-Based Agents

> 💡 A **knowledge-based agent** maintains a **knowledge base (KB)** and uses **inference** to derive new information.

```
┌─────────────────────────────────────────────────────────┐
│                   🧠 Knowledge-Based Agent               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────────┐      ┌─────────────────────┐     │
│   │ 📚 Knowledge    │◄────►│   ⚙️ Inference      │     │
│   │    Base (KB)    │      │     Engine          │     │
│   └─────────────────┘      └─────────────────────┘     │
│            ▲                        │                   │
│            │                        ▼                   │
│   ┌────────┴────────┐      ┌─────────────────────┐     │
│   │ 🌍 Percepts     │      │   🎯 Actions        │     │
│   │   (Tell)        │      │   (Ask/Execute)     │     │
│   └─────────────────┘      └─────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Core Operations** 🔑:
- **TELL**: Add new facts to KB
- **ASK**: Query KB to derive conclusions

---

### 📐 Propositional Logic (命题逻辑)

> 💡 **Propositional Logic** deals with propositions that are either **true** or **false**.

#### 🔹 Syntax & Semantics

| Connective | Symbol | Meaning | Truth Table |
|:----------:|:------:|:--------|:-----------:|
| Negation | ¬ | NOT | ¬P is true when P is false |
| Conjunction | ∧ | AND | P ∧ Q is true when both are true |
| Disjunction | ∨ | OR | P ∨ Q is true when at least one is true |
| Implication | → | IF...THEN | P → Q is false only when P=true, Q=false |
| Biconditional | ↔ | IFF | P ↔ Q is true when P and Q have same value |

#### 🔹 Important Concepts

| Concept | Definition |
|:--------|:-----------|
| 📋 **Tautology** | Always true (e.g., $P \lor \neg P$) |
| ❌ **Contradiction** | Always false (e.g., $P \land \neg P$) |
| ✅ **Satisfiable** | True under some interpretation |
| 🔄 **Equivalence** | $\alpha \equiv \beta$ if same truth value in all models |
| 📊 **Entailment** | $KB \vDash \alpha$: $\alpha$ is true in all models where KB is true |

#### 🔹 Inference Rules

```
┌───────────────────────────────────────────────────────────────┐
│                   📜 Inference Rules                           │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Modus Ponens:          Modus Tollens:                        │
│  α → β, α               α → β, ¬β                             │
│  ─────────              ───────────                           │
│     β                      ¬α                                 │
│                                                               │
│  And-Elimination:       And-Introduction:                     │
│  α₁ ∧ α₂ ∧ ... ∧ αₙ     α₁, α₂, ..., αₙ                       │
│  ─────────────────      ─────────────────                     │
│        αᵢ               α₁ ∧ α₂ ∧ ... ∧ αₙ                   │
│                                                               │
│  Or-Introduction:       Resolution:                           │
│     αᵢ                  α ∨ β, ¬β ∨ γ                         │
│  ────────               ─────────────                         │
│  α₁ ∨ ... ∨ αₙ              α ∨ γ                             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### 🔹 Resolution Algorithm

> 💡 **Resolution** is a complete inference algorithm for propositional logic.

**Steps**:
1. Convert all sentences to **Conjunctive Normal Form (CNF)**
2. Apply resolution rule repeatedly
3. If empty clause derived → contradiction found

**CNF Conversion** 🔄:
- Eliminate ↔ (biconditional)
- Eliminate → (implication)
- Move ¬ inward (De Morgan's laws)
- Distribute ∨ over ∧

---

### 🎯 First-Order Logic / Predicate Logic (谓词逻辑)

> 💡 **First-Order Logic (FOL)** extends propositional logic with objects, relations, and quantifiers.

#### 🔹 Components

| Component | Description | Example |
|:----------|:------------|:--------|
| 🔹 **Constants** | Specific objects | John, 3, Red |
| 📦 **Variables** | Range over objects | x, y, z |
| 🔗 **Predicates** | Relations/properties | Brother(x,y), Prime(x) |
| ⚡ **Functions** | Map objects to objects | Father(John), Plus(x,y) |
| 🔗 **Connectives** | Same as propositional | ∧, ∨, ¬, →, ↔ |
| 📊 **Quantifiers** | Universal / Existential | ∀, ∃ |

#### 🔹 Quantifiers

| Quantifier | Meaning | Example |
|:----------:|:--------|:--------|
| **∀** (Universal) | "For all" | ∀x Cat(x) → Mammal(x) |
| **∃** (Existential) | "There exists" | ∃x Student(x) ∧ Brilliant(x) |

**Quantifier Equivalences** 🔄:
- $\forall x \, P(x) \equiv \neg \exists x \, \neg P(x)$
- $\exists x \, P(x) \equiv \neg \forall x \, \neg P(x)$
- $\forall x \, P(x) \land Q(x) \equiv (\forall x \, P(x)) \land (\forall x \, Q(x))$

#### 🔹 Knowledge Engineering in FOL

1. **Identify tasks**: What questions should be answerable?
2. **Gather knowledge**: Collect relevant domain facts
3. **Define vocabulary**: Constants, functions, predicates
4. **Encode general rules**: General domain knowledge
5. **Encode specific facts**: Particular problem instance

---

### 🔄 Forward & Backward Chaining

> 💡 For **Horn clauses** (disjunctions with at most one positive literal), efficient inference is possible.

#### 🔹 Forward Chaining (Data-Driven)

```python
function FORWARD_CHAINING(KB, query):
    count ← table of Horn clause premises
    inferred ← table of symbols (all false initially)
    agenda ← queue of known true symbols
    
    while agenda is not empty:
        p ← POP(agenda)
        if p == query then return true
        if inferred[p] == false:
            inferred[p] ← true
            for each Horn clause c where p is in premise:
                decrement count[c]
                if count[c] == 0:
                    add head(c) to agenda
    return false
```

- ✅ **Complete** and **sound**
- ⏱️ **Linear time**: $O(|KB|)$
- 📊 May derive many irrelevant facts

#### 🔹 Backward Chaining (Goal-Driven)

> 💡 Works backward from query to known facts.

- ✅ **Complete** and **sound**
- 💾 **Linear space**: depth of proof tree
- 🎯 Focuses only on relevant facts
- ⚠️ May enter infinite loops (need cycle checking)

```
┌────────────────────────────────────────────────────────────┐
│          Forward vs Backward Chaining                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Forward:                    Backward:                     │
│                                                            │
│      Facts ──► Goals            Goals ──► Facts            │
│                                                            │
│  🌊 Data-driven              🎯 Goal-driven                │
│  📈 Bottom-up                📉 Top-down                   │
│  💡 Good for monitoring      💡 Good for diagnosis         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 📊 Summary: Logic Systems

| Feature | Propositional | First-Order |
|:--------|:-------------|:------------|
| **Objects** | ❌ No | ✅ Yes (constants, variables) |
| **Relations** | ❌ No | ✅ Yes (predicates) |
| **Functions** | ❌ No | ✅ Yes |
| **Quantifiers** | ❌ No | ✅ Yes (∀, ∃) |
| **Expressiveness** | Limited | High |
| **Decidability** | Decidable | Semi-decidable |
| **Complexity** | NP-complete | Undecidable |

---

## 🧠 Machine Learning

> 📌 **Machine Learning** enables systems to **learn from data** and improve performance without explicit programming.

```
┌─────────────────────────────────────────────────────────────────┐
│                    🎯 ML Paradigm                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📊 Training Data ──► 🧠 Learning Algorithm ──► 📦 Model      │
│        (Examples)          (Induction)           (Hypothesis)   │
│                                                                 │
│   📦 Model ──► 🔮 Prediction on New Data                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 📚 Learning Types

| Type | Description | Goal | Examples |
|:-----|:------------|:-----|:---------|
| 👨‍🏫 **Supervised** | Labeled data | Learn mapping X→Y | Classification, Regression |
| 🔍 **Unsupervised** | Unlabeled data | Discover patterns | Clustering, Dimensionality reduction |
| 🎮 **Reinforcement** | Trial & error | Maximize reward | Game playing, Robotics |
| 📝 **Semi-supervised** | Mix of labeled/unlabeled | Use unlabeled to help | When labeling is expensive |

---

### 📈 Regression Analysis

> 💡 **Regression** predicts continuous values.

#### 🔹 Linear Regression

> 📐 Models relationship as linear function.

**Hypothesis**: $h_\theta(x) = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n = \theta^T x$

**Cost Function** (MSE): $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$

**Gradient Descent** 🔄:
```python
repeat until convergence:
    θ_j := θ_j - α ∂J(θ)/∂θ_j
    # where ∂J/∂θ_j = (1/m) Σ(h_θ(xⁱ) - yⁱ)x_jⁱ
```

| Method | Pros | Cons |
|:-------|:-----|:-----|
| **Gradient Descent** | Scales to large data | Needs learning rate tuning |
| **Normal Equation** | No iteration needed | Slow for large n ($O(n^3)$) |
| **Stochastic GD** | Fast per iteration | Noisy convergence |

#### 🔹 Logistic Regression

> 📐 For **binary classification** despite the name "regression".

**Sigmoid Function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

**Hypothesis**: $h_\theta(x) = \sigma(\theta^T x) = P(y=1|x;\theta)$

**Cost Function** (Log Loss):
$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$

**Decision Boundary**: Predict $y=1$ if $h_\theta(x) \geq 0.5$

---

### 🧩 Perceptron & Neural Networks

#### 🔹 Perceptron (感知机)

```
        ┌─────────────────────────┐
│        Perceptron Architecture        │
├───────────────────────────────────────┤
│                                       │
│   x₁ ──► [w₁] ──┐                     │
│   x₂ ──► [w₂] ──┼──► Σ ──► σ ──► y   │
│   x₃ ──► [w₃] ──┘    │                │
│                      ▼                │
│                    [b] bias           │
│                                       │
│   y = σ(w₁x₁ + w₂x₂ + w₃x₃ + b)      │
│                                       │
└───────────────────────────────────────┘
```

**Activation Functions** ⚡:

| Function | Formula | Range | Properties |
|:---------|:--------|:------|:-----------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0,1) | Smooth, but vanishing gradient |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1,1) | Zero-centered, still vanishing |
| **ReLU** | $\max(0, x)$ | [0,∞) | Fast, but "dying ReLU" problem |
| **Leaky ReLU** | $\max(\alpha x, x)$ | (-∞,∞) | Solves dying ReLU |

#### 🔹 Multi-Layer Neural Networks

> 💡 **Deep Learning**: Networks with multiple hidden layers.

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
     │               │                 │               │
   [x₁] ─────────► [h₁] ──────────► [h₁'] ────────► [y]
   [x₂] ─────────► [h₂] ──────────► [h₂'] ────────►
   [x₃] ─────────► [h₃] ──────────► [h₃'] ────────►
   [...]         [...]            [...]
   
   Forward:  zˡ = Wˡaˡ⁻¹ + bˡ    aˡ = σ(zˡ)
   Backward: ∂J/∂Wˡ = ∂J/∂zˡ · ∂zˡ/∂Wˡ
```

**Backpropagation Algorithm** 🔄:
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients
4. Update weights using gradient descent

---

### 🌳 Decision Trees & Random Forest

#### 🔹 Decision Trees

> 💡 Split data based on feature values to maximize information gain.

**Entropy**: $H(S) = -\sum_{i} p_i \log_2(p_i)$

**Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$

```
                    [Outlook?]
                   /    │    \
                Sunny  Rain  Overcast
                 │      │       │
              [Humidity?] [Wind?]  [Play]
              /      \   /    \
           High     Normal Strong Weak
            │         │     │     │
          [No]      [Yes] [No] [Yes]
```

| Algorithm | Split Criterion | Handling |
|:----------|:----------------|:---------|
| **ID3** | Information Gain | Categorical only |
| **C4.5** | Gain Ratio | Categorical + Numeric |
| **CART** | Gini Index | Binary trees |

#### 🔹 Ensemble Methods

> 💡 Combine multiple weak learners to create a strong learner.

**Bagging (Bootstrap Aggregating)** 🎒:
- Train multiple models on bootstrap samples
- Average predictions (regression) or vote (classification)
- Reduces variance

**Random Forest** 🌲🌲🌲:
- Bagging + random feature subsets
- Highly parallelizable
- Robust to overfitting

**Boosting** 📈:
- Train models sequentially, focusing on errors
- AdaBoost, Gradient Boosting, XGBoost
- Reduces bias

---

### 🎲 Naive Bayes

> 💡 Probabilistic classifier based on Bayes' theorem with "naive" independence assumption.

**Bayes' Theorem**: $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$

**Naive Assumption**: Features are conditionally independent given the class.

$$P(Y|X_1, ..., X_n) = \frac{P(Y) \prod_{i=1}^{n} P(X_i|Y)}{P(X_1, ..., X_n)}$$

**Types**:
| Variant | Use Case | Distribution |
|:--------|:---------|:-------------|
| **Gaussian NB** | Continuous features | Normal distribution |
| **Multinomial NB** | Text classification | Word counts |
| **Bernoulli NB** | Binary features | Binary distribution |

---

### 🔍 Unsupervised Learning

#### 🔹 K-Means Clustering

> 💡 Partition data into $k$ clusters by minimizing within-cluster sum of squares.

```python
function K_MEANS(Dataset X, int k):
    Initialize k centroids μ₁, ..., μₖ randomly
    repeat until convergence:
        # Assignment step
        for each xᵢ in X:
            cᵢ ← argminⱼ ||xᵢ - μⱼ||²
        
        # Update step
        for j = 1 to k:
            μⱼ ← mean of all xᵢ where cᵢ = j
    return centroids μ and assignments c
```

**Properties**:
- ⏱️ Complexity: $O(n \cdot k \cdot d \cdot i)$ where $i$ = iterations
- ⚠️ Sensitive to initialization and outliers
- 📊 Requires choosing $k$ beforehand

#### 🔹 Hierarchical Clustering

| Type | Description |
|:-----|:------------|
| **Agglomerative** (Bottom-up) | Start with singletons, merge iteratively |
| **Divisive** (Top-down) | Start with all data, split recursively |

**Linkage Criteria** 🔗:
- Single linkage: min distance between clusters
- Complete linkage: max distance between clusters
- Average linkage: avg distance between clusters
- Ward's method: minimize variance

#### 🔹 Dimensionality Reduction

> 💡 Reduce number of features while preserving information.

**PCA (Principal Component Analysis)** 📊:
1. Standardize data
2. Compute covariance matrix
3. Find eigenvectors/eigenvalues
4. Select top $k$ eigenvectors
5. Project data onto new basis

---

### 🎮 Reinforcement Learning (RL)

> 💡 Agent learns to make decisions by performing actions in an environment to maximize cumulative reward.

```
┌─────────────────────────────────────────────────────────────────┐
│              🎮 Reinforcement Learning Loop                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│         ┌──────────┐           Reward R            ┌──────┐    │
│         │          │◄──────────────────────────────┤      │    │
│         │  Agent   │                               │ Env  │    │
│         │          ├──────────────────────────────►│      │    │
│         └────┬─────┘           Action A            └──────┘    │
│              │                                    ▲            │
│              │                                    │            │
│              └────────── State S ─────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 🔹 Key Components

| Component | Symbol | Description |
|:----------|:-------|:------------|
| **State** | $s$ | Environment configuration |
| **Action** | $a$ | What agent can do |
| **Reward** | $r$ | Immediate feedback |
| **Policy** | $\pi(s)$ | Strategy: which action to take |
| **Value** | $V(s)$ | Expected cumulative reward from $s$ |
| **Q-Value** | $Q(s,a)$ | Expected reward for taking $a$ in $s$ |

#### 🔹 Q-Learning

> 💡 Model-free RL algorithm to learn optimal action-value function.

**Bellman Equation**: $Q(s,a) = r + \gamma \max_{a'} Q(s', a')$

**Update Rule**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$$

Where:
- $\alpha$ = learning rate
- $\gamma$ = discount factor

**Exploration vs Exploitation** 🎯:
- $\epsilon$-greedy: With prob $\epsilon$, random action; else best action
- Decaying $\epsilon$: Start exploring, gradually exploit

---

### 🗣️ Natural Language Processing (NLP)

> 💡 Enable computers to understand, interpret, and generate human language.

#### 🔹 Text Preprocessing

| Step | Description | Example |
|:-----|:------------|:--------|
| **Tokenization** | Split into words/tokens | "Hello world" → ["Hello", "world"] |
| **Normalization** | Lowercase, remove punctuation | "Hello!" → "hello" |
| **Stopword Removal** | Remove common words | Remove "the", "is", "at" |
| **Stemming/Lemmatization** | Reduce to root form | "running", "ran" → "run" |

#### 🔹 Text Representations

| Method | Description | Pros/Cons |
|:-------|:------------|:----------|
| **Bag of Words** | Word frequency vectors | Simple, loses word order |
| **TF-IDF** | Weight by inverse document frequency | Reduces common word impact |
| **Word Embeddings** | Dense vectors (Word2Vec, GloVe) | Captures semantic meaning |
| **Transformer** | Contextual embeddings (BERT) | State-of-the-art, expensive |

#### 🔹 NLP Tasks

- 📝 **Text Classification**: Sentiment analysis, spam detection
- 🏷️ **Named Entity Recognition (NER)**: Identify entities (person, org, location)
- 🔗 **Part-of-Speech Tagging**: Label word types (noun, verb, etc.)
- 🔄 **Machine Translation**: Translate between languages
- ❓ **Question Answering**: Answer questions from text
- 📝 **Text Summarization**: Generate concise summaries

---

### 📊 ML Model Evaluation

#### 🔹 Metrics for Classification

| Metric | Formula | Use Case |
|:-------|:--------|:---------|
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | Balanced classes |
| **Precision** | $TP / (TP + FP)$ | Minimize false positives |
| **Recall** | $TP / (TP + FN)$ | Minimize false negatives |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Balance precision/recall |
| **AUC-ROC** | Area under ROC curve | Threshold-independent |

```
                Predicted
              ┌───────┬───────┐
         Yes  │  TP   │  FN   │   ← Actual Positive
Actual        ├───────┼───────┤
         No   │  FP   │  TN   │   ← Actual Negative
              └───────┴───────┘
                  ↑         ↑
            Predicted   Predicted
            Positive    Negative
```

#### 🔹 Bias-Variance Tradeoff

```
┌─────────────────────────────────────────────────────────────┐
│              📊 Bias-Variance Tradeoff                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   High Bias (Underfitting)      High Variance (Overfitting) │
│        ┌─────┐                      ┌─┐ ┌─┐ ┌─┐             │
│       /       \                   /   X   X   \            │
│      /         \                /  X     X     X           │
│     /           \             X    X   X   X    X          │
│    /             \          X  X   X       X  X  X         │
│                                                             │
│   Too simple                     Too complex                │
│   Miss patterns                  Fits noise                 │
│                                                             │
│   Solution:                      Solution:                  │
│   • More features                • More data                │
│   • More complex model           • Regularization           │
│   • Reduce regularization        • Simpler model            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Regularization Techniques** 🔧:
- **L1 (Lasso)**: $\lambda \sum |w_i|$ → Sparse weights, feature selection
- **L2 (Ridge)**: $\lambda \sum w_i^2$ → Small weights, no sparsity
- **Dropout**: Randomly drop neurons during training
- **Early Stopping**: Stop when validation error increases

---

### 🎯 Summary: ML Algorithms

| Task | Algorithms | Key Considerations |
|:-----|:-----------|:-------------------|
| **Regression** | Linear, Polynomial, SVR | Feature scaling, non-linearity |
| **Classification** | Logistic Regression, SVM, k-NN, Trees | Class imbalance, decision boundary |
| **Neural Nets** | MLP, CNN, RNN, Transformers | Architecture, hyperparameters |
| **Clustering** | k-Means, DBSCAN, Hierarchical | Choosing k, density assumptions |
| **Dimensionality** | PCA, t-SNE, Autoencoders | Information loss, visualization |
| **RL** | Q-Learning, Policy Gradient, Actor-Critic | Exploration, reward design |

---

> 📚 **End of Notes** | 🎓 Artificial Intelligence Course Material | 🏆 Happy Learning!