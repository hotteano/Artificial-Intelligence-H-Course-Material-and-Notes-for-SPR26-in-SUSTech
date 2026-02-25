# Artificial Intelligence (H) Notes



## An Introduction to AI

### What is Artificial Intelligence?

Artificial Intelligence (AI) is not formally defined, but it generally refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI can be categorized into two main types: narrow AI, which is designed for specific tasks, and general AI, which has the ability to perform any intellectual task that a human can do.

### History of AI

The concept of AI has been around for centuries, with early ideas appearing in mythology and fiction. However, the formal field of AI research began in the mid-20th century. Key milestones include:
> - 1956: The term "Artificial Intelligence" was  coined at the Dartmouth Conference.
> - 1960s-1970s: Early AI programs were developed, such as ELIZA and SHRDLU.
> - 1980s: The rise of expert systems, which were designed to mimic the decision-making abilities of human experts.
> - 1990s: The development of machine learning algorithms and the advent of big data.
> - 2000s: The emergence of deep learning and neural networks, leading to significant advancements in AI capabilities.

## Agents

What is an Agent?

> An agent is an entity that perceives its environment through sensors and acts upon that environment through actuators. Agents can be simple, such as a thermostat, or complex, such as a self-driving car.

The agent can be sketch as follows:

```
+-----------------+
|   Input Form    | < --Sensor -- +---------------+
|Decisive Function|               |  Environment  |
|   Output Form   | --Actuator--> +---------------+
+-----------------+
```

## Search Algorithms

### Uninformed Search

Uninformed Search (also known as **Blind Search**) refers to search strategies that have no additional information about states beyond that provided in the problem definition. All they can do is generate successors and distinguish a goal state from a non-goal state.

| Algorithm | Complete? | Optimal? | Time Complexity | Space Complexity |
|-----------|:---------:|:--------:|:---------------:|:----------------:|
| **BFS** | Yes | Yes (if cost=1) | $O(b^d)$ | $O(b^d)$ |
| **UCS** | Yes | Yes | $O(b^{1+\lfloor C^*/\epsilon \rfloor})$ | $O(b^{1+\lfloor C^*/\epsilon \rfloor})$ |
| **DFS** | No | No | $O(b^m)$ | $O(bm)$ |
| **DLS** | No | No | $O(b^\ell)$ | $O(b\ell)$ |
| **IDS** | Yes | Yes (if cost=1) | $O(b^d)$ | $O(bd)$ |
| **BDS** | Yes | Yes | $O(b^{d/2})$ | $O(b^{d/2})$ |

> **Notation**: $b$ = branching factor, $d$ = depth of shallowest solution, $m$ = maximum depth, $\ell$ = depth limit, $C^*$ = cost of optimal solution

#### Breadth-First Search (BFS)

Expands the shallowest unexpanded node first using a **FIFO queue**.

```
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

**Properties**: Complete if $b$ is finite; Optimal if step costs are equal; High space complexity is the main drawback.

#### Uniform Cost Search (UCS)

Expands the node with the lowest path cost $g(n)$ using a **priority queue**.

- Equivalent to Dijkstra's algorithm
- Always finds the least-cost path
- Explores nodes in order of increasing path cost

#### Depth-First Search (DFS)

Expands the deepest unexpanded node first using a **LIFO stack** (or recursion).

```
function DFS(problem):
    return RECURSIVE_DFS(problem, Node(problem.INITIAL_STATE))

function RECURSIVE_DFS(problem, node):
    if problem.IS_GOAL(node.state) then return node
    for child in EXPAND(problem, node):
        result ← RECURSIVE_DFS(problem, child)
        if result ≠ failure then return result
    return failure
```

**Properties**: Not complete (may loop infinitely); Not optimal; Linear space complexity $O(bm)$ is the main advantage.

#### Depth-Limited Search (DLS)

DFS with a predetermined depth limit $\ell$ to avoid infinite paths.

#### Iterative Deepening Search (IDS)

Performs DLS with increasing depth limits: 0, 1, 2, 3, ...

- Combines benefits of BFS (completeness, optimality) and DFS (low space)
- Preferred for large state spaces with unknown solution depth
- Overhead of regenerated nodes is usually small

#### Bidirectional Search

Runs two simultaneous searches: forward from initial state and backward from goal.

- Stops when the two searches meet
- Reduces time and space to $O(b^{d/2})$
- Requires ability to compute predecessors

---

### Informed Search

Informed Search (also known as **Heuristic Search**) uses problem-specific knowledge beyond the problem definition to find solutions more efficiently. This knowledge is encoded in a **heuristic function** $h(n)$ that estimates the cost from node $n$ to the goal.

| Algorithm | Complete? | Optimal? | Time | Space |
|-----------|:---------:|:--------:|:----:|:-----:|
| **Greedy Best-First** | No | No | $O(b^m)$ | $O(b^m)$ |
| **A\*** | Yes | Yes | Exponential | Exponential |
| **IDA\*** | Yes | Yes | Exponential | $O(bd)$ |
| **RBFS** | Yes | Yes | Exponential | $O(bd)$ |
| **SMA\*** | Yes | Yes | Exponential | Limited |

#### Heuristic Function Properties

**Admissible**: $h(n) \leq h^*(n)$ for all $n$, where $h^*(n)$ is the true cost to goal.
- Never overestimates the cost to reach the goal
- Required for A* optimality

**Consistent (Monotonic)**: $h(n) \leq c(n, a, n') + h(n')$ for all $n, n', a$
- Triangle inequality: estimated cost never decreases faster than actual step cost
- Every consistent heuristic is admissible

#### Greedy Best-First Search

Expands the node that appears closest to the goal according to $h(n)$.

- Evaluation function: $f(n) = h(n)$
- Greedy approach: tries to minimize estimated cost to goal
- Not complete and not optimal
- Can get stuck in loops

#### A* Search

Expands the node with the lowest combined cost: $f(n) = g(n) + h(n)$

- $g(n)$: actual cost from start to $n$
- $h(n)$: estimated cost from $n$ to goal
- $f(n)$: estimated total cost of cheapest solution through $n$

**Optimality**: A* is optimal if $h(n)$ is admissible (for tree search) or consistent (for graph search).

```
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

#### Iterative Deepening A* (IDA*)

Combines iterative deepening with A* evaluation.

- Uses $f$-cost limit instead of depth limit
- Threshold is the smallest $f$-cost that exceeded the previous threshold
- Memory-efficient: $O(bd)$ space
- Suitable for problems with large state spaces

#### Recursive Best-First Search (RBFS)

Recursive algorithm that mimics best-first search with linear space.

- Keeps track of the best alternative path available
- Backtracks when current path exceeds this alternative
- Memory efficient but may re-expand nodes

#### Simplified Memory-Bounded A* (SMA*)

A* with memory limit; when memory is full, drops the worst node.

- Prunes nodes with highest $f$-cost
- Remembers best descendant's cost in parent
- Complete if any solution fits in memory
- Optimal if optimal solution fits in memory

### Local Search

Local Search algorithms operate by searching from a current state to neighboring states, without keeping track of paths or reached states. They are suitable for **optimization problems** where the goal is to find the best state according to an **objective function**.

**Characteristics**:
- Constant space complexity $O(1)$ — only keep current state
- Can find reasonable solutions in large or infinite state spaces
- Not systematic — may miss optimal solutions
- Complete/optimal only with infinite time or specific conditions

| Algorithm | Complete? | Optimal? | Escape Local Optima? |
|-----------|:---------:|:--------:|:--------------------:|
| **Hill Climbing** | No | No | No |
| **Random Restart HC** | Yes (probabilistic) | No | Restart-based |
| **Simulated Annealing** | Yes (probabilistic) | Yes (probabilistic) | Yes |
| **Local Beam Search** | No | No | Parallel exploration |
| **Genetic Algorithm** | No | No | Crossover/Mutation |

#### State Space Landscape

Local search can be visualized as navigating a landscape:
- **Global maximum**: best possible state
- **Local maximum**: better than neighbors, but not the best
- **Plateau**: flat area where neighbors have equal value
- **Ridge**: sequence of local maxima, difficult to navigate

```
        Global Max
           /\\
          /  \\
         /    \\      Local Max
   _____/      \\____    /\\
  /              \\  \\  /  \\
 /                \\--\\/    \\____
/                                  \\____
```

#### Hill Climbing (Steepest-Ascent)

Greedy local search that always moves to the best neighboring state.

```
function HILL_CLIMBING(problem):
    current ← problem.INITIAL_STATE
    while True:
        neighbor ← highest-valued successor of current
        if VALUE(neighbor) ≤ VALUE(current) then return current
        current ← neighbor
```

**Variants**:
- **Stochastic HC**: chooses randomly among uphill moves
- **First-Choice HC**: generates successors randomly, picks first improvement
- **Random-Restart HC**: conducts multiple hill-climbing searches from random initial states

**Problems**:
- **Local maxima**: stuck at peaks that aren't the global maximum
- **Ridges**: cause slow progress or getting stuck
- **Plateaus**: flat areas with no uphill direction

#### Simulated Annealing

Combines hill climbing with random walk to escape local maxima.

Inspired by metallurgical annealing: heat metal then cool slowly to reach low-energy crystalline state.

```
function SIMULATED_ANNEALING(problem, schedule):
    current ← problem.INITIAL_STATE
    for t = 1 to ∞:
        T ← schedule(t)                    // temperature
        if T = 0 then return current
        next ← randomly selected successor of current
        ΔE ← VALUE(next) - VALUE(current)
        if ΔE > 0 then current ← next      // uphill: always accept
        else current ← next with probability e^(ΔE/T)  // downhill: probabilistic
```

**Key Properties**:
- Early: high temperature → more random exploration
- Late: low temperature → greedy hill climbing
- With slow enough cooling, probability of optimal solution approaches 1

#### Local Beam Search

Keeps track of $k$ states instead of just one.

```
function LOCAL_BEAM_SEARCH(problem, k):
    states ← k randomly generated states
    while True:
        successors ← []
        for each state in states:
            successors ← successors ∪ ALL_SUCCESSORS(state)
        states ← k best successors
        if all states have same value then return best(states)
```

- Information is shared among parallel searches
- If one search finds a good path, others follow
- Can suffer from **lack of diversity** (all states cluster)
- **Stochastic Beam Search**: chooses $k$ successors probabilistically

#### Genetic Algorithms (GA)

Population-based search inspired by biological evolution.

```
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

**Key Operations**:
- **Selection**: probabilistically choose parents based on fitness
- **Crossover**: combine two parents to create offspring
- **Mutation**: random alteration with small probability

**State Representation**: Usually encoded as strings (binary, real-valued, etc.)

**Applications**: Function optimization, scheduling, design problems, neural network training

#### Comparison Summary

| Aspect | Systematic Search | Local Search |
|--------|------------------|--------------|
| Memory | High ($O(b^d)$) | Low ($O(1)$) |
| Complete | Often yes | Usually no |
| Optimal | Often yes | Usually no |
| Best for | Finding paths | Optimization |
| State space | Any | Large or continuous |

### Adversarial Search

Adversarial Search deals with **multi-agent environments** where agents have conflicting goals (competitive games). Unlike single-agent search, the agent must consider the opponent's actions, assuming the opponent plays optimally to minimize the agent's utility.

**Key Characteristics**:
- **Zero-sum games**: One player's gain is another's loss (sum of utilities = constant)
- **Perfect information**: All players know the complete game state (e.g., chess)
- **Deterministic**: No random elements in state transitions
- **Turn-taking**: Players alternate moves

**Game Formulation**:
- **Initial State**: Starting position and player to move
- **Actions(s)**: Legal moves in state $s$
- **Result(s, a)**: Transition model
- **Terminal-Test(s)**: Is the game over?
- **Utility(s, p)**: Final value for player $p$ in terminal state $s$

#### Minimax Algorithm

Assumes the opponent plays optimally to minimize our utility. The algorithm computes the **minimax value** for each node.

```
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

**Properties**:
- **Complete**: Yes (if game tree is finite)
- **Optimal**: Yes (against optimal opponent)
- **Time Complexity**: $O(b^m)$ — exponential in depth
- **Space Complexity**: $O(bm)$ — depth-first exploration

> $b$ = branching factor, $m$ = maximum depth

#### Alpha-Beta Pruning

Optimization of Minimax that prunes branches that cannot influence the final decision.

- **α (alpha)**: Best value that MAX can guarantee at current path
- **β (beta)**: Best value that MIN can guarantee at current path

```
function ALPHA_BETA_SEARCH(state):
    return argmax_{a ∈ ACTIONS(state)} MIN_VALUE(RESULT(state, a), -∞, +∞)

function MAX_VALUE(state, α, β):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← -∞
    for each a in ACTIONS(state):
        v ← MAX(v, MIN_VALUE(RESULT(state, a), α, β))
        if v ≥ β then return v          // β cutoff
        α ← MAX(α, v)
    return v

function MIN_VALUE(state, α, β):
    if TERMINAL_TEST(state) then return UTILITY(state)
    v ← +∞
    for each a in ACTIONS(state):
        v ← MIN(v, MAX_VALUE(RESULT(state, a), α, β))
        if v ≤ α then return v          // α cutoff
        β ← MIN(β, v)
    return v
```

**Properties**:
- **Same result as Minimax** (pruning doesn't affect optimality)
- **Best-case**: $O(b^{m/2})$ — with perfect move ordering
- **Average-case**: $O(b^{3m/4})$
- **Space**: $O(bm)$ — same as Minimax

**Move Ordering**: Heuristic to order moves for better pruning:
- Try captures before quiet moves
- Try moves with good historical scores
- Killer heuristic: moves that caused cutoffs before

#### Cutting Off Search & Evaluation Functions

For most games, exploring to terminal states is impossible (chess has ~$10^{40}$ nodes).

**Approach**: Cut off search early and apply **evaluation function**:

```
function CUTOFF_TEST(state, depth):
    return depth > limit or TERMINAL_TEST(state)

function EVAL(state):
    return estimated utility of state
```

**Evaluation Function Design**:
- **Features**: Material, position, mobility, king safety, etc.
- **Weighted linear function**: $\text{Eval}(s) = w_1f_1(s) + w_2f_2(s) + ... + w_nf_n(s)$
- Must preserve **transitivity**: if $A > B$, then $\text{Eval}(A) > \text{Eval}(B)$

**Quiescence Search**: Extend search in "unquiet" positions (e.g., when captures are possible) to avoid horizon effect.

#### Expectimax (Stochastic Games)

For games with **chance elements** (e.g., dice rolls, card draws).

Add **chance nodes** to the game tree:

```
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

**Properties**:
- Expected value calculation weighted by probabilities
- Used in backgammon, poker, etc.
- Can combine with alpha-beta pruning (expectiminimax)

#### Monte Carlo Tree Search (MCTS)

Widely used in modern game AI (AlphaGo, Leela Chess).

Four steps repeated iteratively:

```
1. SELECTION: Select child using UCB1 formula from root to leaf
   UCB1 = (wins/visits) + C × √(ln(parent_visits)/visits)

2. EXPANSION: Expand one child of the selected node

3. SIMULATION: Play random rollout from new node to terminal

4. BACKPROPAGATION: Update statistics up the tree
```

**Upper Confidence Bound (UCB1)**: Balances exploitation (high win rate) vs exploration (few visits).

**Advantages**:
- No domain knowledge required (pure simulation)
- Anytime algorithm: can stop at any point
- Handles large branching factors well
- Parallelizable

#### Summary Table

| Algorithm | Perfect Info | Deterministic | Optimal | Complexity |
|-----------|:------------:|:-------------:|:-------:|:----------:|
| **Minimax** | Yes | Yes | Yes | $O(b^m)$ |
| **Alpha-Beta** | Yes | Yes | Yes | $O(b^{m/2})$ |
| **Expectimax** | Yes | No | Yes | $O(b^m)$ |
| **MCTS** | Yes/No | Both | No | Polynomial |

#### Applications Beyond Games

- **Auction design**: Bidding strategies
- **Network security**: Attacker-defender models
- **Economic modeling**: Competitive markets
- **Military planning**: Adversarial scenarios
- **Robotics**: Multi-agent coordination/competition

## Mathematical Logic

We omit this section for now, as it is not directly related to search algorithms and game playing.

## Machine Learning Architectures

### Regression

#### Linear Regression

#### Logistic Regression

### Perceptron and Neural Networks

### Decision Trees and Naive Bayes

### Esemble Learning and Clustering

### NLP