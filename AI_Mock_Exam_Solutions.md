# AI Mock Final Exam — 详细解答

> 课程：Artificial Intelligence (H) | Spring 2026 | SUSTech

---

## Problem 1: Search (15 pts)

### (a) BFS 树与扩展顺序

BFS 按层扩展，首次发现的节点构成搜索树，字母顺序打破平局。

| 步骤 | 出队（扩展） | 新发现的子节点 | 队列（FIFO） |
|------|-------------|---------------|-------------|
| 1 | **A** | B, C, D | B, C, D |
| 2 | **B** | E, F | C, D, E, F |
| 3 | **C** | G (Goal!) | D, E, F, G |
| 4 | **D** | H | E, F, G, H |
| 5 | **E** | I | F, G, H, I |
| 6 | **F** | — | G, H, I |

**扩展顺序：** **A → B → C → D → E → F → G**

**BFS 树：**

```
        A
      / | \
     B  C  D
    / \  \  \
   E   F  G  H
   |
   I
```

> 注：F 在 C 下也邻接，但已在 B 下生成，不重复加入树；I 在 F 下同理。

---

### (b) A\* 搜索（c = 1）

$f(n) = g(n) + h(n)$，优先队列按 $f$ 排序，同 $f$ 按字母序。

| 扩展节点 | 生成的节点 | $g$ | $h$ | $f$ | 优先队列状态（按 $f$） |
|---------|-----------|-----|-----|-----|----------------------|
| **A** | B | 1 | 3 | 4 | C(3), B(4), D(4) |
| | C | 1 | 2 | **3** | |
| | D | 1 | 3 | 4 | |
| **C** | F | 2 | 1 | **3** | G(**2**), F(3), B(4), D(4) |
| | G | 2 | 0 | **2** | |
| **G** | — | — | — | — | 目标被选中扩展，停止 |

**生成节点的 $(g, h, f)$：**

- A: $(0, 4, 4)$
- B: $(1, 3, 4)$
- C: $(1, 2, 3)$
- D: $(1, 3, 4)$
- F: $(2, 1, 3)$
- G: $(2, 0, 2)$

**被扩展节点：** A → C → G

---

### (c) 保证可采纳性的 $c$ 范围

可采纳要求 $\forall n: h(n) \le h^*(n)$，其中 $h^*(n)$ 为到目标 G 的真实最短代价。

| 节点 | $h(n)$ | $h^*(n)$（以 $c$ 为单位） | 可采纳条件 |
|------|--------|--------------------------|-----------|
| A | 4 | $2c$ | $4 \le 2c$ |
| B | 3 | $2c$ | $3 \le 2c$ |
| C | 2 | $c$ | $2 \le c$ |
| D | 3 | $2c$ | $3 \le 2c$ |
| E | 3 | $2c$ | $3 \le 2c$ |
| F | 1 | $c$ | $1 \le c$ |
| H | 2 | $c$ | $2 \le c$ |
| I | 1 | $c$ | $1 \le c$ |

最紧约束来自 **C** 和 **H**：$c \ge 2$。

$$
\boxed{c \ge 2}
$$

---

## Problem 2: Minimax and Alpha-Beta Pruning (13 pts)

### (a) Minimax 值

| 节点 | 计算 | 值 |
|------|------|-----|
| $A_1$ | $\max(3, 12)$ | 12 |
| $A_2$ | $\max(8, 2)$ | 8 |
| $A_3$ | $\max(4, 6)$ | 6 |
| **A** | $\min(12, 8, 6)$ | **6** |
| $B_1$ | $\max(14, 5)$ | 14 |
| $B_2$ | $\max(2, 1)$ | 2 |
| $B_3$ | $\max(7, 3)$ | 7 |
| **B** | $\min(14, 2, 7)$ | **2** |
| $C_1$ | $\max(6, 9)$ | 9 |
| $C_2$ | $\max(4, 0)$ | 4 |
| $C_3$ | $\max(10, 11)$ | 11 |
| **C** | $\min(9, 4, 11)$ | **4** |
| **Root** | $\max(6, 2, 4)$ | **6** |

---

### (b) Alpha-Beta 最终 $(\alpha, \beta)$

| 节点 | 类型 | 最终 $(\alpha, \beta)$ |
|------|------|----------------------|
| Root | MAX | $(6, +\infty)$ |
| A | MIN | $(-\infty, 6)$ |
| $A_1$ | MAX | $(12, +\infty)$ |
| $A_2$ | MAX | $(8, 12)$ |
| $A_3$ | MAX | $(6, 8)$ |
| B | MIN | $(6, 2)$ |
| $B_1$ | MAX | $(14, +\infty)$ |
| $B_2$ | MAX | $(6, 14)$ |
| $B_3$ | MAX | **NA** (pruned) |
| C | MIN | $(6, 4)$ |
| $C_1$ | MAX | $(9, +\infty)$ |
| $C_2$ | MAX | $(6, 9)$ |
| $C_3$ | MAX | **NA** (pruned) |

---

### (c) 被剪枝的分支

- **$B_3$ 下的两个叶节点**（值为 7 和 3）
- **$C_3$ 下的两个叶节点**（值为 10 和 11）

即整个 $B_3$ 与 $C_3$ 子树被剪枝。

---

### (d) MAX 最终选择的路径

Root 选子节点 **A**（值为 6）。

---

## Problem 3: CSP and AC-3 (12 pts)

### (a) AC-3 最坏时间复杂度

$$
\boxed{O(ed^3)}
$$

其中 $e$ 为弧（约束）数，$d$ 为最大域大小。

- 每条弧最多入队 $d$ 次（域大小最多从 $d$ 减到 1）。
- 每次 `REVISE` 需检查两个域的所有值对，耗时 $O(d^2)$。
- 总复杂度 $O(ed \cdot d^2) = O(ed^3)$。

---

### (b) AC-3 执行过程

初始：$D_X = D_Y = D_Z = \{1,2,3,4\}$，约束 $X < Y,\; Y < Z$。

弧队列：$[X \to Y,\; Y \to X,\; Y \to Z,\; Z \to Y]$

| 出队弧 | `REVISE` 结果 | 域变化 | 新加入队列 |
|--------|--------------|--------|-----------|
| $Y \to Z$ | $Y=4$ 无支持 | $D_Y = \{1,2,3\}$ | $X \to Y$ |
| $Z \to Y$ | $Z=1$ 无支持 | $D_Z = \{2,3,4\}$ | $Y \to Z$ |
| $X \to Y$ | $X=3,4$ 无支持 | $D_X = \{1,2\}$ | $Y \to X$ |
| $Y \to X$ | $Y=1$ 无支持 | $D_Y = \{2,3\}$ | $X \to Y,\; Z \to Y$ |
| $X \to Y$ | 无变化 | — | — |
| $Y \to Z$ | 无变化 | — | — |
| $Y \to X$ | 无变化 | — | — |
| $Z \to Y$ | $Z=2$ 无支持 | $D_Z = \{3,4\}$ | $Y \to Z$ |
| $Y \to Z$ | 无变化 | — | — |

**最终域：**

$$
\boxed{D_X = \{1,2\},\quad D_Y = \{2,3\},\quad D_Z = \{3,4\}}
$$

---

## Problem 4: Logic and CNF (12 pts)

### (a) 转换为 CNF

$$
(p \land q) \to \neg(p \leftrightarrow r)
$$

1. $p \leftrightarrow r \equiv (\neg p \lor r) \land (\neg r \lor p)$
2. $\neg(p \leftrightarrow r) \equiv (p \land \neg r) \lor (r \land \neg p)$
3. 原式 $\equiv \neg(p \land q) \lor [(p \land \neg r) \lor (r \land \neg p)]$
4. $\equiv (\neg p \lor \neg q) \lor (p \land \neg r) \lor (r \land \neg p)$

分配律化简（真值表可验证）后得到：

$$
\boxed{\neg p \lor \neg q \lor \neg r}
$$

（仅一个子句）

---

### (b) 命题逻辑与 CNF

| 规则 | 命题逻辑 | CNF |
|------|---------|-----|
| (i) $R_1, R_3$ 同选同不选 | $x_1 \leftrightarrow x_3$ | $(\neg x_1 \lor x_3) \land (\neg x_3 \lor x_1)$ |
| (ii) $R_2, R_4$ 恰好选一个 | $x_2 \oplus x_4$ | $(x_2 \lor x_4) \land (\neg x_2 \lor \neg x_4)$ |
| (iii) $R_2 \to R_1$ | $x_2 \to x_1$ | $(\neg x_2 \lor x_1)$ |

---

### (c) Unit Propagation（已知 $x_2 = \text{true}$）

1. 由 (iii)：$\neg x_2 \lor x_1 = \text{false} \lor x_1 \Rightarrow \boxed{x_1 = \text{true}}$
2. 由 (i)：$\neg x_1 \lor x_3 = \text{false} \lor x_3 \Rightarrow \boxed{x_3 = \text{true}}$
3. 由 (ii)：$\neg x_2 \lor \neg x_4 = \text{false} \lor \neg x_4 \Rightarrow \boxed{x_4 = \text{false}}$

---

## Problem 5: Perceptron and Logistic Regression (10 pts)

### (a) 感知机规则

**预测：**

$$
\hat y = \begin{cases} +1, & w^T x + b \ge 0 \\ -1, & \text{otherwise} \end{cases}
$$

**更新（当 $\hat y \neq y$ 时）：**

$$
w \leftarrow w + \eta y x,\qquad b \leftarrow b + \eta y
$$

---

### (b) 一次更新

给定 $w = (-1, 1)^T,\; b = 0,\; \eta = 0.5,\; x = (2, 1)^T,\; y = +1$：

$$
w^T x + b = (-1)(2) + (1)(1) = -1 < 0 \;\Rightarrow\; \hat y = -1 \neq y
$$

执行更新：

$$
w \leftarrow (-1, 1)^T + 0.5 \cdot (+1) \cdot (2, 1)^T = (-1, 1)^T + (1, 0.5)^T = \boxed{(0,\, 1.5)^T}
$$

$$
b \leftarrow 0 + 0.5 \cdot (+1) = \boxed{0.5}
$$

---

### (c) 逻辑回归梯度下降推导

$h_i = \sigma(z_i),\; z_i = w^T x_i + b,\; \sigma'(z) = \sigma(z)(1 - \sigma(z))$

对 $w_j$ 求偏导：

$$
\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_i \left[\frac{y_i}{h_i} - \frac{1 - y_i}{1 - h_i}\right] \frac{\partial h_i}{\partial w_j}
$$

其中 $\frac{\partial h_i}{\partial w_j} = h_i(1 - h_i)x_{ij}$，代入化简：

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_i (h_i - y_i) x_{ij}
$$

**更新规则：**

$$
\boxed{w \leftarrow w - \frac{\eta}{m} \sum_{i=1}^m (h_i - y_i) x_i,\qquad b \leftarrow b - \frac{\eta}{m} \sum_{i=1}^m (h_i - y_i)}
$$

---

## Problem 6: SVM (10 pts)

### (a) 软间隔 SVM 原问题

$$
\min_{w, b, \xi} \; \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i
$$

约束：

$$
y_i(w^T x_i + b) \ge 1 - \xi_i,\qquad \xi_i \ge 0,\quad i = 1, \dots, m
$$

---

### (b) 两点的显式约束

$(x_1, y_1) = (2, +1),\; (x_2, y_2) = (-2, -1)$：

$$
\boxed{\begin{aligned}
2w + b &\ge 1 - \xi_1 \\
2w - b &\ge 1 - \xi_2 \\
\xi_1, \xi_2 &\ge 0
\end{aligned}}
$$

---

### (c) 最优 $b$

给定 $w^* = \min(C, \frac{1}{2})$：

**Case 1：$C \ge \frac{1}{2} \Rightarrow w^* = \frac{1}{2}$**

此时 $2w^* = 1$，约束变为 $1 + b \ge 1 - \xi_1$ 与 $1 - b \ge 1 - \xi_2$。
最小化 $\xi_1 + \xi_2 = |b|$，最小值在 $b = 0$。

$$
\boxed{b^* = 0}
$$

**Case 2：$0 < C < \frac{1}{2} \Rightarrow w^* = C$**

此时 $1 - 2C > 0$。当 $|b| \le 1 - 2C$ 时，$\xi_1 + \xi_2 = 2(1 - 2C)$ 为常数且最小。

$$
\boxed{b^* \in [\,2C - 1,\; 1 - 2C\,]}
$$

---

## Problem 7: Naive Bayes (12 pts)

数据：$Y = +$（ID 1, 2, 5, 7）共 4 例；$Y = -$（ID 3, 4, 6, 8）共 4 例。Laplace $\alpha = 1$。

### (a) 类先验

$$
\boxed{P(Y = +) = \frac{1}{2},\qquad P(Y = -) = \frac{1}{2}}
$$

---

### (b) 条件概率

对 $Y = +$（分母 $4 + 2 = 6$）：

| $X_j$ | $X_j = 1$ | $X_j = 0$ |
|-------|-----------|-----------|
| $X_1$ | 5/6 | 1/6 |
| $X_2$ | 3/6 = 1/2 | 3/6 = 1/2 |
| $X_3$ | 3/6 = 1/2 | 3/6 = 1/2 |
| $X_4$ | 1/6 | 5/6 |

对 $Y = -$（分母 6）：

| $X_j$ | $X_j = 1$ | $X_j = 0$ |
|-------|-----------|-----------|
| $X_1$ | 1/6 | 5/6 |
| $X_2$ | 3/6 = 1/2 | 3/6 = 1/2 |
| $X_3$ | 3/6 = 1/2 | 3/6 = 1/2 |
| $X_4$ | 5/6 | 1/6 |

---

### (c) 预测 $x = (1, 0, 1, 0)$

$$
\begin{aligned}
P(Y = + \mid x) &\propto \frac{1}{2} \cdot \frac{5}{6} \cdot \frac{1}{2} \cdot \frac{1}{2} \cdot \frac{5}{6} = \frac{25}{288} \\[4pt]
P(Y = - \mid x) &\propto \frac{1}{2} \cdot \frac{1}{6} \cdot \frac{1}{2} \cdot \frac{1}{2} \cdot \frac{1}{6} = \frac{1}{288}
\end{aligned}
$$

$Y = +$ 得分远大于 $Y = -$，故预测为 $\boxed{+}$。

---

### (d) 不使用独立性假设

- 4 个二进制特征 + 1 个二进制标签，完整联合分布有 $2^5 = \boxed{32}$ 种组合。
- 概率和为 1 的约束下，独立参数个数为 $\boxed{31}$。

---

## Problem 8: Bayesian Networks and Sampling (14 pts)

### (a) 联合分布分解

$$
\boxed{P(A, B, C, D) = P(A) \, P(B \mid A) \, P(C \mid B) \, P(D \mid C)}
$$

---

### (b) Prior Sampling 估计 $P(C)$

8 个样本中 $C$ 出现：#2, #3, #4, #6, #8 $\Rightarrow$ 5 次。

$$
\boxed{P(C) \approx \frac{5}{8}}
$$

---

### (c) Rejection Sampling 估计 $P(C \mid A, \neg D)$

保留满足 $A = \text{true}$ 且 $D = \text{false}$ 的样本：

- #1 $(A, B, \neg C, \neg D)$ ✓
- #2 $(A, \neg B, C, \neg D)$ ✓
- #6 $(A, B, C, \neg D)$ ✓

共保留 **3 个**样本；其中 $C$ 为真：#2, #6（2 个）。

$$
\boxed{P(C \mid A, \neg D) \approx \frac{2}{3}}
$$

**被保留样本：** #1, #2, #6

---

### (d) Likelihood Weighting 权重（证据 $B, \neg D$）

权重 $w = P(B \mid A) \cdot P(\neg D \mid C)$：

| 样本 | 计算 | 权重 |
|------|------|------|
| $\neg A, B, C, \neg D$ | $\frac{3}{4} \times \frac{5}{6}$ | $\boxed{\frac{5}{8}}$ |
| $A, B, C, \neg D$ | $\frac{1}{5} \times \frac{5}{6}$ | $\boxed{\frac{1}{6}}$ |
| $A, B, \neg C, \neg D$ | $\frac{1}{5} \times \frac{1}{8}$ | $\boxed{\frac{1}{40}}$ |
| $\neg A, B, \neg C, \neg D$ | $\frac{3}{4} \times \frac{1}{8}$ | $\boxed{\frac{3}{32}}$ |

---

### (e) 加权估计 $P(\neg A \mid B, \neg D)$

总权重：

$$
W = \frac{5}{8} + \frac{1}{6} + \frac{1}{40} + \frac{3}{32} = \frac{437}{480}
$$

$\neg A$ 的权重和（样本 #1 和 #4）：

$$
W_{\neg A} = \frac{5}{8} + \frac{3}{32} = \frac{345}{480}
$$

$$
\boxed{P(\neg A \mid B, \neg D) \approx \frac{345}{437} = \frac{15}{19}}
$$

---

## Problem 9: Genetic Algorithm and Course Projects (12 pts)

### (a) 公平对比 GA 与反向传播

1. **相同架构**：层数、神经元数、激活函数一致。
2. **相同数据与划分**：训练/验证/测试集相同。
3. **相同评估指标**：如测试集准确率、交叉熵损失。
4. **相同计算预算**：总训练时间或总前向传播次数相当。
5. **公平的超参数调优**：两种方法都用验证集调参。
6. **多次随机运行取平均**：消除随机性带来的方差。

---

### (b) Project 1 的 GA 设计

- **State / Individual**：二进制染色体表示选中的 $S_1$ 和 $S_2$ 节点，满足 $|S_1| + |S_2| = k$（可用修复算子保证预算约束）。
- **Fitness Function**：通过 **Monte Carlo 模拟**（运行 30~200 次独立级联扩散）估计目标函数 $E[|V| - |r_1 \oplus r_2|]$。
- **Stopping Criterion**：达到最大代数（如 100 代）；或连续多代最优适应度不再提升；或达到时间上限。

---

### (c) AUC 与 nDCG@5

| 指标 | 衡量内容 |
|------|---------|
| **AUC** | 全局排序能力。随机取一正一负样本，正样本排在负样本前的概率。 |
| **nDCG@5** | Top-5 推荐列表质量。考虑相关性分级，位置越靠前权重越高（折扣因子）。 |

**为何可能一升一降：**

- **AUC 关注全局排序**，**nDCG@5 只关注头部 5 个位置**。
- 若模型改善了尾部 item 的相对顺序但对 Top-5 无影响 $\Rightarrow$ AUC 升、nDCG@5 不变。
- 若模型过度优化头部（如利用流行度偏差），Top-5 更相关 $\Rightarrow$ nDCG@5 升，但全局排序可能变差 $\Rightarrow$ AUC 降。
