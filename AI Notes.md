# AI(H) 期末复习笔记

> 根据 `ppts/AI Review(1).pdf`、样卷 `ppts/Exam Paper (3).pdf`、`ppts/Logic and Bayesian Questions(1).pdf` 和 `ppts/AI_Lec01-Lec12` 整理。  
> 目标：精炼复习、公式清楚、做题时可以直接套模板。

---

## 0. 考试重点预测

### 高概率考点：样卷中直接出现

| 优先级 | 主题 | 必须会做什么 |
|---|---|---|
| 高 | Search 搜索 | 画 BFS 搜索树；画 A* 搜索树；判断 heuristic 在边权 $c$ 下是否 admissible。 |
| 高 | Minimax / Alpha-Beta | 填每个节点最终 $\alpha/\beta$；标出剪枝分支；标出最终选择路径。 |
| 高 | CSP / AC-3 | 写 AC-3 最坏复杂度；逐步展示 domain pruning。 |
| 高 | Logic / CNF | 把命题逻辑公式化为 CNF。 |
| 高 | Logic / DPLL / Resolution | 命题逻辑建模、CNF、DPLL 推理；一阶逻辑转 CNF 并用 resolution 证明。 |
| 高 | Perceptron / Logistic Regression | 写感知机更新公式；推导逻辑回归梯度更新。 |
| 高 | SVM | 写 soft-margin SVM 原始优化问题和约束；会做简单一维例题。 |
| 高 | Naive Bayes | 计算先验/条件概率；预测类别；解释独立性假设。 |
| 高 | Bayesian Network | 画 BN；用 Bayes rule 计算后验；会 prior/rejection/likelihood weighting/Gibbs sampling。 |
| 中高 | Genetic Algorithm | 设计公平实验，比较 GA 训练神经网络和 backpropagation。 |

### 中概率考点：Review PDF 明确提到

- Confusion matrix：accuracy、precision、recall、F1。
- ROC/AUC 和 threshold 选择。
- Decision Tree：entropy、information gain。
- K-means、PCA/LLE 基本流程。
- SVM kernel trick、training metric vs performance metric。
- Generalization 估计：random split、cross-validation、bootstrap。
- Hyperparameter tuning：grid search、heuristic/black-box search。
- Bayesian Network：条件独立和推理。
- Recommender System：content-based、collaborative filtering、Pearson correlation、matrix factorization。
- Knowledge Graph：RDF triples、entity recognition、relation extraction、completion。

---

## 1. Search 搜索

### 1.1 搜索问题的定义

一个搜索问题通常包括：

- Initial state：初始状态。
- Actions $Actions(s)$：在状态 $s$ 可以做的动作。
- Transition model $Result(s,a)$：执行动作后的状态。
- Goal test：判断是否到达目标。
- Path cost $g(n)$：从起点到当前节点的代价。

常用符号：

- $b$：branching factor，分支因子。
- $d$：最浅目标节点深度。
- $m$：最大搜索深度。
- $C^*$：最优解总代价。
- $h(n)$：从 $n$ 到目标的启发式估计。
- $h^*(n)$：从 $n$ 到目标的真实最小代价。

### 1.2 BFS

规则：优先展开最浅的节点。如果有多个节点同层，按题目要求的顺序，例如 alphabetical order。

性质：

| 算法 | Complete | Optimal | Time | Space |
|---|---:|---:|---:|---:|
| BFS | 是，若 $b$ 有限 | 是，若每步代价相同 | $O(b^d)$ | $O(b^d)$ |

画 BFS 搜索树模板：

1. 把起点放进 frontier。
2. 每次取出最早进入 frontier 的最浅节点。
3. 按题目 tie-breaking 顺序加入子节点。
4. 按题目要求，在 goal 被生成或被取出时停止。
5. 注意画的是 search tree，不一定是原图；tree search 中同一个 graph node 可能重复出现。

### 1.3 UCS、Greedy、A*

| 算法 | Evaluation function | 含义 |
|---|---|---|
| Uniform Cost Search | $f(n)=g(n)$ | 展开当前路径代价最小的节点。 |
| Greedy Best-First | $f(n)=h(n)$ | 展开估计离目标最近的节点。 |
| A* | $f(n)=g(n)+h(n)$ | 展开估计总代价最小的节点。 |

A* 三个核心量：

- $g(n)$：从起点到 $n$ 的真实代价。
- $h(n)$：从 $n$ 到目标的估计代价。
- $f(n)=g(n)+h(n)$：经过 $n$ 到目标的估计总代价。

Admissible heuristic：

$$
h(n) \le h^*(n), \quad \forall n
$$

意思：启发函数永远不能高估真实最短代价。

Consistent heuristic：

$$
h(n) \le c(n,a,n') + h(n'), \quad \forall (n,n')
$$

重要结论：

- Consistent 一定 admissible。
- A* tree search 在 $h$ admissible 时 optimal。
- A* graph search 在 $h$ consistent 时 optimal。
- 如果 $h_2(n) \ge h_1(n)$ 对所有 $n$ 成立，且二者都 admissible，则 $h_2$ dominates $h_1$，通常更高效。

### 1.4 样卷题型：求边权 $c$ 的范围保证 admissible

如果每条边代价都是 $c$，从节点 $n$ 到目标的最少边数是 $dist\_edges(n,G)$，则：

$$
h^*(n) = c \cdot dist\_edges(n,G)
$$

要保证 admissible：

$$
h(n) \le c \cdot dist\_edges(n,G), \quad \forall n
$$

所以：

$$
c \ge \frac{h(n)}{dist\_edges(n,G)}, \quad \forall n \ne G
$$

最终答案写成：

$$
c \ge \max_{n \ne G} \frac{h(n)}{dist\_edges(n,G)}
$$

还要检查：

$$
h(G)=0
$$

如果 $h(G)>0$，则对任何正的 $c$ 都不 admissible。

### 1.5 Beyond Classical Search：solution space search

Classical search 通常用 search tree；beyond classical search 更常直接在 solution space 里搜索。

通用框架：

1. 生成初始解 $x^{(0)}$。
2. 计算 objective/evaluation $f(x^{(0)})$。
3. 用 search operator 生成新解 $x'$。
4. 计算 $f(x')$。
5. 用 replacement criterion 决定是否接受 $x'$。
6. 重复直到满足停止条件。

常见 solution representations：

- Continuous vector。
- Binary string。
- Integer vector。
- Permutation。

常见 search operators：

- Binary string：flip one bit。
- Permutation：swap two elements。
- Continuous vector：Gaussian perturbation。

Local search / hill climbing：

$$
\text{if } f(x')>f(x^{(t)}),\quad x^{(t+1)}=x'
$$

Simulated annealing：

若 $x'$ 更差，也可能接受：

$$
P(\text{accept})=e^{\Delta E/T}
$$

其中 $T$ 是 temperature。$T$ 高时更随机，$T$ 低时更贪心。

Tabu Search：

- 用 tabu list 记录最近访问过的解或操作。
- 避免短期内回到旧状态。
- 主要用于逃离 local optimum 和循环。

Gradient-based optimization：

若目标函数连续可导，可以沿梯度方向更新：

$$
x \leftarrow x-\eta \nabla f(x)
$$

若是最大化问题，则：

$$
x \leftarrow x+\eta \nabla f(x)
$$

Quadratic Programming / convexity：

- 若 objective 是 quadratic，constraints 是 linear，就是 QP。
- 若无约束且 convex，可以令导数为 0，直接解线性系统。
- 判断 convex 的常见充分条件：quadratic matrix $Q$ positive definite。

---

## 2. Minimax 和 Alpha-Beta Pruning

### 2.1 Minimax

适用于 deterministic、two-player、zero-sum、perfect-information games。

规则：

$$
V_{\text{MAX}}(n)=\max_{c \in Children(n)} V(c)
$$

$$
V_{\text{MIN}}(n)=\min_{c \in Children(n)} V(c)
$$

最终 root 选择对 MAX 最有利的 child。

### 2.2 Alpha 和 beta 的含义

- $\alpha$：当前路径上，MAX 已经能保证的最好值。
- $\beta$：当前路径上，MIN 已经能保证的最好值。
- 剪枝条件：

$$
\alpha \ge \beta
$$

### 2.3 做题模板

1. 通常按从左到右访问叶子节点，除非题目指定其他顺序。
2. MAX 节点：
   - 初始 $v=-\infty$。
   - 每看一个 child，更新 $v=\max(v, child\_value)$。
   - 更新 $\alpha=\max(\alpha,v)$。
3. MIN 节点：
   - 初始 $v=+\infty$。
   - 每看一个 child，更新 $v=\min(v, child\_value)$。
   - 更新 $\beta=\min(\beta,v)$。
4. 一旦 $\alpha \ge \beta$，剩余 child 剪掉。
5. 题目若要求给 alpha/beta，被剪枝节点写 `NA`。
6. 最终路径：从 root 开始，每层选值等于父节点最终值的 child。

易错点：

- Alpha 主要在 MAX 节点更新。
- Beta 主要在 MIN 节点更新。
- 剪枝不改变 minimax 值。
- 剪枝结果依赖子节点访问顺序。

---

## 3. CSP 和 AC-3

### 3.1 CSP 定义

CSP 是三元组：

$$
CSP=(X,D,C)
$$

- $X=\{X_1,\ldots,X_n\}$：变量。
- $D=\{D_1,\ldots,D_n\}$：每个变量的 domain。
- $C$：约束条件。

答题格式：

$$
X=\{X,Y,Z\}
$$

$$
D_X=D_Y=D_Z=\{1,2,3\}
$$

$$
C=\{X<Y,\;Y<Z\}
$$

### 3.2 AC-3

弧 $(X_i,X_j)$ arc-consistent 的含义：

$$
\forall x \in D_i,\; \exists y \in D_j
$$

使得 $X_i$ 和 $X_j$ 之间的约束成立。

AC-3 最坏复杂度：

$$
O(ed^3)
$$

其中：

- $e$：arc 或 constraint 的数量，具体看课程记法。
- $d$：最大 domain 大小。

为什么是 $O(ed^3)$：

- $Revise(X_i,X_j)$ 最多检查 $d$ 个 $X_i$ 的值。
- 每个值最多和 $d$ 个 $X_j$ 的值比较。
- 一次 revise 是 $O(d^2)$。
- 每条 arc 可能因为 domain 删除被重新加入最多 $O(d)$ 次。
- 所以总复杂度是 $O(ed^3)$。

### 3.3 样卷 AC-3 例题

已知：

$$
D_X=D_Y=D_Z=\{1,2,3\}
$$

$$
C=\{X<Y,\;Y<Z\}
$$

初始 arcs：

$$
(X,Y),\;(Y,X),\;(Y,Z),\;(Z,Y)
$$

逐步 pruning：

1. Revise $(X,Y)$，约束 $X<Y$：
   - $X=3$ 找不到 $Y>3$。
   - $D_X=\{1,2\}$。
2. Revise $(Y,X)$，约束 $X<Y$：
   - $Y=1$ 找不到 $X<1$。
   - $D_Y=\{2,3\}$。
3. Revise $(Y,Z)$，约束 $Y<Z$：
   - $Y=3$ 找不到 $Z>3$。
   - $D_Y=\{2\}$。
4. 因为 $D_Y$ 变化，需要重新检查相关 arcs。
5. Revise $(Z,Y)$，约束 $Y<Z$：
   - 现在 $D_Y=\{2\}$，只有 $Z=3$ 可行。
   - $D_Z=\{3\}$。
6. 再 Revise $(X,Y)$：
   - 现在 $D_Y=\{2\}$，只有 $X=1$ 可行。
   - $D_X=\{1\}$。

最终 domains：

$$
D_X=\{1\},\quad D_Y=\{2\},\quad D_Z=\{3\}
$$

### 3.4 Backtracking 常用启发式

- MRV：Minimum Remaining Values，先选合法取值最少的变量。
- Degree heuristic：若 MRV 打平，选参与约束最多的变量。
- LCV：Least Constraining Value，先选对其他变量限制最少的值。

---

## 4. Logic 和 CNF

### 4.1 必背等价式

Implication：

$$
p \to q \equiv \neg p \lor q
$$

Double negation：

$$
\neg \neg p \equiv p
$$

De Morgan：

$$
\neg(p \land q) \equiv \neg p \lor \neg q
$$

$$
\neg(p \lor q) \equiv \neg p \land \neg q
$$

Distributive laws：

$$
p \lor (q \land r) \equiv (p \lor q) \land (p \lor r)
$$

$$
p \land (q \lor r) \equiv (p \land q) \lor (p \land r)
$$

CNF 是 AND of OR-clauses：

$$
(a \lor b \lor \neg c) \land (\neg a \lor d)
$$

### 4.2 CNF 转换模板

1. 消去 $\leftrightarrow$ 和 $\to$。
2. 用 De Morgan 把 $\neg$ 推到最里面。
3. 分配律：把 OR 分配到 AND 上。
4. 化简重复项或吸收项。

### 4.3 样卷公式

题目：

$$
p \to \neg(p \lor q)
$$

转换：

$$
\begin{aligned}
p \to \neg(p \lor q)
&\equiv \neg p \lor \neg(p \lor q) \\
&\equiv \neg p \lor (\neg p \land \neg q) \\
&\equiv \neg p
\end{aligned}
$$

CNF：

$$
\neg p
$$

### 4.4 命题逻辑建模例题：排球队

来自 `Logic and Bayesian Questions(1).pdf` 的题型。设：

$$
x_i =
\begin{cases}
True, & \text{player } i \text{ plays}\\
False, & \text{player } i \text{ does not play}
\end{cases}
$$

规则：

1. Players 4 and 6 need to play together：

$$
x_4 \leftrightarrow x_6
$$

2. Player 3 does not play iff player 1 does not play：

$$
\neg x_3 \leftrightarrow \neg x_1
$$

等价于：

$$
x_3 \leftrightarrow x_1
$$

3. Either player 3 or player 6 appears, but not both：

$$
(x_3 \lor x_6)\land \neg(x_3\land x_6)
$$

也就是 XOR：

$$
x_3 \oplus x_6
$$

4. If players 9 and 12 play, then player 4 must play：

$$
(x_9\land x_{12})\to x_4
$$

5. 题目额外给定 players 1 and 12 play：

$$
x_1,\quad x_{12}
$$

### 4.5 排球队例题转 CNF

双条件：

$$
x_4 \leftrightarrow x_6
\equiv
(\neg x_4\lor x_6)\land(\neg x_6\lor x_4)
$$

$$
x_3 \leftrightarrow x_1
\equiv
(\neg x_3\lor x_1)\land(\neg x_1\lor x_3)
$$

XOR：

$$
x_3 \oplus x_6
\equiv
(x_3\lor x_6)\land(\neg x_3\lor \neg x_6)
$$

Implication：

$$
(x_9\land x_{12})\to x_4
\equiv
\neg x_9\lor \neg x_{12}\lor x_4
$$

加上 evidence：

$$
x_1,\quad x_{12}
$$

完整 CNF：

$$
\begin{aligned}
&(\neg x_4\lor x_6)
\land(\neg x_6\lor x_4)
\land(\neg x_3\lor x_1)
\land(\neg x_1\lor x_3)\\
&\land(x_3\lor x_6)
\land(\neg x_3\lor \neg x_6)
\land(\neg x_9\lor \neg x_{12}\lor x_4)
\land x_1
\land x_{12}
\end{aligned}
$$

### 4.6 DPLL 推理模板

DPLL 常用三件事：

- Unit clause propagation：如果有单子句 $x$，就令 $x=True$。
- Pure symbol heuristic：如果某个符号只以一种极性出现，可以直接赋值满足它。
- Splitting：如果推不动，选一个变量分支尝试 True/False。

排球队例题中：

1. 由 unit clause 得：

$$
x_1=True,\quad x_{12}=True
$$

2. 由 $(\neg x_1\lor x_3)$ 和 $x_1=True$ 得：

$$
x_3=True
$$

3. 由 $(\neg x_3\lor \neg x_6)$ 和 $x_3=True$ 得：

$$
x_6=False
$$

4. 由 $(\neg x_6\lor x_4)$ 与 $(\neg x_4\lor x_6)$ 表示 $x_4\leftrightarrow x_6$，所以：

$$
x_4=False
$$

5. 由 $\neg x_9\lor \neg x_{12}\lor x_4$，以及 $x_{12}=True,\ x_4=False$ 得：

$$
x_9=False
$$

结论：如果 players 1 and 12 必须上场，则其他四人中：

$$
x_3=True,\quad x_4=False,\quad x_6=False,\quad x_9=False
$$

也就是只有 player 3 应该上场。

### 4.7 FOL 推理基础：UI、EI、Unification

Universal Instantiation, UI：

如果有：

$$
\forall x,\; P(x)
$$

可以代入任意常量 $a$：

$$
P(a)
$$

Existential Instantiation, EI：

如果有：

$$
\exists x,\; P(x)
$$

可以引入一个新的常量 $k$：

$$
P(k)
$$

注意：$k$ 必须是新的名字，不能随便用已经存在的对象。

Unification：

Unification 是找 substitution，让两个表达式变成一样。

例子：

$$
Knows(John,x)
$$

和：

$$
Knows(y,Mary)
$$

可以用 substitution：

$$
\{x/Mary,\ y/John\}
$$

统一成：

$$
Knows(John,Mary)
$$

Resolution in FOL = propositional resolution + unification。

### 4.8 一阶逻辑 Resolution 例题

题目类型：把自然语言转 FOL，再用 resolution 证明 ZhangSan is Happy。

谓词定义：

- $PassAI(x)$：$x$ passes the AI exam。
- $Prize(x)$：$x$ wins a prize。
- $Happy(x)$：$x$ is happy。
- $Willing(x)$：$x$ is willing to learn。
- $Lucky(x)$：$x$ is lucky。
- 常量 $z$：ZhangSan。

知识库：

1. Anyone who passes the AI exam and wins a prize is happy：

$$
\forall x,\; PassAI(x)\land Prize(x)\to Happy(x)
$$

2. Anyone willing to learn or lucky can pass all exams：

$$
\forall x,\; Willing(x)\lor Lucky(x)\to PassAI(x)
$$

3. ZhangSan is not willing to learn but lucky：

$$
\neg Willing(z),\quad Lucky(z)
$$

4. Any lucky person can win a prize：

$$
\forall x,\; Lucky(x)\to Prize(x)
$$

要证明：

$$
Happy(z)
$$

Resolution 做法：加入反证目标：

$$
\neg Happy(z)
$$

转 CNF：

$$
\neg PassAI(x)\lor \neg Prize(x)\lor Happy(x)
$$

$$
\neg Willing(x)\lor PassAI(x)
$$

$$
\neg Lucky(x)\lor PassAI(x)
$$

$$
\neg Lucky(x)\lor Prize(x)
$$

$$
\neg Willing(z)
$$

$$
Lucky(z)
$$

$$
\neg Happy(z)
$$

Resolution 推导：

1. 由 $Lucky(z)$ 和 $\neg Lucky(x)\lor PassAI(x)$ 得：

$$
PassAI(z)
$$

2. 由 $Lucky(z)$ 和 $\neg Lucky(x)\lor Prize(x)$ 得：

$$
Prize(z)
$$

3. 由 $PassAI(z)$ 和 $\neg PassAI(x)\lor \neg Prize(x)\lor Happy(x)$ 得：

$$
\neg Prize(z)\lor Happy(z)
$$

4. 由 $Prize(z)$ 和 $\neg Prize(z)\lor Happy(z)$ 得：

$$
Happy(z)
$$

5. 与反证目标 $\neg Happy(z)$ 矛盾，得到 empty clause：

$$
\Box
$$

所以：

$$
KB\models Happy(z)
$$

---

## 5. Perceptron 和 Logistic Regression

### 5.0 Machine Learning 基本原则

Machine Learning 可以看成：

$$
\text{Learning}=\text{search in hypothesis/model space}
$$

三件核心事：

- Representation：数据和模型怎么表示。
- Algorithm：怎么搜索/优化模型。
- Evaluation：怎么衡量模型好坏。

Lecture 中的总结：

$$
Representation + Algorithm + Evaluation = Model
$$

Generalization 是核心目标：模型不仅要在 training data 上表现好，也要在 unseen data 上表现好。

Overfitting：

- 训练集表现很好。
- 测试集/真实环境表现差。
- 原因通常是模型太复杂，记住了噪声。

Occam's Razor：

> 在能解释数据的前提下，模型应尽量简单。

MAP 分类思想：

$$
\hat y=\arg\max_{\omega_i}P(\omega_i\mid x)
$$

由 Bayes rule：

$$
P(\omega_i\mid x)=\frac{p(x\mid \omega_i)P(\omega_i)}{p(x)}
$$

因为 $p(x)$ 对所有类别相同，所以：

$$
\hat y=\arg\max_{\omega_i}p(x\mid \omega_i)P(\omega_i)
$$

Parametric vs nonparametric：

- Parametric：假设数据服从某个分布，比如 Gaussian，再估计分布参数。
- Nonparametric：不强行假设固定分布形式，通常有更多需要调的超参数。

### 5.1 Perceptron

若标签是 $\{-1,+1\}$：

$$
\hat y = sign(w^T x+b)
$$

只有分类错误时更新：

$$
\text{if } y(w^T x+b)\le 0:
\quad
w \leftarrow w+\eta yx,\quad b \leftarrow b+\eta y
$$

若标签是 $\{0,1\}$，常见写法：

$$
w \leftarrow w+\eta(y-\hat y)x
$$

$$
b \leftarrow b+\eta(y-\hat y)
$$

### 5.2 Logistic Regression

Sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

模型：

$$
h_w(x)=\sigma(w^T x+b)=P(y=1\mid x)
$$

Binary cross-entropy loss：

$$
J(w,b)
=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i\log h_i+(1-y_i)\log(1-h_i)
\right]
$$

其中：

$$
h_i=\sigma(w^T x_i+b)
$$

关键导数：

$$
\frac{d\sigma(z)}{dz}=\sigma(z)(1-\sigma(z))
$$

Gradient：

$$
\frac{\partial J}{\partial w}
=\frac{1}{m}\sum_{i=1}^{m}(h_i-y_i)x_i
$$

$$
\frac{\partial J}{\partial b}
=\frac{1}{m}\sum_{i=1}^{m}(h_i-y_i)
$$

Gradient descent 更新：

$$
w \leftarrow w-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_i-y_i)x_i
$$

$$
b \leftarrow b-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_i-y_i)
$$

推导题答题骨架：

1. 令 $z_i=w^T x_i+b$，$h_i=\sigma(z_i)$。
2. 写出 cross-entropy loss。
3. 使用结论：

$$
\frac{\partial loss_i}{\partial z_i}=h_i-y_i
$$

4. 链式法则：

$$
\frac{\partial z_i}{\partial w}=x_i,
\quad
\frac{\partial z_i}{\partial b}=1
$$

5. 对所有样本求和并取平均。

---

## 6. SVM

### 6.1 Hard-margin SVM

标签 $y_i\in\{-1,+1\}$：

$$
\min_{w,b}\ \frac{1}{2}\lVert w\rVert^2
$$

subject to：

$$
y_i(w^T x_i+b)\ge 1,\quad i=1,\ldots,m
$$

Margin width：

$$
\frac{2}{\lVert w\rVert}
$$

### 6.2 Kernel Trick

SVM 的优化和预测主要依赖 dot product：

$$
x_i^T x_j
$$

Kernel trick 用 kernel function 替代显式高维映射后的点积：

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

作用：

- 不显式计算 $\phi(x)$，也能在高维 feature space 中做线性分类。
- 在原始空间中表现为 nonlinear decision boundary。

常见 kernel：

$$
K(x,z)=x^Tz
$$

$$
K(x,z)=(x^Tz+c)^d
$$

$$
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)
$$

分别是 linear、polynomial、RBF kernel。

### 6.3 Soft-margin SVM

Primal objective：

$$
\min_{w,b,\xi}\ \frac{1}{2}\lVert w\rVert^2+C\sum_{i=1}^{m}\xi_i
$$

subject to：

$$
y_i(w^T x_i+b)\ge 1-\xi_i
$$

$$
\xi_i\ge 0
$$

等价 hinge loss 形式：

$$
\min_{w,b}\ \frac{1}{2}\lVert w\rVert^2
+C\sum_{i=1}^{m}\max\left(0,1-y_i(w^T x_i+b)\right)
$$

$C$ 的含义：

- $C$ 大：更重视惩罚分类错误或 margin violation，regularization 较弱。
- $C$ 小：允许更多 violation，regularization 较强。

### 6.4 样卷一维 SVM

两个点：

$$
(x_1,y_1)=(1,1),\quad (x_2,y_2)=(-1,-1)
$$

Soft-margin primal：

$$
\min_{w,b,\xi_1,\xi_2}\ \frac{1}{2}w^2+C(\xi_1+\xi_2)
$$

subject to：

$$
w+b\ge 1-\xi_1
$$

$$
w-b\ge 1-\xi_2
$$

$$
\xi_1,\xi_2\ge 0
$$

如果题目给出最优：

$$
w^*=\min(C,1)
$$

则：

- 若 $C\ge 1$，$w^*=1$，约束给出 $b\ge 0$ 且 $b\le 0$，所以：

$$
b^*=0
$$

- 若 $0<C<1$，$w^*=C$，使 hinge loss 最小的 $b$ 可以是：

$$
C-1\le b^*\le 1-C
$$

所以：

$$
b^*\in [C-1,1-C]
$$

---

## 7. Naive Bayes

### 7.1 Bayes theorem

$$
P(Y\mid X)=\frac{P(X\mid Y)P(Y)}{P(X)}
$$

分类时，分母 $P(X)$ 对所有类别一样，所以：

$$
\hat y=\arg\max_y P(y)P(x\mid y)
$$

Naive independence assumption：

$$
P(x_1,x_2,\ldots,x_n\mid y)
=\prod_{j=1}^{n}P(x_j\mid y)
$$

预测公式：

$$
\hat y=\arg\max_y P(y)\prod_{j=1}^{n}P(x_j\mid y)
$$

为了避免连乘下溢，也可以用 log：

$$
\hat y=\arg\max_y
\left[
\log P(y)+\sum_{j=1}^{n}\log P(x_j\mid y)
\right]
$$

### 7.2 Laplace smoothing

对于 categorical feature value $x_j=v$：

$$
P(x_j=v\mid y)
=\frac{count(x_j=v,y)+\alpha}{count(y)+\alpha K}
$$

其中：

- $K$：feature $j$ 的可能取值个数。
- 通常 $\alpha=1$。

### 7.3 做题模板

1. 计算 class priors：

$$
P(Y=y)=\frac{count(Y=y)}{N}
$$

2. 对每个 feature、每个 class，计算条件概率：

$$
P(X_j=v\mid Y=y)
$$

3. 对每个测试点，分别计算每个类别的 score：

$$
score(y)=P(y)\prod_{j=1}^{n}P(x_j\mid y)
$$

4. 选择 score 最大的类别。
5. 如果出现 0 概率，使用 Laplace smoothing。

### 7.4 如果独立性假设不成立

Naive Bayes 估计的是：

$$
P(x_1\mid y)P(x_2\mid y)\cdots P(x_n\mid y)
$$

如果没有 naive assumption，需要估计完整 joint：

$$
P(x_1,x_2,\ldots,x_n\mid y)
$$

如果有 4 个 binary features 和 1 个 binary label：

- Feature 组合数：$2^4=16$。
- 加上 binary label 后的 feature-label 组合数：$2\cdot 2^4=32$。
- 5 个 binary variables 的 full joint 独立参数数：$2^5-1=31$。

如果题目问“至少需要多少数据点覆盖所有 feature-label 组合”，答：

$$
32
$$

如果问“feature patterns 有多少种”，答：

$$
16
$$

答题时要说明你采用的是哪个理解。

---

## 8. Genetic Algorithm

### 8.1 基本流程

1. 把候选解编码成 individual/chromosome。
2. 初始化 population。
3. 计算 fitness。
4. 选择 parents。
5. 通过 crossover 产生 offspring。
6. 以小概率 mutation。
7. 选择 survivors，形成下一代 population。
8. 到达收敛、最大代数或时间预算后停止。

### 8.2 关键词

- Individual：一个候选解。
- Population：一组候选解。
- Fitness：要最大化的评价分数。
- Selection：好个体更大概率被选中。
- Crossover：父代组合生成子代。
- Mutation：随机小改动。
- Elitism：保留最优秀个体。

### 8.3 样卷题型：GA vs Backprop 训练神经网络

公平比较 checklist：

1. 使用相同 dataset split：train/validation/test 完全一致。
2. 使用相同 neural network architecture。
3. 使用相同 preprocessing 和 input features。
4. 使用相同 evaluation metric，例如 accuracy、F1、test loss。
5. 给相同 compute budget 或 wall-clock time budget。
6. 两种方法都在 validation set 上调 hyperparameters。
7. 多个 random seeds 重复实验，报告 mean 和 standard deviation。
8. 最终只在 untouched test set 上做一次最终评价。

可以直接写的答案：

```text
I would compare GA and backpropagation on the same fixed neural network architecture
and the same train/validation/test split.
For GA, one individual is the flattened vector of all network weights.
Fitness is validation accuracy or negative validation loss.
For backpropagation, I train the same architecture using SGD/Adam.
Both methods receive the same compute or time budget and are repeated over multiple random seeds.
Finally, I report mean and standard deviation on the same test set.
```

---

## 9. Evaluation Metrics

Confusion matrix：

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

公式：

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
$$

$$
TPR=Recall=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{FP+TN}
$$

什么时候用：

- Accuracy：类别比较均衡。
- Precision：false positive 代价高。
- Recall：false negative 代价高。
- F1：需要平衡 precision 和 recall。

ROC：

- X-axis：$FPR$。
- Y-axis：$TPR$。
- 每个 threshold 对应 ROC 上一个点。
- AUC 衡量排序能力；$1.0$ 完美，$0.5$ 随机猜。

Threshold 选择：

- 选择靠近左上角 $(FPR=0,TPR=1)$ 的点。
- 如果 false positives 代价高，threshold 取高一些。
- 如果 false negatives 代价高，threshold 取低一些。

### 9.1 估计 Generalization

Generalization performance 是随机变量，不能只看一次训练结果。

常见估计方法：

- Random split：随机划分 train/test。
- Cross-validation：例如 $k$-fold CV，轮流用一部分做 validation。
- Bootstrap：有放回采样构造多个训练集。

报告结果时最好写：

$$
mean \pm std
$$

也就是多次实验的平均值和标准差。

注意：

- Training objective 不一定等于真正关心的 performance metric。
- 比如训练时优化 cross-entropy，但最终关心 F1 或 AUC。
- 选择 metric 时要和 user requirement 一致。

### 9.2 Hyperparameter Tuning

Hyperparameters 是训练前要设定的参数，例如：

- SVM：kernel type、kernel parameters、regularization parameter $C$。
- Neural Network：hidden nodes、activation、architecture、learning rate。
- Decision Tree：tree depth、branching factor。
- K-means：number of clusters $k$。

调参没有解析解，因为每个 hyperparameter 的效果要通过 generalization performance 估计。

常见方法：

- Grid search：枚举组合，简单但贵。
- Random search：随机试，比 grid search 更省。
- Heuristic / black-box optimization：把调参看成黑盒优化问题。

标准流程：

1. 在 training set 上训练模型。
2. 在 validation set 上选 hyperparameters。
3. 最后只用 test set 做最终报告。

---

## 10. Decision Trees

Entropy：

$$
H(S)=-\sum_i p_i\log_2(p_i)
$$

Information gain：

$$
IG(S,A)=H(S)-\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}H(S_v)
$$

Gini impurity：

$$
Gini(S)=1-\sum_i p_i^2
$$

选择 split 的步骤：

1. 计算 parent entropy $H(S)$。
2. 按 feature value 划分数据。
3. 计算 child entropy 的加权平均。
4. 用 parent entropy 减去 weighted child entropy。
5. 选择 information gain 最大的 feature。

防止过拟合：

- 限制 maximum depth。
- 限制 minimum samples per leaf。
- Pruning。
- Random forest / bagging。

---

## 11. Unsupervised Learning

### 11.1 K-means

目标函数：

$$
\min \sum_{i=1}^{m}\lVert x_i-\mu_{c_i}\rVert^2
$$

算法：

1. 选择 $k$ 个初始 centroids。
2. Assignment step：

$$
c_i=\arg\min_j \lVert x_i-\mu_j\rVert^2
$$

3. Update step：

$$
\mu_j=\frac{1}{|C_j|}\sum_{x_i\in C_j}x_i
$$

4. 重复直到 assignments 或 centroids 不再变化。

特点：

- 对初始化敏感。
- 对 outliers 敏感。
- 需要提前选择 $k$。
- K-means++ 可以改善初始化。

### 11.2 PCA

目标：找到最大方差方向，并把数据投影到低维空间。

步骤：

1. 中心化数据：

$$
X_{centered}=X-\bar X
$$

2. 计算 covariance：

$$
\Sigma=\frac{1}{m}X_{centered}^T X_{centered}
$$

3. 求 eigenvectors/eigenvalues。
4. 选择前 $k$ 个最大 eigenvalues 对应的 eigenvectors。
5. 投影：

$$
Z=X_{centered}W_k
$$

### 11.3 LLE

Locally Linear Embedding 保留局部邻域结构：

1. 为每个点找 nearest neighbors。
2. 用 neighbors 线性重构该点，求 reconstruction weights。
3. 在低维空间中保持这些 weights 不变。

---

## 12. Bayesian Networks 和 Uncertainty

Bayesian network 用 DAG 表示 joint distribution：

$$
P(X_1,\ldots,X_n)=\prod_i P(X_i\mid Parents(X_i))
$$

条件独立：

- 给定 parents 后，一个节点与它的 non-descendants 条件独立。
- D-separation 用来判断 evidence 是否阻断路径。

常见 inference：

- Exact inference：enumeration、variable elimination。
- Approximate inference：sampling、likelihood weighting、Gibbs sampling。

Rational decision 的核心：

$$
\text{Probability}+\text{Utility}\Rightarrow \text{Expected Utility}
$$

选择期望效用最大的 action：

$$
a^*=\arg\max_a EU(a)
$$

其中：

$$
EU(a)=\sum_s P(s\mid evidence,a)U(s)
$$

### 12.0 Exact Inference

如果 query 是 $X$，evidence 是 $e$，hidden variables 是 $Y$，则：

$$
P(X\mid e)=\alpha P(X,e)
$$

$$
P(X\mid e)=\alpha\sum_Y P(X,e,Y)
$$

其中 $\alpha$ 是 normalization constant，使概率和为 1。

Enumeration：

- 枚举所有 hidden variables 的取值。
- 简单但可能指数级昂贵。

Variable Elimination：

- 不是一次性展开完整 joint。
- 先把局部因子相乘，再逐个 sum out hidden variables。
- 比 naive enumeration 更高效。

Markov blanket：

节点 $X$ 的 Markov blanket 包括：

- Parents of $X$。
- Children of $X$。
- Other parents of $X$'s children。

给定 Markov blanket 后，$X$ 与网络中其他变量条件独立。

Gibbs sampling 每次重采样一个变量时，只需要看该变量的 Markov blanket。

### 12.1 画 Bayesian Network

如果一个变量直接影响另一个变量，就画一条有向边。

PDF 例题：性别影响头发长短。

- $G$：Gender，取值 Boy/Girl。
- $H$：Hair，取值 Long/Short。

结构：

$$
G \to H
$$

联合分布分解：

$$
P(G,H)=P(G)P(H\mid G)
$$

### 12.2 Bayes rule 例题：短发学生是女生的概率

题目给出：

$$
P(Boy)=0.7,\quad P(Girl)=0.3
$$

男生长短发比例 $1:9$：

$$
P(Short\mid Boy)=0.9
$$

女生长短发比例 $8:2$：

$$
P(Short\mid Girl)=0.2
$$

求：

$$
P(Girl\mid Short)
$$

用 Bayes rule：

$$
P(Girl\mid Short)
=
\frac{P(Short\mid Girl)P(Girl)}
{P(Short)}
$$

其中：

$$
P(Short)
=P(Short\mid Boy)P(Boy)+P(Short\mid Girl)P(Girl)
$$

代入：

$$
P(Girl\mid Short)
=
\frac{0.2\cdot 0.3}
{0.9\cdot 0.7+0.2\cdot 0.3}
=
\frac{0.06}{0.69}
=
\frac{2}{23}
\approx 0.087
$$

### 12.3 Prior Sampling

Prior sampling：按照 Bayesian network 的拓扑顺序，从先验开始逐个采样。

如果给了一组样本，估计 $P(C)$：

$$
\hat P(C)=\frac{\#\text{samples with }C=True}{\#\text{all samples}}
$$

PDF 例题给 8 个 samples，其中 $C=True$ 的有 5 个，所以：

$$
\hat P(C)=\frac{5}{8}
$$

### 12.4 Rejection Sampling

Rejection sampling 用来估计条件概率，比如：

$$
P(C\mid A,\neg D)
$$

做法：

1. 只保留符合 evidence 的样本，即 $A=True$ 且 $D=False$。
2. 丢掉不符合 evidence 的样本。
3. 在保留下来的样本中统计 $C=True$ 的比例。

PDF 例题中保留下来的样本是：

$$
(A,B,\neg C,\neg D),\quad
(A,\neg B,C,\neg D),\quad
(A,B,C,\neg D)
$$

其中 $C=True$ 的有 2 个，总共 3 个：

$$
\hat P(C\mid A,\neg D)=\frac{2}{3}
$$

### 12.5 Likelihood Weighting

Likelihood weighting 用来估计有 evidence 的条件概率。

规则：

- Evidence variables 固定，不随机采样。
- Non-evidence variables 正常采样。
- 每个样本有 weight：

$$
w=\prod_{E_i\in Evidence}P(E_i=e_i\mid Parents(E_i))
$$

PDF 中网络可看作链：

$$
A\to B\to C\to D
$$

表中概率：

$$
P(A)=\frac14
$$

$$
P(B\mid A)=\frac15,\quad P(B\mid \neg A)=\frac34
$$

$$
P(C\mid B)=\frac12,\quad P(C\mid \neg B)=\frac13
$$

$$
P(D\mid C)=\frac16,\quad P(D\mid \neg C)=\frac78
$$

估计：

$$
P(\neg A\mid B,\neg D)
$$

Evidence 是 $B=True,\ D=False$，所以每个样本权重：

$$
w=P(B\mid A)\cdot P(\neg D\mid C)
$$

注意第二项要看样本里的 $C$，第一项要看样本里的 $A$。

四个 weighted samples：

| Sample | Weight |
|---|---:|
| $\neg A,\ B,\ C,\ \neg D$ | $P(B\mid \neg A)P(\neg D\mid C)=\frac34\cdot\frac56=\frac58$ |
| $A,\ B,\ C,\ \neg D$ | $P(B\mid A)P(\neg D\mid C)=\frac15\cdot\frac56=\frac16$ |
| $A,\ B,\ \neg C,\ \neg D$ | $P(B\mid A)P(\neg D\mid \neg C)=\frac15\cdot\frac18=\frac{1}{40}$ |
| $\neg A,\ B,\ \neg C,\ \neg D$ | $P(B\mid \neg A)P(\neg D\mid \neg C)=\frac34\cdot\frac18=\frac{3}{32}$ |

估计 $P(\neg A\mid B,\neg D)$：

$$
\hat P(\neg A\mid B,\neg D)
=
\frac{\text{weights of samples with }\neg A}
{\text{all weights}}
$$

$$
=
\frac{\frac58+\frac{3}{32}}
{\frac58+\frac16+\frac{1}{40}+\frac{3}{32}}
=
\frac{345}{437}
\approx 0.789
$$

### 12.6 Gibbs Sampling / MCMC 判断题

Gibbs sampling 的核心规则：

1. Evidence variables 固定不变。
2. 每一步只重新采样一个 non-evidence variable。
3. 所以连续两个样本之间，最多只能有一个非 evidence 变量发生变化。

PDF 例题：evidence 是：

$$
A=True
$$

因此任何合法 sequence 中，$A$ 都必须一直是 True。

判断 sequence 是否可能由 Gibbs sampling 产生：

- 如果某一步 $A$ 变成 $\neg A$，一定不可能。
- 如果相邻两行同时改变两个或更多非 evidence variables，也不可能。
- 如果每次只改变 $B,C,D$ 中最多一个，且 $A$ 固定为 True，则可能。

PDF 中：

- Sequence 1：可能。
- Sequence 2：不可能，因为 evidence $A$ 被改变。
- Sequence 3：可能。
- Sequence 4：不可能，因为某一步同时改变了两个非 evidence variables。

---

## 13. Recommender Systems 和 Knowledge Graphs

### 13.1 Recommender systems

核心目标：学习一个打分函数：

$$
score(user,item)
$$

类型：

- Content-based：推荐与用户喜欢过的内容相似的 item。
- Collaborative filtering：利用 user-item interaction pattern。
- Hybrid：结合多种信号。

主要挑战：

- Cold start。
- Sparsity。
- Scalability。
- Accuracy 和 efficiency 的 trade-off。

### 13.2 Pearson correlation in collaborative filtering

对用户 $u$ 和 $v$：

$$
sim(u,v)=
\frac{
\sum_i (r_{u,i}-\bar r_u)(r_{v,i}-\bar r_v)
}{
\sqrt{\sum_i (r_{u,i}-\bar r_u)^2}
\sqrt{\sum_i (r_{v,i}-\bar r_v)^2}
}
$$

含义：

- 衡量两个用户或 item 的 normalized covariance。
- 用来寻找相似 users/items。
- 再根据邻居预测 missing ratings。

基于 user-user similarity 的预测例子：

$$
f(u,i)=\sum_{u'\in U}c_{u,u'}r_{u',i}
$$

基于 item-item similarity 的预测例子：

$$
f(u,i)=\sum_{i'\in M}c_{i,i'}r_{u,i'}
$$

### 13.3 Matrix Factorization

给定 user-item interaction matrix：

$$
R\in \mathbb{R}^{n\times m}
$$

矩阵分解假设：

$$
R\approx PQ^T
$$

其中：

$$
P\in \mathbb{R}^{n\times d},\quad Q\in \mathbb{R}^{m\times d}
$$

- $P_u$：user $u$ 的 embedding vector。
- $Q_i$：item $i$ 的 embedding vector。
- $d$：embedding dimension。

预测：

$$
\hat r_{u,i}=P_uQ_i^T
$$

训练目标通常只在已知 interaction $R'$ 上计算：

$$
\min_{P,Q}\sum_{(u,i)\in R'}
\left(r_{u,i}-P_uQ_i^T\right)^2
$$

优点：

- 比 correlation vector 更低维。
- Scalability 更好。

缺点：

- Interpretability 不如 Pearson/correlation-based 方法。

### 13.4 Knowledge graph

Knowledge graph 存储 triples：

$$
(head\ entity,\ relation,\ tail\ entity)
$$

RDF triple 写法：

$$
\langle head,\ relation,\ tail\rangle
$$

构建来源：

- Structured databases。
- 从文本中做 information extraction。
- 人工或领域知识。

Automatic entity recognition：

- 目标：从文本中识别 meaningful entities。
- 统计方法：TF-IDF、entropy。
- 机器学习方法：NER，sequence labeling。

TF-IDF 直觉：

- 一个词在当前文档频繁出现。
- 在其他文档中不常出现。
- 则更可能是重要实体/关键词。

Entropy 直觉：

$$
H(u)=-\sum_{x\in \mathcal{X}}p(x)\log p(x)
$$

如果词 $u$ 的左右邻居很多样，entropy 高，更可能是一个 meaningful entity。

NER 标签例子：

```text
B-People, I-People, O, B-Contest
```

Relation extraction：

- 输入：包含两个 entities 的句子。
- 输出：这两个 entities 的 relation category。
- 可以看成 text classification。

Completion 方法：

- Path-based methods。
- Embedding-based methods，例如 TransE-style entity/relation vectors。

KG-based recommender systems 利用实体关系提升 item representation 和 explainability。

---

## 14. Reinforcement Learning

核心元素：

- State $s$。
- Action $a$。
- Reward $r$。
- Policy $\pi(a\mid s)$。
- Discount factor $\gamma$。
- Value function $V(s)$。
- Action-value function $Q(s,a)$。

Return：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

Bellman optimality：

$$
V^*(s)=
\max_a\sum_{s'}P(s'\mid s,a)
\left[
R(s,a,s')+\gamma V^*(s')
\right]
$$

Q-learning update：

$$
Q(s,a)\leftarrow Q(s,a)+
\alpha\left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right]
$$

Exploration：

$$
\epsilon\text{-greedy}:
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## 15. 考前最后检查

进考场前，确保下面这些不用看笔记也会写：

1. 按 tie-breaking 画 BFS 和 A* 搜索树。
2. 计算 $f(n)=g(n)+h(n)$，并求 admissible 的 $c$ 范围。
3. 解释 local search、simulated annealing、tabu search 的区别。
4. 填 $\alpha/\beta$，标出 alpha-beta pruning 的剪枝分支。
5. 跑 AC-3，写清楚每一步 domain 怎么变。
6. 把 $p\to\neg(p\lor q)$ 转成 CNF：$\neg p$。
7. 把自然语言规则转成 propositional logic，再转 CNF。
8. 用 DPLL 做 unit propagation，推出变量真假。
9. 把一阶逻辑规则转 CNF，并用 resolution 推出 contradiction。
10. 写 perceptron 和 logistic regression 的更新公式。
11. 写 hard/soft-margin SVM objective、constraints 和 kernel trick。
12. 用 smoothing 做 Naive Bayes 预测。
13. 用 Bayes rule 算 $P(Girl\mid Short)$ 这类后验概率。
14. 会 exact inference、variable elimination、prior/rejection/likelihood weighting/Gibbs sampling。
15. 计算 accuracy、precision、recall、F1、TPR、FPR。
16. 说明 cross-validation、bootstrap、grid search 如何用于评估/调参。
17. 写 matrix factorization 的 $R\approx PQ^T$ 和预测公式。
18. 解释 KG triples、NER、relation extraction、KGC。
19. 解释如何公平比较 GA 和 backpropagation。
