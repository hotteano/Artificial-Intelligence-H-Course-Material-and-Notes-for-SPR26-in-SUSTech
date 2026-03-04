# IEMP Heuristic (`IEMP_heur.py`)

本文档描述 `IEMP_heur.py` 中启发式算法的核心思想、形式化启发式函数、求解流程和参数含义。

## 1. 问题目标

给定有向图 `G=(V,E)`、两组初始种子 `I1, I2` 和预算 `k`，需要寻找 `S1, S2`，满足：

- `|S1| + |S2| <= k`
- 最大化平衡曝光（等价于最小化对称差）：

\[
\max \mathbb{E}\left[\left|V - \left(r(S_1\cup I_1) \triangle r(S_2\cup I_2)\right)\right|\right]
\]

其中 `r(·)` 是“reached/exposed”节点集合（包含尝试激活失败但被尝试过的节点）。

---

## 2. 启发式函数（形式化）

算法先用 IMRank + LFA 分别得到两阵营的节点影响分数：`M1(v), M2(v)`。

归一化后：

\[
\hat M_1(v)=\frac{M_1(v)}{\max_u M_1(u)},\quad
\hat M_2(v)=\frac{M_2(v)}{\max_u M_2(u)}
\]

对候选节点 `v`，定义加入 `S1` 和 `S2` 的启发式分数：

\[
h_1(v\mid S_1,S_2)=\hat M_1(v)-\alpha\hat M_2(v)-\lambda\max(0,|S_1|-|S_2|)
\]

\[
h_2(v\mid S_1,S_2)=\hat M_2(v)-\alpha\hat M_1(v)-\lambda\max(0,|S_2|-|S_1|)
\]

含义：

- 第一项：鼓励节点对本阵营的传播能力；
- 第二项（`alpha`）：惩罚“对对方阵营也很有利”的节点；
- 第三项（`balance_lambda`）：惩罚 `S1/S2` 大小不平衡。

---

## 3. 如何寻找解

### Step A: 计算自洽 IMRank 分数

对每个 campaign `c∈{1,2}`：

1. 用加权出度初始化排名；
2. 迭代执行 `ranking -> LFA -> rerank`（最多 `max_iter` 次）；
3. 得到收敛后的分数 `Mc` 与排名 `rc`。

### Step B: 构造候选池

从 `r1` 和 `r2` 各取一部分 Top 节点，组成候选池（大小约 `candidate_size`），并与可选集合
`V \ (I1 ∪ I2)` 相交。

### Step C: 预算内贪心分配

重复 `min(k, |available|)` 轮：

- 对候选 `v` 同时计算 `h1(v)` 与 `h2(v)`；
- 选择分数最大的动作（把哪个 `v` 放入 `S1` 或 `S2`）；
- 更新集合并继续。

最终输出 `S1, S2` 到 `-b` 指定文件。

---

## 4. 关键参数

- `--max-iter`：IMRank 迭代次数上限；
- `--candidate-size`：每轮贪心评估的候选池规模；
- `--alpha`：跨阵营惩罚权重；
- `--balance-lambda`：`S1/S2` 规模平衡惩罚权重。

---

## 5. 复杂度（实现级）

记：

- `n=|V|`, `m=|E|`
- `T=max_iter`, `C=candidate_size`, `k=budget`

实现中的总体复杂度可写为：

- 时间复杂度（主导项）：

\[
O(T\,m\log n + kC)
\]

- 空间复杂度：

\[
O(n+m)
\]

实际中通常 `k << n` 且 `C << n`，主要耗时来自 IMRank-LFA 分数计算。

---

## 6. 运行示例

```powershell
Set-Location "c:\Users\edwar\Artificial-Intelligence-H-Course-Material-and-Notes-for-SPR26-in-SUSTech\Project1\Project1\Project1\Testcase\Heuristic"
C:/Users/edwar/AppData/Local/Programs/Python/Python312/python.exe .\IEMP_heur.py -n ".\map3\dataset2" -i ".\map3\seed2" -b ".\map3\seed_balanced_imrank_test" -k 15 --max-iter 8 --candidate-size 160 --alpha 0.5 --balance-lambda 0.12
```

如需打分：

```powershell
Set-Location "c:\Users\edwar\Artificial-Intelligence-H-Course-Material-and-Notes-for-SPR26-in-SUSTech\Project1\Project1\Project1\Testcase\Evaluator"
C:/Users/edwar/AppData/Local/Programs/Python/Python312/python.exe .\Evaluator.py -n "..\Heuristic\map3\dataset2" -i "..\Heuristic\map3\seed2" -b "..\Heuristic\map3\seed_balanced_imrank_test" -k 15 -o ".\tmp_eval.txt" --simulations 300
```