# IEMP_Evol.py - 遗传算法 (Genetic Algorithm + Simulated Annealing)

本文档描述 `IEMP_ga.py` 中遗传算法的核心思想、编码方案、遗传算子和参数设置。

---

## 1. 问题建模与编码方案

### 1.1 问题目标

给定有向图 $G=(V,E)$、两组初始种子 $I_1, I_2$ 和预算 $k$，需要寻找 $S_1, S_2$，满足：

- $|S_1| + |S_2| \leq k$
- 最大化平衡曝光（最小化对称差）：

$$
\max \mathbb{E}\left[\left|V - \left(r(I_1\cup S_1) \triangle r(I_2\cup S_2)\right)\right|\right]
$$

### 1.2 二维向量编码

个体 (Individual) 被编码为一个二元组：

```python
Individual(S1: Set[int], S2: Set[int])
```

- **S1**: Campaign 1 的平衡种子集合
- **S2**: Campaign 2 的平衡种子集合
- **约束**: $S_1 \cap S_2 = \emptyset$ 且 $S_1, S_2 \subseteq V \setminus (I_1 \cup I_2)$

这种编码方式直接对应问题的解空间，便于遗传操作。

---

## 2. 适应度评估

### 2.1 Monte Carlo 模拟

适应度函数通过 IC (Independent Cascade) 模型的 Monte Carlo 模拟计算：

```python
fitness = E[|V - (r(I1∪S1) △ r(I2∪S2))|]
        = 平均被两个阵营同时曝光或同时未曝光的节点数
```

### 2.2 评估策略

- **进化过程中**: 使用 30 次模拟进行快速评估
- **最终评估**: 使用 100-200 次模拟获得精确结果

---

## 3. 遗传算子

### 3.1 选择算子：锦标赛选择 (Tournament Selection)

从种群中随机选择 `tournament_size` 个个体，返回适应度最高者。

**优点**: 平衡选择压力，避免早熟收敛。

### 3.2 交叉算子：均匀交叉 (Uniform Crossover)

对 S1 和 S2 分别进行均匀交叉：

```
Parent 1: S1={a,b,c}, S2={x,y}
Parent 2: S1={b,d},   S2={y,z}

交叉后:
Child 1: S1={a,b,d}, S2={x,y}    (从父代1继承a,b，从父代2继承d)
Child 2: S1={b,c},   S2={y,z}    (从父代1继承b,c，从父代2继承z)
```

### 3.3 变异算子

包含五种变异操作：

| 变异类型 | 操作描述 | 作用 |
|---------|---------|------|
| **Add** | 向 S1 或 S2 添加一个新节点 | 探索新的节点组合 |
| **Remove** | 从 S1 或 S2 删除一个节点 | 释放预算空间 |
| **Swap** | 交换 S1 和 S2 中的各一个节点 | 调整阵营分配 |
| **Transfer** | 将节点从 S1 移到 S2 或反之 | 重新平衡阵营 |
| **Replace** | 用一个新节点替换现有节点 | 局部搜索改进 |

---

## 4. 模拟退火策略 (Simulated Annealing)

### 4.1 接受准则

引入模拟退火机制以跳出局部最优：

- 如果新解更好 ($\Delta \geq 0$): 总是接受
- 如果新解更差 ($\Delta < 0$): 以概率 $\exp(\Delta / T)$ 接受

其中 $\Delta = fitness_{new} - fitness_{current}$，$T$ 是当前温度。

### 4.2 降温 schedule

$$T_{new} = \max(T_{min}, T_{current} \times \alpha)$$

- 初始温度: 50.0
- 降温系数: 0.95
- 最低温度: 1.0

---

## 5. 算法流程

```
1. 初始化种群 (混合随机和贪心个体)
2. 评估初始种群
3. For generation = 1 to max_generations:
   a. 保留精英个体
   b. While 新种群未满:
      i.   锦标赛选择父代
      ii.  交叉产生子代
      iii. 变异子代
      iv.  评估子代
      v.   模拟退火决定是否接受
   c. 更新最优解
   d. 降温
   e. 检查早停条件
4. 返回最优解
```

---

## 6. 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--pop-size` | 40 | 种群大小 |
| `--generations` | 80 | 最大进化代数 |
| `--crossover-rate` | 0.8 | 交叉概率 |
| `--mutation-rate` | 0.3 | 变异概率 |
| `--elite-size` | 4 | 精英保留数量 |
| `--tournament-size` | 3 | 锦标赛规模 |
| `--initial-temp` | 50.0 | 模拟退火初始温度 |
| `--cooling-rate` | 0.95 | 降温系数 |
| `--eval-simulations` | 200 | 最终评估模拟次数 |

---

## 7. 运行示例 (符合 Project 要求的命令格式)

### 基本运行

```powershell
Set-Location "Project1\Project1\Project1\Testcase\Evolutionary"
python IEMP_Evol.py -n ".\map1\dataset1" -i ".\map1\seed" -b ".\map1\seed_balanced_evol" -k 10
```

### 完整参数设置

```powershell
python IEMP_Evol.py `
    -n ".\map1\dataset1" `
    -i ".\map1\seed" `
    -b ".\map1\seed_balanced_evol" `
    -k 10 `
    --pop-size 50 `
    --generations 100 `
    --seed 42
```

---

## 8. 结果验证

使用 Evaluator 验证进化算法的结果：

```powershell
Set-Location "Project1\Project1\Project1\Testcase\Evaluator"
python Evaluator.py `
    -n "..\Evolutionary\map1\dataset1" `
    -i "..\Evolutionary\map1\seed" `
    -b "..\Evolutionary\map1\seed_balanced_evol" `
    -k 10 `
    -o ".\eval_evol_result.txt" `
    --simulations 1000
```

---

## 9. 算法复杂度

- **时间复杂度**: $O(G \times P \times S \times (|V| + |E|))$
  - $G$: 进化代数
  - $P$: 种群大小
  - $S$: 每次评估的 Monte Carlo 模拟次数
  
- **空间复杂度**: $O(P \times k + |V| + |E|)$
  - 存储种群和图结构

---

## 10. 与启发式算法的比较

| 特性 | 遗传算法 (GA) | 启发式算法 (IMRank) |
|-----|--------------|-------------------|
| 搜索策略 | 全局搜索 | 贪心局部搜索 |
| 计算开销 | 较高 (需多次模拟) | 较低 (LFA估计) |
| 解的质量 | 通常更好 | 快速但可能局部最优 |
| 参数调节 | 较多参数 | 较少参数 |
| 适用场景 | 预算充足、追求高质量解 | 快速求解、大规模图 |

---

## 11. 注意事项

1. **随机种子**: 使用 `--seed` 参数可复现结果
2. **早停机制**: 连续 20 代无改进时自动停止
3. **预算约束**: 算法确保 $|S_1| + |S_2| \leq k$
4. **模拟次数**: 增加 `--eval-simulations` 可提高结果稳定性
