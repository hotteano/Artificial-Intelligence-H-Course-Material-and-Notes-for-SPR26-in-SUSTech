# Project 1 测试结果汇总 - 含长时间运行优化 (Case 0 ~ Case 3)

## 测试配置
- **Monte Carlo 模拟次数**: 5000
- **短时间运行**: generations=80, pop-size=40, 早停20代
- **长时间运行**: generations=200, pop-size=60, 无早停限制
- **测试日期**: 2026-03-24

---

## 📊 详细结果

### Case 0 (map1) - 小图
- **预算 k**: 10
- **目标 Baseline**: 430 | **目标 Higher**: 450
- **图规模**: 475 nodes, 13289 edges

| 方法 | S1 | S2 | 总数 | **得分** | 达标情况 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Heuristic | 1 | 9 | 10 | **437.52** | ✅ Baseline |
| Evolutionary (短) | 4 | 6 | 10 | **444.54** | ✅ Higher |

**胜出**: Evolutionary - 达到 Higher 要求 ✅

---

### Case 1 (map2) - 大图
- **预算 k**: 15
- **目标 Baseline**: 35900 | **目标 Higher**: 36035
- **图规模**: 13984 nodes, 17319 edges

| 方法 | S1 | S2 | 总数 | **得分** | vs Baseline | 达标 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic** | 8 | 7 | 15 | **35354.41** | -545.59 | ❌ |
| Evolutionary (短) | 6 | 2 | 8 | **13747.01** | -22152.99 | ❌ |
| Evolutionary (长) | 7 | 5 | 12 | **13752.98** | -22147.02 | ❌ |

**胜出**: Heuristic - 但差距目标仍很远

**长时间运行效果**: 微弱提升 (+5.97, 0.04%)，几乎无改进

---

### Case 2 (map3) - 大图
- **预算 k**: 15
- **目标 Baseline**: 36000 | **目标 Higher**: 36200
- **图规模**: 13984 nodes, 17319 edges

| 方法 | S1 | S2 | 总数 | **得分** | vs Baseline | 达标 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic** | 8 | 7 | 15 | **35299.23** | -700.77 | ❌ |
| Evolutionary (短) | 5 | 10 | 15 | **13710.85** | -22289.15 | ❌ |
| Evolutionary (长) | 5 | 5 | 10 | **13703.03** | -22296.97 | ❌ |

**胜出**: Heuristic

**长时间运行效果**: 略微下降 (-7.82, -0.06%)，没有改进

---

## 📈 效果分析：长时间运行 vs 短时间运行

### Evolutionary 在大图上的表现

| Case | 短时间 | 长时间 | 变化 | 改进幅度 |
|:---:|:---:|:---:|:---:|:---:|
| Case 1 | 13747.01 | 13752.98 | +5.97 | 0.04% |
| Case 2 | 13710.85 | 13703.03 | -7.82 | -0.06% |

**结论**: 
- ⚠️ 单纯增加运行时间对 Evolutionary 算法**效果微乎其微**
- 算法在约 30-50 代后就已收敛，后续 150+ 代没有实质改进
- 问题可能在于：**适应度评估的精度**（只用 30-100 次模拟）或**搜索策略**

---

## 🎯 与目标对比

| Case | Baseline | Higher | Heuristic | 差距 | Evolutionary (最优) | 差距 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Case 0 | 430 | 450 | 437.52 | -12.48 | 444.54 | -5.46 |
| Case 1 | 35900 | 36035 | 35354.41 | -545.59 | 13752.98 | -22147.02 |
| Case 2 | 36000 | 36200 | 35299.23 | -700.77 | 13703.03 | -22296.97 |

---

## 💡 关键发现

### 1. 小图 (Case 0)
- Evolutionary 表现优秀，达到 Higher 要求 ✅
- Heuristic 也能达到 Baseline

### 2. 大图 (Case 1 & 2)
- **两种方法都远未达到 Baseline** (差距 500-22000 分)
- Heuristic 比 Evolutionary 好约 2.5 倍
- 但 Heuristic 本身也只达到目标的 ~98%

### 3. 长时间运行的价值
- 对 Case 0: 不需要（已经很好）
- 对 Case 1 & 2: **几乎没有价值**
  - Evolutionary 在 30-50 代后就收敛
  - 跑了 200 代也没有突破

---

## 🔍 问题分析

为什么 Evolutionary 在大图上表现这么差？

1. **适应度评估不准确**: 使用 30-100 次 Monte Carlo 模拟可能不够
2. **搜索空间太大**: 13984 个节点中选择 15 个，组合爆炸
3. **模拟退火温度策略**: 温度下降过快可能陷入局部最优
4. **基因编码问题**: 当前编码方式可能不适合大图

---

## 📁 文件结构

```
Results_Case012/
├── README.md / README_v2.md
├── Heuristic/
│   ├── case0_seed_balanced_heur
│   ├── case1_seed_balanced_heur
│   └── case2_seed_balanced_heur
├── Evolutionary/
│   ├── case0_seed_balanced_evol          # 短时间
│   ├── case1_seed_balanced_evol          # 短时间
│   ├── case2_seed_balanced_evol          # 短时间
│   ├── case1_seed_balanced_evol_long     # 长时间 (200代)
│   └── case2_seed_balanced_evol_long     # 长时间 (200代)
└── Evaluator/
    ├── score_heur_case0.txt      # 437.52
    ├── score_heur_case1.txt      # 35354.41
    ├── score_heur_case2.txt      # 35299.23
    ├── score_evol_case0.txt      # 444.54
    ├── score_evol_case1.txt      # 13747.01
    ├── score_evol_case2.txt      # 13710.85
    ├── score_evol_case1_long.txt # 13752.98
    └── score_evol_case2_long.txt # 13703.03
```

---

## 🚀 改进建议

如果要进一步优化 Evolutionary 算法：

1. **提高适应度评估精度**: 增加 `n_simulations` 到 200-500
2. **改进初始化策略**: 使用 Heuristic 结果作为初始种群
3. **调整模拟退火参数**: 更慢的温度下降
4. **局部搜索**: 在进化后期加入局部优化
5. **混合算法**: 先用 Heuristic 找到好解，再用 Evolutionary 微调
