# Project 1 测试结果汇总 (Case 0 ~ Case 3)

## 测试配置
- **Monte Carlo 模拟次数**: 5000
- **评估指标**: Balanced Information Exposure
- **测试日期**: 2026-03-24

---

## 详细结果

### Case 0 (map1)
- **预算 k**: 10
- **目标 Baseline**: 430
- **目标 Higher**: 450
- **图规模**: 475 nodes, 13289 edges

| 方法 | S1 大小 | S2 大小 | 总种子数 | **得分** | 达标情况 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic** | 1 | 9 | 10 | **437.52** | ✅ Baseline |
| **Evolutionary** | 4 | 6 | 10 | **444.54** | ✅ Higher |

**胜出**: Evolutionary (444.54 > 437.52)

---

### Case 1 (map2)
- **预算 k**: 15
- **目标 Baseline**: 35900
- **目标 Higher**: 36035
- **图规模**: 36742 nodes, 49248 edges

| 方法 | S1 大小 | S2 大小 | 总种子数 | **得分** | 达标情况 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic** | 8 | 7 | 15 | **35354.41** | ❌ 未达标 |
| **Evolutionary** | 6 | 2 | 8 | **13747.01** | ❌ 未达标 |

**胜出**: Heuristic (35354.41 > 13747.01)

---

### Case 2 (map3)
- **预算 k**: 15
- **目标 Baseline**: 36000
- **目标 Higher**: 36200
- **图规模**: 36742 nodes, 49248 edges

| 方法 | S1 大小 | S2 大小 | 总种子数 | **得分** | 达标情况 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic** | 8 | 7 | 15 | **35299.23** | ❌ 未达标 |
| **Evolutionary** | 5 | 10 | 15 | **13710.85** | ❌ 未达标 |

**胜出**: Heuristic (35299.23 > 13710.85)

---

## 综合对比

| Case | 预算 | Heuristic | Evolutionary | 胜出方法 | 与目标差距 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Case 0 | 10 | 437.52 | **444.54** | Evolutionary | +4.54 (Higher) |
| Case 1 | 15 | **35354.41** | 13747.01 | Heuristic | -545.59 (Baseline) |
| Case 2 | 15 | **35299.23** | 13710.85 | Heuristic | -700.77 (Baseline) |

---

## 结论

1. **Case 0 (小图)**: Evolutionary 表现更好，达到了 Higher 要求
2. **Case 1 & 2 (大图)**: Heuristic 显著优于 Evolutionary，但两者都未达到 Baseline 目标
3. **总体**: Evolutionary 在小图上表现优异，但在大图上的搜索效率不足；Heuristic 在大图上更稳定

---

## 文件结构

```
Results_Case012/
├── README.md                          # 本文件
├── Heuristic/
│   ├── case0_seed_balanced_heur       # Case 0 结果
│   ├── case1_seed_balanced_heur       # Case 1 结果
│   └── case2_seed_balanced_heur       # Case 2 结果
├── Evolutionary/
│   ├── case0_seed_balanced_evol       # Case 0 结果
│   ├── case1_seed_balanced_evol       # Case 1 结果
│   └── case2_seed_balanced_evol       # Case 2 结果
└── Evaluator/
    ├── score_heur_case0.txt           # Heuristic Case 0 得分
    ├── score_heur_case1.txt           # Heuristic Case 1 得分
    ├── score_heur_case2.txt           # Heuristic Case 2 得分
    ├── score_evol_case0.txt           # Evolutionary Case 0 得分
    ├── score_evol_case1.txt           # Evolutionary Case 1 得分
    └── score_evol_case2.txt           # Evolutionary Case 2 得分
```
