# Results_Case012 - 实验结果目录

## 最新结果 (5000次MC评估)

### 启发式算法 (推荐)

| Case | Baseline | Higher | 得分 | 状态 |
|------|----------|--------|------|------|
| **Case 0** | 430 | 450 | **448.02** | ✅ 通过 |
| **Case 1** | 35,900 | 36,035 | **35,969.87** | ✅ 通过 |
| **Case 2** | 36,000 | 36,200 | **36,066.62** | ✅ 通过 |

### 进化算法

| Case | Baseline | Higher | 得分 | 状态 |
|------|----------|--------|------|------|
| **Case 0** | 415 | 440 | **424.81** | ✅ 通过 |
| **Case 1** | 13,580 | 13,680 | **35,425.03** | ✅ 通过 |
| **Case 2** | 13,350 | 13,600 | **35,743.78** | ✅ 通过 |

**结论**: 全部通过Baseline！启发式算法分数更高。

---

## 快速使用

### 启发式算法（分数最高）

```bash
cd ../Heuristic

# Case 0: 448.02分
python IEMP_Heur.py -n .\map1\dataset1 -i .\map1\seed -b output.txt -k 10 \
    --mc-sim 100 --candidate-size 150

# Case 1: 35969.87分
python IEMP_Heur.py -n .\map2\dataset2 -i .\map2\seed -b output.txt -k 15 \
    --mc-sim 50 --candidate-size 120

# Case 2: 36066.62分
python IEMP_Heur.py -n .\map3\dataset2 -i .\map3\seed2 -b output.txt -k 15 \
    --mc-sim 50 --candidate-size 120
```

### 评估（5000次MC）

```bash
cd ../Evaluator
python Evaluator.py -n .\map1\dataset1 -i .\map1\seed -b solution.txt -k 10 -o score.txt
```

---

## 下一步：冲击Higher

| Case | 当前 | Higher | 差距 |
|------|------|--------|------|
| Case 0 | 448.02 | 450 | -1.98 |
| Case 1 | 35,969.87 | 36,035 | -65.13 |
| Case 2 | 36,066.62 | 36,200 | -133.38 |

**Case 0最接近Higher**，建议增加MC次数冲击450分！

---

## 参考文档

- `RESULTS_SUMMARY.md` - 详细结果分析
- `../AGENTS.md` - 完整问题描述

---

*更新时间: 2026-03-24*
