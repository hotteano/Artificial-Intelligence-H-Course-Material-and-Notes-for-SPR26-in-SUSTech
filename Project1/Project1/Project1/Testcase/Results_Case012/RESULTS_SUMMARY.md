# IEMP 实验结果总结

## 实验日期
2026-03-24 (最新更新: 包含完整解)

---

## 评估配置

- **MC采样次数**: 5000次
- **目标**: 最大化 Balanced Information Exposure
- **约束**: 不动dataset和seed文件

---

## 两套Baseline标准

### 启发式算法 (Heuristic) Baseline
| Case | Baseline | Higher | k |
|------|----------|--------|---|
| Case 0 | 430 | 450 | 10 |
| Case 1 | 35,900 | 36,035 | 15 |
| Case 2 | 36,000 | 36,200 | 15 |

### 进化算法 (Evolutionary) Baseline
| Case | Baseline | Higher | k |
|------|----------|--------|---|
| Case 0 | 415 | 440 | 10 |
| Case 1 | 13,580 | 13,680 | 14 |
| Case 2 | 13,350 | 13,600 | 14 |

---

## 最新结果 (5000次MC评估)

### 启发式算法 (MC-Guided Heuristic) - 完整解

#### Case 0 - 448.02分 ✅
```
S1 (3节点): 169, 322, 367
S2 (7节点): 27, 66, 105, 197, 280, 316, 404
总预算: 10/10 (100%)
评估得分: 448.0228
```
**参数**: `--mc-sim 100 --candidate-size 150`

#### Case 1 - 35,969.87分 ✅
```
S1 (15节点): 124, 2529, 3187, 5137, 6633, 12802, 14290, 18986, 20098, 21545, 22206, 27138, 28420, 28966, 36528
S2 (0节点): (无)
总预算: 15/15 (100%)
评估得分: 35969.8652
```
**参数**: `--mc-sim 50 --candidate-size 120`

#### Case 2 - 36,066.62分 ✅
```
S1 (12节点): 877, 2123, 2529, 4155, 5824, 15417, 18748, 19286, 24207, 28420, 28966, 30316
S2 (3节点): 13787, 14039, 36345
总预算: 15/15 (100%)
评估得分: 36066.6194
```
**参数**: `--mc-sim 50 --candidate-size 120`

---

### 进化算法 (Binary GA) - 完整解

#### Case 0 - 424.81分 ✅
```
S1 (5节点): 185, 236, 239, 241, 436
S2 (4节点): 57, 200, 280, 362
总预算: 9/10 (90%)
评估得分: 424.8130
```
**参数**: `--pop-size 80 --generations 150 --mc-fine 500`

#### Case 1 - 35,425.03分 ✅
```
S1 (0节点): (无)
S2 (3节点): 6745, 8017, 10884
总预算: 3/15 (20%)
评估得分: 35425.0318
```
**参数**: `--pop-size 50 --generations 80 --mc-fine 200`

#### Case 2 - 36,708.00分 ✅
```
S1 (4节点): 2805, 4592, 5492, 13939
S2 (3节点): 1332, 5439, 13421
总预算: 7/15 (46.7%)
评估得分: 36708.0000
```
**参数**: `--pop-size 50 --generations 80 --mc-fine 200`
**注意**: 使用 `Evolutionary/map3/seed` (5+5=10节点)

---

## 结果分析

### 🎉 全部通过Baseline

| Case | 启发式 | 进化 | 胜者 |
|------|--------|------|-----|
| 0 | **448.02** ✅ | 424.81 ✅ | 启发式 |
| 1 | **35,969.87** ✅ | 35,425.03 ✅ | 启发式 |
| 2 | 36,066.62 ✅ | **36,708.00** ✅ | 进化 |

**Case 2特例**: 进化算法得分略高于启发式，但仍未充分利用预算(7/15)

---

## 解的特征对比

### 预算使用率

| 算法 | Case 0 | Case 1 | Case 2 |
|-----|--------|--------|--------|
| 启发式 | 100% (10/10) | 100% (15/15) | 100% (15/15) |
| 进化 | 90% (9/10) | 20% (3/15) | 46.7% (7/15) |

### S1/S2分配策略

**启发式**:
- Case 0: 3:7 (分散)
- Case 1: 15:0 (全部给S1)
- Case 2: 12:3 (偏向S1)

**进化算法**:
- Case 0: 5:4 (相对平衡)
- Case 1: 0:3 (全部给S2)
- Case 2: 4:3 (相对平衡)

---

## 下一步：冲击Higher

### 距离Higher

| Case | 当前最佳 | Higher | 差距 | 推荐算法 |
|------|---------|--------|------|---------|
| **Case 0** | **448.02** | 450 | -1.98 | 启发式 |
| **Case 1** | **35,969.87** | 36,035 | -65.13 | 启发式 |
| **Case 2** | **36,708.00** | - | - | 已达较好水平 |

### 优化建议

**Case 0 冲击Higher**:
```bash
python IEMP_Heur.py -n .\map1\dataset1 -i .\map1\seed -b output.txt -k 10 \
    --mc-sim 150 --candidate-size 200
```

**Case 1 冲击Higher**:
```bash
python IEMP_Heur.py -n .\map2\dataset2 -i .\map2\seed -b output.txt -k 15 \
    --mc-sim 100 --candidate-size 150
```

---

## 运行命令

### 启发式算法（推荐）
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

### 进化算法
```bash
cd ../Evolutionary

# Case 0: 424.81分
python IEMP_Evol.py -n .\map1\dataset1 -i .\map1\seed -b output.txt -k 10 \
    --pop-size 80 --generations 150

# Case 1: 35425.03分
python IEMP_Evol.py -n .\map2\dataset2 -i .\map2\seed -b output.txt -k 15 \
    --pop-size 50 --generations 80

# Case 2: 36708.00分 (使用map3/seed)
python IEMP_Evol.py -n .\map3\dataset2 -i .\map3\seed -b output.txt -k 15 \
    --pop-size 50 --generations 80
```

### 评估（5000次MC）
```bash
cd ../Evaluator
python Evaluator.py -n .\map1\dataset1 -i .\map1\seed -b solution.txt -k 10 -o score.txt
```

---

## 文件位置

```
Results_Case012/
├── Heuristic/
│   ├── case0.txt      # S1=[169,322,367], S2=[27,66,105,197,280,316,404]
│   ├── case1.txt      # S1=[124,2529,...], S2=[]
│   └── case2.txt      # S1=[877,2123,...], S2=[13787,14039,36345]
├── Evolutionary/
│   ├── case0.txt      # S1=[185,236,239,241,436], S2=[57,200,280,362]
│   ├── case1.txt      # S1=[], S2=[6745,8017,10884]
│   └── case2.txt      # S1=[2805,4592,5492,13939], S2=[1332,5439,13421]
├── Evaluator/
│   ├── heur_case0.txt # 448.0228
│   ├── heur_case1.txt # 35969.8652
│   ├── heur_case2.txt # 36066.6194
│   ├── evol_case0.txt # 424.8130
│   ├── evol_case1.txt # 35425.0318
│   └── evol_case2.txt # 36708.0000
└── RESULTS_SUMMARY.md # 本文档
```

---

*更新时间: 2026-03-24*
*状态: 全部通过Baseline，文档包含完整解*
