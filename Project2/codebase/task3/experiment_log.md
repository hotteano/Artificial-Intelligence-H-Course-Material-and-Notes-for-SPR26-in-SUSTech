# Task 3 特征选择实验记录

## 实验目标

从 256 维图像特征中选取 **不超过 30 个特征**，使得固定预训练 Softmax 分类器在验证集上的准确率尽可能高。

- **Baseline（随机选 30 维）**: ~16.49%
- **约束条件**: mask 中 1 的个数 ≤ 30
- **数据集**: `classification_validation_data.pkl` (9,996 样本, 257 列含索引)
- **固定模型权重**: `image_recognition_model_weights.pkl` (257×10, 含偏置)

---

## 实验方法

### 1. LOO Backward Elimination（向后淘汰）
- **思路**: 从全部 256 个特征开始，每轮尝试去掉一个特征，选择对验证准确率伤害最小的特征永久剔除，直至只剩 30 个。
- **加速技巧**: 利用 Softmax Regression 的线性结构，预计算完整 logits，去特征时只需减去该特征对应的线性贡献，避免重复前向传播。

### 2. Forward Selection（前向选择）
- **思路**: 从空集合开始，每轮从剩余特征中挑选加入后验证准确率提升最大的特征，直到选满 30 个。
- **特点**: 贪心策略，特征一旦被选中不会剔除，速度比 Backward 快（只需 30 轮）。

### 3. Approximate Shapley Value（蒙特卡洛近似 Shapley）
- **思路**: 对特征进行随机排列采样（500 次），逐步加入特征并记录边际准确率增益，平均后得到每个特征的 Shapley Value，取 Top 30。
- **理论基础**: Shapley Value 能公平衡量每个特征在所有可能组合中的平均贡献。

### 4. Mutual Information（互信息）
- **思路**: 计算每个特征与标签的互信息得分，直接选 Top 30。
- **实现**: `sklearn.feature_selection.mutual_info_classif`

### 5. Correlation（皮尔逊相关）
- **思路**: 计算每个特征与整数标签的绝对皮尔逊相关系数，选 Top 30。

### 6. Weight Magnitude（权重绝对值和）
- **思路**: 对固定模型权重（去掉偏置后 256×10），计算每个特征维度对 10 个类别权重的绝对值之和，选 Top 30。

### 7. Weight L2 Norm（权重 L2 范数）
- **思路**: 计算每个特征维度权重向量的 L2 范数作为重要性，选 Top 30。

---

## 实验结果

| 排名 | 方法 | 验证准确率 | 相对 Baseline 提升 | 耗时 |
|:---:|:---|:---:|:---:|:---:|
| **#1** | **LOO Backward Elimination** | **45.94%** | **+29.45%** | 14.49s |
| #2 | Approximate Shapley Value (500 perm) | 45.77% | +29.28% | 89.53s |
| #3 | Forward Selection | 45.56% | +29.07% | 9.38s |
| #4 | Mutual Information | 39.98% | +23.49% | 5.73s |
| #5 | Correlation | 31.07% | +14.58% | 0.02s |
| #6 | Weight L2 Norm | 30.47% | +13.98% | 0.00s |
| #7 | Weight Magnitude | 28.27% | +11.78% | 0.00s |
| — | **Baseline (Random)** | **16.49%** | — | — |

---

## 结果分析

### 第一梯队（> 45%）
- **LOO Backward、Forward Selection、Shapley Value** 三者表现非常接近，均达到 45.5% 以上。
- 这说明：
  1. 基于**模型预测准确率直接反馈**的 wrapper 方法（LOO / Forward）远优于基于统计指标或模型权重的 filter 方法。
  2. LOO 和 Forward 的速度（9~15 秒）远快于 Shapley（~90 秒），但效果几乎持平，性价比最高。

### 第二梯队（~40%）
- **Mutual Information** 作为 filter 方法表现尚可（39.98%），说明特征与标签的非线性相关性有一定指示作用，但缺乏模型反馈使其无法进一步逼近最优子集。

### 第三梯队（~28-31%）
- **Correlation、Weight L2、Weight Magnitude** 效果较差。原因可能包括：
  - 固定模型权重虽然来自预训练，但该权重是针对**全部 256 维**优化的，直接取权重大的维度并不等于"只用这 30 维时效果最好"。
  - 皮尔逊相关只能捕捉线性关系，对多分类问题信息有限。

---

## 结论与建议

1. **最终提交方案**: 采用 **LOO Backward Elimination**（准确率 45.94%，排名第一）。
2. **备选方案**: 若时间受限，**Forward Selection**（45.56%，9.38 秒）是性价比最高的替代。
3. **Shapley Value** 虽然理论优雅，但 500 次排列采样耗时近 90 秒，且准确率略低于 LOO，不太适合本次限时场景。
4. 单纯的 filter 方法（权重、相关、互信息）在这个任务上表现明显不如 wrapper 方法，**建议优先使用基于准确率的贪心搜索**。

---

## 生成的文件

- `mask_code.pkl`: 由 LOO Backward 生成的最优 mask，形状 `(1, 256)`，共 30 个 1。
- `selector.py`: 提交文件，自动加载 `mask_code.pkl`。
- `compare_methods.py`: 多方法对比脚本（包含全部 7 种方法）。
- `experiment_log.md`: 本实验记录文件。
