# Task 1 — Image Classification 实验记录

## 1. 实验目的

在 Project 2 Task 1 的图像分类任务中，使用 `scikit-learn` 中的多种模型进行训练与对比，选择验证准确率最高的模型作为最终提交方案。

## 2. 数据集与预处理

- **训练数据**：`classification_train_data.pkl`，共 49,976 条样本
- **特征维度**：256 维（原始数据第 0 列为索引，已去除）
- **类别数**：10 类（标签 0~9）
- **划分方式**：训练集 80% / 验证集 20%，随机种子 `seed=123`，与 baseline 完全一致
- **预处理**：Z-score 标准化（基于训练集计算 mean 和 std）

## 3. 模型对比结果

| 模型 | 关键配置 | 验证准确率 |
|------|---------|-----------|
| **Baseline (Softmax Regression)** | 手写实现，lr=0.1，10,000 iters | **51.15%** |
| LogisticRegression (C=0.5) | `lbfgs` 求解器，L2 正则化 | 51.32% |
| LogisticRegression (C=1.0) | `lbfgs` 求解器，L2 正则化 | 51.31% |
| LogisticRegression (C=2.0) | `lbfgs` 求解器，L2 正则化 | 51.31% |
| LogisticRegression (saga, C=1.0) | `saga` 求解器 | 51.29% |
| LinearSVC (C=1.0) | 线性核 SVM | 50.66% |
| RandomForest (n=200, max_depth=20) | 限制深度防止过拟合 | 48.00% |
| RandomForest (n=500, max_depth=30) | 进一步增加树数量和深度 | 49.97% |
| RandomForest (无限制深度) | `n_estimators=200` | 46.62%（训练集 100%，严重过拟合） |
| **MLP (128)** | 1 层隐藏层（128 神经元），`early_stopping=True` | **53.60%** |
| MLP (256, 128) | 2 层隐藏层，`early_stopping=True` | **53.82%** |

## 4. 结论与最终方案

- **树模型不适合该任务**：Random Forest 在 256 维稠密特征上严重过拟合，即使限制深度后验证准确率也低于 baseline。
- **线性模型提升有限**：`LogisticRegression` 虽略优于 baseline（约 +0.15%），但提升空间不大。
- **MLP 效果最佳**：引入非线性隐藏层后，验证准确率显著提升至 **53.60%**（128 层）和 **53.82%**（256+128 层），均明显超过 baseline。
- **最终选择**：采用 **MLP (128)** 作为最终提交模型。原因是在准确率差距极小（53.60% vs 53.82%）的情况下，128 层的网络结构更简单、训练更快、模型文件更小，泛化风险更低。

## 5. 最终提交文件

- `classifier.py`：包含 `Classifier` 推理类及 `main()` 训练入口
- `classification_model.pkl`：训练好的 MLP (128) 模型
- `classification_mean.pkl`：训练集均值（标准化用）
- `classification_std.pkl`：训练集标准差（标准化用）
