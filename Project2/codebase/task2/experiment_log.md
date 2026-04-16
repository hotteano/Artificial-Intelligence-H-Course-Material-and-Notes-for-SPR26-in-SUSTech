# Task 2 — Image Retrieval 实验记录

## 1. 实验目的

在 Project 2 Task 2 的图像检索任务中，为每张测试图片从 repository 中检索 5 张最相似的图片。测试准确率定义为每张查询图的命中数 `m/5`，再对整个测试集取平均。评分规则为：超过 baseline 则得满分，否则 0 分。

本实验旨在对比不同距离度量（metric）和预处理策略对检索结果的影响，选择最稳健的方案作为最终提交。

## 2. 数据集与预处理

- **Repository**：`image_retrieval_repository_data.pkl`，共 5,000 条样本，257 列（第 0 列为图片 ID，后 256 列为特征）
- **查询集**：由于 Task 2 未提供带标签的验证集，本实验以 repository 自身的前 1,000 条样本作为自检索（self-retrieval）查询，观察不同方法的 top-5 一致性与速度。
- **预处理对比**：
  - **Raw**：直接使用原始特征
  - **Z-score Normalized**：基于 repository 计算 mean 和 std，再做标准化

## 3. 实验方法

使用 `sklearn.neighbors.NearestNeighbors`（brute-force）替代 baseline 的手写 L2 暴力循环，测试以下 4 种距离度量：

1. **L2 (euclidean)**：baseline 使用的距离
2. **Cosine**：向量夹角距离
3. **Manhattan (L1)**：L1 范数距离
4. **Correlation**：Pearson 相关系数距离

分别在 **Raw** 和 **Normalized** 特征上运行，记录推理时间、Top-1/Top-5 self-hit、MRR（Mean Reciprocal Rank），以及与 baseline L2 的 top-5 重叠率。

## 4. 实验结果

### 4.1 详细指标

| 方法 | Top-1 self-hit | Top-5 self-hit | MRR | 推理时间 (1000 queries) | 与 L2 (raw) 重叠率 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **L2 (Baseline-like) \| raw** | 100.00% | 100.00% | 1.0000 | 1.6711 s | — |
| **Cosine \| raw** | 100.00% | 100.00% | 1.0000 | 0.0815 s | **99.68%** |
| Manhattan (L1) \| raw | 100.00% | 100.00% | 1.0000 | 0.1057 s | 72.78% |
| Correlation \| raw | 100.00% | 100.00% | 1.0000 | 0.6376 s | 93.84% |
| L2 (Baseline-like) \| norm | 100.00% | 100.00% | 1.0000 | 0.0155 s | 38.96% |
| Cosine \| norm | 100.00% | 100.00% | 1.0000 | 0.0802 s | 43.40% |
| Manhattan (L1) \| norm | 100.00% | 100.00% | 1.0000 | 0.1000 s | 39.96% |
| Correlation \| norm | 100.00% | 100.00% | 1.0000 | 0.6231 s | 43.04% |

> 注：self-hit 均为 100% 是因为查询样本本身就在 repository 中，最近邻必然包含自身。更有参考价值的是**与 baseline L2 的重叠率**和**推理速度**。

### 4.2 关键发现

- **Cosine (raw) 与 baseline 最接近**：top-5 重叠率高达 **99.68%**，说明检索结果与 baseline 几乎完全一致，准确率有保障。
- **速度优势显著**：`sklearn` 的 brute-force 实现比 baseline 手写循环快约 **20 倍**（0.08 s vs 1.67 s），在 OJ 上更稳定。
- **标准化后结果差异大**：Z-score + L2 与 raw L2 的重叠率仅 38.96%，说明标准化显著改变了检索排序。虽然其速度最快（0.0155 s），但在没有标签验证的情况下，难以判断哪种排序更符合评测系统的"相似"标准。
- **Manhattan (L1)** 与 baseline 重叠率最低（72.78% / 39.96%），偏离较大，风险较高。

## 5. 结论与最终方案

- **Cosine (raw) 是最稳健的选择**：与 baseline 结果一致性极高（99.68%），同时保留了 `sklearn` 带来的速度优势。由于评分只看准确率是否超过 baseline，而 Cosine 的结果几乎与 baseline 相同，此方案胜率极高。
- **Normalized L2 是潜在备选**：如果后续 OJ 反馈需要进一步调优，可尝试 Z-score + L2，因为它在标准化场景下速度最快，且可能因消除量纲差异而略有提升。

**最终选择**：采用 **Cosine (raw)** 作为 `retrieval.py` 的提交方案。

## 6. 最终提交文件

- `retrieval.py`：基于 `sklearn.neighbors.NearestNeighbors`，metric=`cosine`
- `compare_metrics.py`：本地多指标对比脚本
- `experiment_log.md`：本实验记录文件
