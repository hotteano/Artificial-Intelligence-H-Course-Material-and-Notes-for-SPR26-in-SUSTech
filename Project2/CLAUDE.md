# CLAUDE.md — Project 2

This file provides guidance to Claude Code when working with code in `Project2/`.

## Course Context

This is **SUSTech CS311H: Artificial Intelligence (Spring 2026) — Project 2**.
It consists of three independent machine learning subtasks under `codebase/`.

## Directory Structure

```
Project2/
├── codebase/
│   ├── task1/          # Image Classification
│   ├── task2/          # Image Retrieval
│   └── task3/          # Feature Selection
├── docs/               # Assignment PDFs (password-protected)
└── report/             # Empty placeholder for reports
```

## Common Environment

- **Python**: 3.10
- **Dependencies**: `numpy==1.26.1`, `matplotlib==3.7.1`, `tqdm==4.64.1`, `scikit-learn==1.5.2`
- **Notebooks**: Each task has a `.ipynb` demo showing the baseline pipeline.

---

## Task 1 — Image Classification (`codebase/task1/`)

### Goal
Train a classifier on 256-dimensional image features to predict 10 classes.

### Files
| File | Purpose |
|------|---------|
| `image_classification_demo.ipynb` | Baseline notebook (Softmax Regression) |
| `SoftmaxRegression.py` | Multinomial logistic regression implementation |
| `classifier.py` | **Submission file** — must implement `Classifier` class with `inference(X)` |
| `classification_train_data.pkl` | Training features (shape: 49976 × 257, col 0 is index) |
| `classification_train_label.pkl` | Training labels (shape: 49976 × 2, col 0 is index) |
| `util.py` | Helpers: `load_data`, `save_data`, `split_train_validation`, plotting |

### Key Details
- Data column 0 is an ID/index; actual features are columns 1–256.
- Baseline uses Z-score normalization + Softmax Regression (lr=0.1, 10k iters).
- `Classifier.inference(X)` receives `X.shape = [a, 256]` and returns int predictions of length `a`.
- The submitted `classifier.py` loads pre-trained artifacts (`classification_model.pkl`, `classification_mean.pkl`, `classification_std.pkl`) from its own directory.

### Grading Rule
If test accuracy **exceeds the baseline**, full marks; otherwise **0**.

---

## Task 2 — Image Retrieval (`codebase/task2/`)

### Goal
Given a query image feature, retrieve the **5 most similar images** from a repository.

### Files
| File | Purpose |
|------|---------|
| `image_retrieval_demo.ipynb` | Baseline notebook |
| `NNS.py` | Naïve k-NN implementation (L2 distance, brute force) |
| `retrieval.py` | **Submission file** — must implement `Retrieval` class with `inference(X)` |
| `image_retrieval_repository_data.pkl` | Repository data (shape: 5000 × 257, col 0 is index) |
| `util.py` | Same helpers as task1 |

### Key Details
- Repository has 5000 items, 256 features each (after dropping index column).
- Baseline uses brute-force L2 nearest-neighbor search (`NNS` class, `k=5`).
- `Retrieval.inference(X)` receives `X.shape = [a, 256]` and returns `np.array` of shape `[a, 5]` — each row is the indices of the 5 retrieved repository images.

### Grading Rule
For each query, accuracy = `(correct_retrieved / 5) × 100%`. Average over test set.
If test accuracy **exceeds the baseline**, full marks; otherwise **0**.

---

## Task 3 — Feature Selection (`codebase/task3/`)

### Goal
Select **no more than 30 features** from a 256-dimensional feature vector using a binary mask, such that a fixed pre-trained classifier still performs well.

### Files
| File | Purpose |
|------|---------|
| `feature_selection.ipynb` | Baseline feature selection demo (random selection of 30 indices) |
| `image_recognition.ipynb` | Evaluation notebook — applies mask and tests fixed model |
| `selector.py` | **Submission file** — must implement `Selector` class with `get_mask_code()` |
| `classification_validation_data.pkl` | Validation features (shape: ? × 257) |
| `classification_validation_label.pkl` | Validation labels |
| `image_recognition_model_weights.pkl` | Fixed model weights (cannot retrain) |

### Key Details
- Mask is a binary vector `mask ∈ {0,1}^256`. During inference: `x̌ = x ⊙ mask` (element-wise product).
- Baseline randomly picks 30 indices and sets them to 1.
- `Selector.get_mask_code()` returns the mask vector (should load from `mask_code.pkl` in the same directory).
- At most 30 features may be selected (≤ 30 ones in the mask).

### Grading Rule
Test procedure is identical to `image_recognition.ipynb` but uses hidden test data.
If test accuracy **exceeds the baseline**, full marks; otherwise **0**.

---

## Working Tips

1. **Data preprocessing is consistent across tasks**
   - All `.pkl` data files have an index in column 0. Remember to slice with `[:, 1:]` before training/inference.

2. **Baseline improvements**
   - Task 1: Try hyperparameter tuning, regularization, better initialization, or alternative models (e.g., scikit-learn classifiers).
   - Task 2: Try dimensionality reduction (PCA), different distance metrics (cosine, Manhattan), or approximate NN methods.
   - Task 3: Try greedy forward selection, mutual information, L1 regularization-based selection, or model-based importance scores.

3. **Submission artifacts**
   - `classifier.py` (task1)
   - `retrieval.py` (task2)
   - `selector.py` + `mask_code.pkl` (task3)

4. **Password-protected PDFs**
   - `docs/Project2-Description(1).pdf` and `docs/Project 2 Evaluation Details.pdf` are encrypted and cannot be read directly.
