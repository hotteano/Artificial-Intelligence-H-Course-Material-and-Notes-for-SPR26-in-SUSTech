import os
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif


def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_mask(path, mask_1d):
    mask_code = mask_1d.astype(float).reshape(1, -1)
    with open(path, 'wb') as f:
        pickle.dump(mask_code, f)
    return mask_code


def eval_accuracy(X_val, y_true, weights, mask_1d):
    """Evaluate validation accuracy with a binary mask."""
    X_masked = X_val * mask_1d
    X_bias = np.hstack([np.ones((X_val.shape[0], 1)), X_masked])
    logits = X_bias @ weights
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)


# --------------------------------------------------------------------------- #
# Method 1: LOO Backward Elimination (existing)
# --------------------------------------------------------------------------- #
def method_loo(X_val, y_true, weights, target=30):
    N, n_features = X_val.shape
    X_bias = np.hstack([np.ones((N, 1)), X_val])
    logits = X_bias @ weights

    active_features = list(range(n_features))
    mask = np.ones(n_features, dtype=bool)

    pbar = tqdm(total=n_features - target, desc="LOO")
    while len(active_features) > target:
        best_acc = -1.0
        best_j = -1
        best_col = -1
        for j in active_features:
            col = j + 1
            logits_without_j = logits - (X_bias[:, col:col + 1] @ weights[col:col + 1, :])
            acc = np.mean(np.argmax(logits_without_j, axis=1) == y_true)
            if acc > best_acc:
                best_acc = acc
                best_j = j
                best_col = col
        logits -= X_bias[:, best_col:best_col + 1] @ weights[best_col:best_col + 1, :]
        mask[best_j] = False
        active_features.remove(best_j)
        pbar.update(1)
    pbar.close()
    return mask.astype(float)


# --------------------------------------------------------------------------- #
# Method 2: Weight Magnitude (absolute sum over classes)
# --------------------------------------------------------------------------- #
def method_weight_magnitude(X_val, y_true, weights, target=30):
    feature_weights = weights[1:, :]  # drop bias row
    importance = np.sum(np.abs(feature_weights), axis=1)
    top_indices = np.argsort(importance)[-target:]
    mask = np.zeros(X_val.shape[1])
    mask[top_indices] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Method 3: Weighted L2 Norm over classes
# --------------------------------------------------------------------------- #
def method_weight_l2(X_val, y_true, weights, target=30):
    feature_weights = weights[1:, :]
    importance = np.linalg.norm(feature_weights, axis=1)
    top_indices = np.argsort(importance)[-target:]
    mask = np.zeros(X_val.shape[1])
    mask[top_indices] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Method 4: Greedy Forward Selection
# --------------------------------------------------------------------------- #
def method_forward_selection(X_val, y_true, weights, target=30):
    N, n_features = X_val.shape
    X_bias_full = np.hstack([np.ones((N, 1)), X_val])

    selected = []
    remaining = set(range(n_features))

    for _ in tqdm(range(target), desc="Forward"):
        best_acc = -1.0
        best_j = -1
        for j in remaining:
            cols = [0] + [jj + 1 for jj in selected + [j]]
            logits = X_bias_full[:, cols] @ weights[cols, :]
            acc = np.mean(np.argmax(logits, axis=1) == y_true)
            if acc > best_acc:
                best_acc = acc
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)

    mask = np.zeros(n_features)
    mask[selected] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Method 5: Mutual Information
# --------------------------------------------------------------------------- #
def method_mutual_info(X_val, y_true, weights, target=30):
    mi = mutual_info_classif(X_val, y_true, discrete_features=False, random_state=42)
    top_indices = np.argsort(mi)[-target:]
    mask = np.zeros(X_val.shape[1])
    mask[top_indices] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Method 6: Approximate Shapley Value (Monte-Carlo permutation sampling)
# --------------------------------------------------------------------------- #
def method_shapley_approx(X_val, y_true, weights, target=30, n_permutations=500):
    """
    Approximate Shapley values by random permutations.
    For each permutation, we do a forward pass adding features one by one
    and record the marginal accuracy gain.
    """
    N, n_features = X_val.shape
    X_bias_full = np.hstack([np.ones((N, 1)), X_val])

    # baseline: no features (only bias)
    logits_bias = X_bias_full[:, 0:1] @ weights[0:1, :]
    acc_empty = np.mean(np.argmax(logits_bias, axis=1) == y_true)

    shapley_values = np.zeros(n_features)

    for _ in tqdm(range(n_permutations), desc="Shapley"):
        perm = np.random.permutation(n_features)
        prev_acc = acc_empty
        current_set = [0]  # bias always included
        for j in perm:
            current_set.append(j + 1)
            logits = X_bias_full[:, current_set] @ weights[current_set, :]
            acc = np.mean(np.argmax(logits, axis=1) == y_true)
            marginal = acc - prev_acc
            shapley_values[j] += marginal
            prev_acc = acc

    shapley_values /= n_permutations
    top_indices = np.argsort(shapley_values)[-target:]
    mask = np.zeros(n_features)
    mask[top_indices] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Method 7: Correlation with label (max absolute correlation across classes)
# --------------------------------------------------------------------------- #
def method_correlation(X_val, y_true, weights, target=30):
    n_features = X_val.shape[1]
    cors = np.zeros(n_features)
    for j in range(n_features):
        # Compute absolute correlation between feature j and one-hot labels
        # Using a simple proxy: correlation with integer label
        cors[j] = np.abs(np.corrcoef(X_val[:, j], y_true)[0, 1])
    top_indices = np.argsort(cors)[-target:]
    mask = np.zeros(n_features)
    mask[top_indices] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Main comparison
# --------------------------------------------------------------------------- #
def main():
    root_path = os.path.dirname(os.path.abspath(__file__))

    validation_data = load_data(os.path.join(root_path, "classification_validation_data.pkl"))
    validation_label = load_data(os.path.join(root_path, "classification_validation_label.pkl"))
    weights = load_data(os.path.join(root_path, "image_recognition_model_weights.pkl"))

    X_val = validation_data[:, 1:]
    y_true = validation_label[:, 1:].reshape(-1).astype(int)

    methods = {
        "LOO_Backward": method_loo,
        "Weight_Magnitude": method_weight_magnitude,
        "Weight_L2": method_weight_l2,
        "Forward_Selection": method_forward_selection,
        "Mutual_Info": method_mutual_info,
        "Shapley_Approx_500": method_shapley_approx,
        "Correlation": method_correlation,
    }

    results = []
    for name, func in methods.items():
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        t0 = time.time()
        if name == "Shapley_Approx_50":
            mask_1d = func(X_val, y_true, weights, target=30, n_permutations=50)
        else:
            mask_1d = func(X_val, y_true, weights, target=30)
        elapsed = time.time() - t0
        acc = eval_accuracy(X_val, y_true, weights, mask_1d)
        results.append((name, acc, elapsed, mask_1d))
        print(f"[{name}] Accuracy: {acc:.4f} | Time: {elapsed:.2f}s")

    # Rank results
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print("FINAL RANKING")
    print(f"{'='*60}")
    for rank, (name, acc, elapsed, _) in enumerate(results, 1):
        print(f"#{rank} {name:25s} | Acc: {acc:.4f} | Time: {elapsed:.2f}s")

    # Save the best mask as mask_code.pkl
    best_name, best_acc, _, best_mask = results[0]
    best_path = os.path.join(root_path, "mask_code.pkl")
    save_mask(best_path, best_mask)
    print(f"\nBest method '{best_name}' mask saved to {best_path}")


if __name__ == "__main__":
    main()
