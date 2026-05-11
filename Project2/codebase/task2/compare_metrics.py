import pickle
import numpy as np
import time
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
from pathlib import Path


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_data(file_name):
    path = os.path.join(_SCRIPT_DIR, file_name)
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_retrieval(model, queries, ground_truth_indices, name="Metric"):
    """
    Evaluate retrieval performance using self-retrieval as proxy.
    For query i, the ideal first match is ground_truth_indices[i].
    Here ground_truth_indices is simply np.arange(len(queries)) because
    we query against the repository itself.
    """
    start = time.time()
    distances, indices = model.kneighbors(queries)
    elapsed = time.time() - start

    n = len(queries)
    top1_hits = 0
    top5_hits = 0
    rr_sum = 0.0

    for i in range(n):
        gt = ground_truth_indices[i]
        retrieved = indices[i]
        if retrieved[0] == gt:
            top1_hits += 1
        if gt in retrieved:
            top5_hits += 1
            rank = np.where(retrieved == gt)[0][0]
            rr_sum += 1.0 / (rank + 1)

    top1_acc = top1_hits / n * 100
    top5_acc = top5_hits / n * 100
    mrr = rr_sum / n

    print(f"\n[{name}]")
    print(f"  Inference time : {elapsed:.4f}s ({elapsed / n * 1000:.3f} ms/query)")
    print(f"  Top-1 self-hit : {top1_acc:.2f}%")
    print(f"  Top-5 self-hit : {top5_acc:.2f}%")
    print(f"  MRR            : {mrr:.4f}")
    print(f"  Avg distance   : {np.mean(distances):.4f}")

    return {
        "name": name,
        "time": elapsed,
        "top1": top1_acc,
        "top5": top5_acc,
        "mrr": mrr,
        "indices": indices,
    }


def compute_overlap(results_a, results_b):
    """Average Jaccard-like overlap of top-5 sets."""
    n = len(results_a)
    overlaps = []
    for i in range(n):
        set_a = set(results_a[i])
        set_b = set(results_b[i])
        inter = len(set_a & set_b)
        overlaps.append(inter / 5.0)
    return np.mean(overlaps)


def main():
    print("Loading repository data...")
    repository_data = _load_data("image_retrieval_repository_data.pkl")
    print(f"Repository shape: {repository_data.shape}")

    # Remove index column
    repo_features = repository_data[:, 1:]

    # Use first 1000 samples as queries (self-retrieval)
    n_queries = 1000
    queries = repo_features[:n_queries]
    gt_indices = np.arange(n_queries)

    metrics_to_test = [
        ("euclidean", "L2 (Baseline-like)"),
        ("cosine", "Cosine"),
        ("manhattan", "Manhattan (L1)"),
        ("correlation", "Correlation"),
    ]

    results = []
    print("\n" + "=" * 60)
    print("RAW FEATURES")
    print("=" * 60)
    for metric, label in metrics_to_test:
        nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric=metric, n_jobs=1)
        nn.fit(repo_features)
        res = evaluate_retrieval(nn, queries, gt_indices, name=f"{label} | raw")
        results.append(res)

    # Standardized features
    mean = np.mean(repo_features, axis=0)
    std = np.std(repo_features, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    repo_norm = (repo_features - mean) / std_safe
    queries_norm = repo_norm[:n_queries]

    print("\n" + "=" * 60)
    print("Z-SCORE NORMALIZED FEATURES")
    print("=" * 60)
    for metric, label in metrics_to_test:
        nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric=metric, n_jobs=1)
        nn.fit(repo_norm)
        res = evaluate_retrieval(nn, queries_norm, gt_indices, name=f"{label} | norm")
        results.append(res)

    # PCA reduced features
    print("\n" + "=" * 60)
    print("PCA REDUCED FEATURES")
    print("=" * 60)
    scaler_pca = StandardScaler()
    repo_pca_scaled = scaler_pca.fit_transform(repo_features)
    queries_pca_scaled = repo_pca_scaled[:n_queries]

    pca_configs = [
        (0.90, "PCA 90% variance"),
        (0.95, "PCA 95% variance"),
        (64, "PCA 64-dim"),
        (32, "PCA 32-dim"),
    ]
    for n_comp, label in pca_configs:
        pca = PCA(n_components=n_comp, random_state=42)
        repo_reduced = pca.fit_transform(repo_pca_scaled)
        queries_reduced = repo_reduced[:n_queries]
        nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine', n_jobs=1)
        nn.fit(repo_reduced)
        res = evaluate_retrieval(nn, queries_reduced, gt_indices, name=f"Cosine | {label}")
        results.append(res)

    # LLE reduced features
    print("\n" + "=" * 60)
    print("LLE REDUCED FEATURES")
    print("=" * 60)
    scaler_lle = StandardScaler()
    repo_lle_scaled = scaler_lle.fit_transform(repo_features)
    queries_lle_scaled = repo_lle_scaled[:n_queries]

    lle_configs = [
        (15, 25),
        (20, 30),
        (30, 40),
        (31, 33),   # Tuned from Lab10 (Olivetti Faces)
    ]
    for n_comp, n_nei in lle_configs:
        print(f"\nFitting LLE (n_components={n_comp}, n_neighbors={n_nei}) ...")
        lle = LocallyLinearEmbedding(
            n_neighbors=n_nei,
            n_components=n_comp,
            method='modified',
            eigen_solver='dense',
            random_state=42,
            n_jobs=1
        )
        repo_reduced = lle.fit_transform(repo_lle_scaled)
        queries_reduced = lle.transform(queries_lle_scaled)
        nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine', n_jobs=1)
        nn.fit(repo_reduced)
        res = evaluate_retrieval(nn, queries_reduced, gt_indices, name=f"Cosine | LLE {n_comp}d/{n_nei}n")
        results.append(res)

    # Compare overlaps with L2 raw baseline
    baseline_indices = results[0]["indices"]
    print("\n" + "=" * 60)
    print("OVERLAP WITH L2 (RAW) BASELINE")
    print("=" * 60)
    for res in results[1:]:
        overlap = compute_overlap(baseline_indices, res["indices"])
        print(f"  {res['name']:<40} overlap: {overlap * 100:.2f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Method':<45} {'Top-1':>8} {'Top-5':>8} {'MRR':>8} {'Time(s)':>10}")
    print("-" * 85)
    for res in results:
        print(
            f"{res['name']:<45} {res['top1']:>7.2f}% {res['top5']:>7.2f}% {res['mrr']:>8.4f} {res['time']:>10.4f}"
        )


if __name__ == "__main__":
    main()
