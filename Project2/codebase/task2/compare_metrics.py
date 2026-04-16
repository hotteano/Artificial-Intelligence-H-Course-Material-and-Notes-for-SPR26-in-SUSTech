import pickle
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def _load_data(file_name):
    with open(file_name, 'rb') as f:
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

    # Compare overlaps with L2 raw baseline
    baseline_indices = results[0]["indices"]
    print("\n" + "=" * 60)
    print("OVERLAP WITH L2 (RAW) BASELINE")
    print("=" * 60)
    for res in results[1:]:
        overlap = compute_overlap(baseline_indices, res["indices"])
        print(f"  {res['name']:<30} overlap: {overlap * 100:.2f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Method':<35} {'Top-1':>8} {'Top-5':>8} {'MRR':>8} {'Time(s)':>10}")
    print("-" * 75)
    for res in results:
        print(
            f"{res['name']:<35} {res['top1']:>7.2f}% {res['top5']:>7.2f}% {res['mrr']:>8.4f} {res['time']:>10.4f}"
        )


if __name__ == "__main__":
    main()
