import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import util
from NNS import NNS
from retrieval import Retrieval

print("=" * 60)
print("Loading data...")
repo_data = util.load_data('./image_retrieval_repository_data.pkl')
print(f"Raw repo data shape: {repo_data.shape}")

# Drop index column
repo_features = repo_data[:, 1:]
print(f"Feature shape (after dropping index): {repo_features.shape}")

# Use first 100 samples as queries for quick comparison
query_features = repo_features[:100]

print("=" * 60)
print("Testing Retrieval (cosine, sklearn NearestNeighbors)...")
try:
    start = time.time()
    retrieval_model = Retrieval(repository_data=repo_data)
    retrieval_results = retrieval_model.inference(query_features)
    retrieval_time = time.time() - start
    print(f"  SUCCESS! Time: {retrieval_time:.3f}s")
    print(f"  Results shape: {retrieval_results.shape}")
    print(f"  First query top-5 indices: {retrieval_results[0]}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    retrieval_results = None

print("=" * 60)
print("Testing Baseline NNS (L2, brute force)...")
try:
    start = time.time()
    nns_model = NNS(k=5)
    nns_model.fit(X_train=repo_features)
    nns_results = nns_model.predict(query_features)
    nns_time = time.time() - start
    print(f"  SUCCESS! Time: {nns_time:.3f}s")
    print(f"  Results shape: {nns_results.shape}")
    print(f"  First query top-5 indices: {nns_results[0]}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    nns_results = None

print("=" * 60)
if retrieval_results is not None and nns_results is not None:
    print("Comparing Retrieval vs Baseline NNS...")

    # Compute overlap ratio per query
    overlaps = []
    for i in range(len(query_features)):
        set_r = set(retrieval_results[i])
        set_n = set(nns_results[i])
        overlap = len(set_r & set_n)
        overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)
    print(f"  Average top-5 overlap between cosine and L2: {avg_overlap:.2f} / 5")
    print(f"  Overlap distribution:")
    for k in range(6):
        count = sum(1 for o in overlaps if o == k)
        print(f"    {k} common items: {count} queries")

    # Check if results are identical (they shouldn't be, different metrics)
    identical = np.array_equal(retrieval_results, nns_results)
    print(f"  Results completely identical: {identical}")

print("=" * 60)
print("Done.")
