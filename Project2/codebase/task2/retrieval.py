import pickle
from typing import List
import numpy as np
from pathlib import Path
import os
from sklearn.neighbors import NearestNeighbors


class Retrieval:
    def __init__(self, repository_data):
        """
        You can load the model as a member variable while instantiation the classifier
        Args:
            repository_data:    The image repository which you need to search in. Data content is same with the
                                given file `image_retrieval_repository_data.pkl`

        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine', n_jobs=1)
        retrieval_repository_data = repository_data[:, 1:]
        self.model.fit(X=retrieval_repository_data)

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 images that are most similar to the given image in the repository
        Args:
            X:  All the feature vector of the data which needs to be retrieved the similar images. X.shape=[a, 256],
                a is the number of the data that needs to be retrieved.

        Returns:
            A numpy array with shape=[a, 5], where a is the number of the data that needs to be retrieved. It can
            be seen as a matrix with size=ax5, each row of the matrix is the indices of the 5 images that are most
            similar to the given image in the repository.
        """
        distances, indices = self.model.kneighbors(X)
        return indices


def _load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def main():
    print("Loading repository data...")
    repository_data = _load_data("image_retrieval_repository_data.pkl")
    print(f"Repository shape: {repository_data.shape}")

    # Build retrieval model
    retrieval = Retrieval(repository_data=repository_data)

    # Test on first 1000 samples (same as demo notebook)
    test_queries = repository_data[:1000, 1:]
    print(f"Test queries shape: {test_queries.shape}")

    import time
    start = time.time()
    results = retrieval.inference(test_queries)
    elapsed = time.time() - start

    print(f"Results shape: {results.shape}")
    print(f"Inference time for 1000 queries: {elapsed:.4f}s")
    print(f"First 5 results for query 0: {results[0]}")


if __name__ == "__main__":
    main()
