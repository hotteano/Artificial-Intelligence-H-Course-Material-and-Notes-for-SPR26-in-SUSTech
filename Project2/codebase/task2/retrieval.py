import pickle
from typing import List
import numpy as np
from pathlib import Path
import os
from sklearn.neighbors import NearestNeighbors


class Retrieval:
    def __init__(self, repository_data = None):
        """
        Initialize the retrieval model.
        """
        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine', n_jobs=1)
        root_path = os.path.dirname(os.path.abspath(__file__))
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
