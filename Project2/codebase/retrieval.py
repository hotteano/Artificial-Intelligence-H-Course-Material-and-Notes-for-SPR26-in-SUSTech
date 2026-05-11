import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
import os


class Retrieval:
    def __init__(self, repository_data=None):
        """
        Initialize the retrieval model with PCA / LLE dimensionality reduction.
        """
        # Drop index column
        X = repository_data[:, 1:].astype(np.float64)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        method = 'lle'

        self.dr_model = None
        X_reduced = X_scaled

        if method == 'pca':
            n_components = 0.95
            self.dr_model = PCA(n_components=n_components, random_state=42)
            X_reduced = self.dr_model.fit_transform(X_scaled)

        elif method == 'lle':
            n_components = 20
            n_neighbors = 30
            self.dr_model = LocallyLinearEmbedding(
                n_neighbors=n_neighbors,
                n_components=n_components,
                method='modified',      # supports transform() for new queries
                eigen_solver='dense',
                random_state=42,
                n_jobs=1
            )
            X_reduced = self.dr_model.fit_transform(X_scaled)

        # Fit nearest neighbors in the reduced space
        self.model = NearestNeighbors(
            n_neighbors=5,
            algorithm='brute',
            metric='cosine',
            n_jobs=1
        )
        self.model.fit(X=X_reduced)

    def inference(self, X: np.array) -> np.array:
        """
        Find the 5 most similar images for each query.
        """
        X = np.asarray(X, dtype=np.float64)

        # Apply same preprocessing as repository
        X_scaled = self.scaler.transform(X)

        if self.dr_model is not None:
            X_reduced = self.dr_model.transform(X_scaled)
        else:
            X_reduced = X_scaled

        distances, indices = self.model.kneighbors(X_reduced)
        return indices
