from typing import List

import numpy as np
import pickle
import os

class Selector:
    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        mask_path = os.path.join(root_path, 'mask_code.pkl')

        if os.path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                self.mask_code = pickle.load(f)
        else:
            self.mask_code = self._generate_mask(root_path)
            with open(mask_path, 'wb') as f:
                pickle.dump(self.mask_code, f)

    def get_mask_code(self) -> np.array:
        """
        Returns: The mask matrix for the indices of the 30 features.
        """
        return self.mask_code

    def _generate_mask(self, root_path):
        # Load validation data and fixed model weights
        with open(os.path.join(root_path, 'classification_validation_data.pkl'), 'rb') as f:
            validation_data = pickle.load(f)
        with open(os.path.join(root_path, 'classification_validation_label.pkl'), 'rb') as f:
            validation_label = pickle.load(f)
        with open(os.path.join(root_path, 'image_recognition_model_weights.pkl'), 'rb') as f:
            weights = pickle.load(f)

        # Drop index column
        X_val = validation_data[:, 1:]  # (N, 256)
        y_true = validation_label[:, 1:].reshape(-1).astype(int)

        N, n_features = X_val.shape
        # Precompute biased input and full logits for fast LOO evaluation
        X_bias = np.hstack([np.ones((N, 1)), X_val])
        logits = X_bias @ weights

        active_features = list(range(n_features))
        mask = np.ones(n_features, dtype=bool)
        target_features = 30

        while len(active_features) > target_features:
            best_acc = -1.0
            best_j = -1
            best_col = -1

            # Leave-One-Out: evaluate removing each active feature
            for j in active_features:
                col = j + 1  # weight row shift because row 0 is bias
                logits_without_j = logits - (X_bias[:, col:col + 1] @ weights[col:col + 1, :])
                acc = np.mean(np.argmax(logits_without_j, axis=1) == y_true)
                if acc > best_acc:
                    best_acc = acc
                    best_j = j
                    best_col = col

            # Permanently remove the least important feature
            logits -= X_bias[:, best_col:best_col + 1] @ weights[best_col:best_col + 1, :]
            mask[best_j] = False
            active_features.remove(best_j)

        return mask.astype(float).reshape(1, n_features)
