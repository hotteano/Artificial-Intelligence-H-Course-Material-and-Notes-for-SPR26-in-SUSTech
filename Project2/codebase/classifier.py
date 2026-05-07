import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier


class Classifier:
    def __init__(self):
        """
        Initialize the classifier.
        """
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=2000,
            early_stopping=True,
            random_state=123
        )
        self.fit()

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None):
        """
        Load data, preprocess, and train the model.

        Parameters:
        - X_train: Training feature data (optional).
        - y_train: Training labels (optional). Supports both integer labels
                   and one-hot encoded labels (like SoftmaxRegression).
        - X_val: Validation feature data (optional).
        - y_val: Validation labels (optional).

        If X_train and y_train are provided, they are used directly.
        Otherwise, the data is loaded from the default paths in the directory.
        """
        if X_train is not None and y_train is not None:
            classification_train_data = X_train
            classification_train_label = y_train
            # Handle one-hot encoded labels, consistent with SoftmaxRegression
            if classification_train_label.ndim > 1 and classification_train_label.shape[1] > 1:
                classification_train_label = np.argmax(classification_train_label, axis=1)
            else:
                classification_train_label = classification_train_label.reshape(-1)
        else:
            root_path = os.path.dirname(os.path.abspath(__file__))
            train_data_path = os.path.join(root_path, "classification_train_data.pkl")
            train_label_path = os.path.join(root_path, "classification_train_label.pkl")
            
            with open(train_data_path, 'rb') as f:
                classification_train_data = pickle.load(f)
            with open(train_label_path, 'rb') as f:
                classification_train_label = pickle.load(f)

            # Preprocessing: remove index column from local loaded files
            classification_train_data = classification_train_data[:, 1:]
            classification_train_label = classification_train_label[:, 1:].reshape(-1)

        # Normalization
        self.mean = np.mean(classification_train_data, axis=0)
        self.std_dev = np.std(classification_train_data, axis=0)
        self.std_dev[self.std_dev == 0] = 1.0
        
        classification_train_data = (classification_train_data - self.mean) / self.std_dev

        # Train MLP
        self.model.fit(classification_train_data, classification_train_label)

    def inference(self, X: np.array) -> np.array:
        """
        Predict class labels for samples in X.

        Args:
            X: All the feature vectors with dim=256 of the data which needs to be classified.
               X.shape=[a, 256], a is the number of the test data.

        Returns:
            All classification results, an int vector with dim=a, where a is the number of
            the test data. The ith element of the results vector is the classification result
            of ith test data, which is the index of the category.
        """
        std_safe = np.where(self.std_dev == 0, 1.0, self.std_dev)
        return self.model.predict((X - self.mean) / std_safe)
