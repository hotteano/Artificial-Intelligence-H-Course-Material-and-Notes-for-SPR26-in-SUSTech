import numpy as np
import pickle
from pathlib import Path
import os


class Classifier:
    def __init__(self):
        """
        You can load the model as a member variable while instantiation the classifier
        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = pickle.load(open(Path(root_path, 'classification_model.pkl'), 'rb'))
        self.mean = pickle.load(open(Path(root_path, 'classification_mean.pkl'), 'rb'))
        self.std_dev = pickle.load(open(Path(root_path, 'classification_std.pkl'), 'rb'))

    def inference(self, X: np.array) -> np.array:
        """
        Inference a single data
        Args:
            X:  All the feature vectors with dim=256 of the data which needs to be classified, X.shape=[a, 256], a is the
                number of the test data.

        Returns:
            All classification results, is an int vector with dim=a, where a is the number of the test data. The ith
            element of the results vector is the classification result of ith test data, which is the index of the
            category.
        """
        std_safe = np.where(self.std_dev == 0, 1.0, self.std_dev)
        return self.model.predict((X - self.mean) / std_safe)


def _load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def _save_data(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {file_name}")


def _split_train_validation(data, labels, train_ratio=0.8, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_train_samples = int(train_ratio * num_samples)
    train_indices = indices[:num_train_samples]
    validation_indices = indices[num_train_samples:]
    return (data[train_indices], labels[train_indices]), (data[validation_indices], labels[validation_indices])


def main():
    # 1. Data Loading
    print("Loading data...")
    classification_train_data = _load_data("classification_train_data.pkl")
    classification_train_label = _load_data("classification_train_label.pkl")

    # 2. Data Preprocessing: remove index column
    classification_train_data = classification_train_data[:, 1:]
    classification_train_label = classification_train_label[:, 1:].reshape(-1)

    print("Classification Train Data Shape:", classification_train_data.shape)
    print("Classification Train Label Shape:", classification_train_label.shape)

    # 3. Normalization
    mean = np.mean(classification_train_data, axis=0)
    std_dev = np.std(classification_train_data, axis=0)
    std_dev[std_dev == 0] = 1.0
    classification_train_data = (classification_train_data - mean) / std_dev

    # 4. Dataset Splitting (same as notebook baseline)
    train_ratio = 0.8
    seed = 123
    (train_data, train_labels), (validation_data, validation_labels) = _split_train_validation(
        classification_train_data, classification_train_label,
        train_ratio=train_ratio, random_seed=seed
    )

    print(f"Train samples: {len(train_labels)}, Val samples: {len(validation_labels)}")
    print("-" * 60)

    # 5. Train MLP (128) and evaluate on validation set
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=1000,
        early_stopping=True,
        random_state=seed
    )

    print("Training MLP (128) ...")
    model.fit(train_data, train_labels)

    train_acc = model.score(train_data, train_labels)
    val_acc = model.score(validation_data, validation_labels)
    print(f"  -> Train accuracy: {train_acc:.4f}")
    print(f"  -> Val accuracy:   {val_acc:.4f}")

    # 6. Retrain on ALL data and save artifacts
    print("Retraining MLP (128) on full dataset...")
    final_model = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=1000,
        early_stopping=True,
        random_state=seed
    )
    final_model.fit(classification_train_data, classification_train_label)

    _save_data("classification_model.pkl", final_model)
    _save_data("classification_mean.pkl", mean)
    _save_data("classification_std.pkl", std_dev)
    print("Done! You can now submit classifier.py along with the .pkl files.")


if __name__ == "__main__":
    main()
