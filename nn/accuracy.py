import numpy as np


class Accuracy:
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        raise NotImplementedError


class RegressionAccuracy(Accuracy):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        squared_errors = np.square(predictions - labels)
        mse = np.mean(squared_errors)
        accuracy = 1.0 / (1.0 + mse)  # Calculate accuracy using a transformation of MSE
        return accuracy


class BinaryAccuracy:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        predicted_classes = (predictions >= self.threshold).astype(int)
        correct_predictions = np.sum(predicted_classes == labels)
        total_examples = labels.size
        accuracy = correct_predictions / total_examples
        return accuracy


class CategoricalAccuracy(Accuracy):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        correct_predictions = np.sum(predicted_classes == true_classes)
        total_examples = labels.shape[1]
        accuracy = correct_predictions / total_examples
        return accuracy


class SparseCategoricalAccuracy(Accuracy):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        predicted_classes = np.argmax(predictions, axis=1)
        correct_predictions = np.sum(predicted_classes == labels)
        total_examples = labels.shape[0]
        accuracy = correct_predictions / total_examples
        return accuracy


class TopKCategoricalAccuracy(Accuracy):
    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        top_k_predictions = np.argsort(predictions, axis=0)[-self.k:]
        true_classes = np.argmax(labels, axis=0)
        correct_predictions = np.sum(
            np.any(top_k_predictions == true_classes[np.newaxis, :], axis=0))
        total_examples = labels.shape[1]
        accuracy = correct_predictions / total_examples
        return accuracy


class SparseTopKCategoricalAccuracy(Accuracy):
    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        top_k_predictions = np.argsort(predictions, axis=1)[:, -self.k:]
        labels_reshaped = labels.reshape((-1, 1))
        correct_predictions = np.sum(np.any(top_k_predictions == labels_reshaped, axis=1))
        total_examples = labels.shape[0]
        accuracy = correct_predictions / total_examples
        return accuracy
