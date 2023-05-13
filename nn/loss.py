from abc import abstractmethod

import numpy as np

from nn.common import Differentiable


class Loss(Differentiable):
    """
    This abstract class must be implemented by concrete Loss classes
    """

    @abstractmethod
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...


class BinaryCrossEntropy(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        loss = (
            np.multiply(labels, np.log(predictions + epsilon))
            + np.multiply(1 - labels, np.log(1 - predictions + epsilon))
        )
        return -1 * np.mean(loss)

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        return -1 * (
            np.divide(labels, predictions + epsilon) -
            np.divide(1 - labels, 1 - predictions + epsilon)
        )


class CategoricalCrossEntropy(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        loss = -np.mean(np.sum(labels * np.log(predictions), axis=0))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        gradients = -labels / predictions
        gradients = np.nan_to_num(gradients, nan=0.0)
        return gradients


class SparseCategoricalCrossentropy(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        loss = - \
            np.mean(np.log(predictions[np.arange(labels.shape[0]), labels]))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        grad = np.zeros_like(predictions)
        grad[np.arange(labels.shape[0]), labels] = -1 / \
            predictions[np.arange(labels.shape[0]), labels]
        return grad


class KLDivergence(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0)
        labels = np.clip(labels, epsilon, 1.0)
        loss = np.sum(labels * np.log(labels / predictions))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1.0)
        labels = np.clip(labels, epsilon, 1.0)
        return -labels / predictions


class Poisson(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, np.inf)
        loss = np.mean(predictions - labels * np.log(predictions))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, np.inf)
        return predictions - labels / predictions


class MeanSquaredError(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.mean(np.square(predictions - labels))

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return 2 * (predictions - labels) / labels.size


class MeanAbsoluteError(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(predictions - labels))

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.sign(predictions - labels) / labels.size


class RootMeanSquaredError(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        loss = np.sqrt(np.mean(np.square(predictions - labels)))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return (predictions - labels) / (2 * labels.shape[1] * np.sqrt(np.mean(np.square(predictions - labels))))


class MeanAbsolutePercentageError(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        loss = np.mean(np.abs(predictions - labels) /
                       np.maximum(np.abs(labels), np.finfo(np.float64).eps)) * 100
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.sign(predictions - labels) / (labels * labels.shape[1])


class MeanSquaredLogarithmicError(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        loss = np.mean(np.square(np.log1p(predictions) - np.log1p(labels)))
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return (2 * (np.log1p(predictions) - np.log1p(labels))) / labels.shape[1]


class CosineSimilarity(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        predictions_norm = np.linalg.norm(predictions, axis=0)
        labels_norm = np.linalg.norm(labels, axis=0)
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        similarity = np.sum(predictions * labels, axis=0) / \
            (predictions_norm * labels_norm + epsilon)
        loss = -np.mean(similarity)
        return loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        predictions_norm = np.linalg.norm(predictions, axis=0)
        labels_norm = np.linalg.norm(labels, axis=0)
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        similarity = np.sum(predictions * labels, axis=0) / \
            (predictions_norm * labels_norm + epsilon)
        grad = -labels / (predictions_norm * labels_norm + epsilon) - \
            predictions * similarity / (predictions_norm ** 3 + epsilon)
        return grad


class HuberLoss(Loss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        diff = np.abs(predictions - labels)
        quadratic = np.minimum(diff, self.delta)
        linear = diff - quadratic
        return np.mean(0.5 * np.square(quadratic) + self.delta * linear)

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        diff = predictions - labels
        is_small_error = np.logical_and(
            np.abs(diff) <= self.delta, np.abs(diff) != 0)
        gradient = np.zeros_like(diff)
        gradient[is_small_error] = diff[is_small_error]
        gradient[~is_small_error] = self.delta * np.sign(diff[~is_small_error])
        return gradient / labels.size
