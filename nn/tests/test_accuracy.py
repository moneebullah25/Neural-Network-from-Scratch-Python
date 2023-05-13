import unittest
import numpy as np
from nn.accuracy import RegressionAccuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy, TopKCategoricalAccuracy, SparseTopKCategoricalAccuracy


class TestAccuracyAlgorithms(unittest.TestCase):
    def test_regression_accuracy(self):
        predictions = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 2.5, 3.5])
        accuracy_calculator = RegressionAccuracy()
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 0.8571428571428571)

    def test_binary_accuracy(self):
        predictions = np.array([0.2, 0.7, 0.9])
        labels = np.array([0, 1, 1])
        accuracy_calculator = BinaryAccuracy()
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_categorical_accuracy(self):
        predictions = np.array([[0.2, 0.3], [0.4, 0.6], [0.9, 0.1]])
        labels = np.array([[0, 1], [1, 0], [1, 0]])
        accuracy_calculator = CategoricalAccuracy()
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_sparse_categorical_accuracy(self):
        predictions = np.array([[0.2, 0.3], [0.4, 0.6], [0.9, 0.1]])
        labels = np.array([1, 1, 0])
        accuracy_calculator = SparseCategoricalAccuracy()
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_top_k_categorical_accuracy(self):
        predictions = np.array([[0.2, 0.4, 0.9], [0.3, 0.6, 0.1]])
        labels = np.array([[1, 1, 1], [0, 0, 0]])
        accuracy_calculator = TopKCategoricalAccuracy(k=2)
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 1.0)


    def test_sparse_top_k_categorical_accuracy(self):
        predictions = np.array([[0.2, 0.3], [0.4, 0.6], [0.9, 0.1]])
        labels = np.array([1, 1, 0])
        accuracy_calculator = SparseTopKCategoricalAccuracy(k=2)
        accuracy = accuracy_calculator(predictions, labels)
        self.assertAlmostEqual(accuracy, 1.0)
