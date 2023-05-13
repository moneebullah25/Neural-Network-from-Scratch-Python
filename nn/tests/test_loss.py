import unittest
import numpy as np
from nn.loss import *


class TestLossFunctions(unittest.TestCase):
    def test_binary_cross_entropy(self):
        bce = BinaryCrossEntropy()
        predictions = np.array([0.9, 0.1, 0.8])
        labels = np.array([1, 0, 1])
        loss = bce(predictions, labels)
        expected_loss = -1 * np.mean(
            [np.log(0.9), np.log(1 - 0.1), np.log(0.8)]
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_categorical_cross_entropy(self):
        cce = CategoricalCrossEntropy()
        predictions = np.array([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]])
        labels = np.array([[0, 1, 0], [1, 0, 0]])
        expected_loss = -np.mean(np.sum(labels * np.log(predictions), axis=0))
        calculated_loss = cce(predictions, labels)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_sparse_categorical_crossentropy(self):
        sce = SparseCategoricalCrossentropy()
        predictions = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        labels = np.array([0, 0, 1])
        loss = sce(predictions, labels)
        expected_loss = -np.mean(
            [np.log(0.9), np.log(0.8), np.log(0.3)]
        )
        self.assertAlmostEqual(loss, expected_loss)

    def test_kl_divergence(self):
        kld = KLDivergence()
        predictions = np.array([0.4, 0.6])
        labels = np.array([0.5, 0.5])
        loss = kld(predictions, labels)
        expected_loss = np.sum(labels * np.log(labels / predictions))
        self.assertAlmostEqual(loss, expected_loss)

    def test_poisson(self):
        p = Poisson()
        predictions = np.array([1, 2, 3])
        labels = np.array([1, 2, 3])
        loss = p(predictions, labels)
        expected_loss = np.mean(predictions - labels * np.log(predictions))
        self.assertAlmostEqual(loss, expected_loss)

    def test_mean_squared_error(self):
        mse = MeanSquaredError()
        predictions = np.array([1, 2, 3])
        labels = np.array([1, 2, 3])
        loss = mse(predictions, labels)
        expected_loss = np.mean(np.square(predictions - labels))
        self.assertAlmostEqual(loss, expected_loss)

    def test_mean_absolute_error(self):
        mae = MeanAbsoluteError()
        predictions = np.array([1, 2, 3])
        labels = np.array([1, 2, 3])
        loss = mae(predictions, labels)
        expected_loss = np.mean(np.abs(predictions - labels))
        self.assertAlmostEqual(loss, expected_loss)

    def test_mean_absolute_percentage_error(self):
        mape = MeanAbsolutePercentageError()
        predictions = np.array([100, 200])
        labels = np.array([150, 250])
        loss = mape(predictions, labels)
        expected_loss = (np.mean(np.abs(predictions - labels) /
                         np.maximum(np.abs(labels), np.finfo(np.float64).eps)) * 100)
        self.assertAlmostEqual(loss, expected_loss)

    def test_root_mean_squared_error(self):
        rmse = RootMeanSquaredError()
        predictions = np.array([1, 2])
        labels = np.array([1.5, 2.5])
        loss = rmse(predictions, labels)
        expected_loss = np.sqrt(np.mean(np.square(predictions - labels)))
        self.assertAlmostEqual(loss, expected_loss)

    def test_mean_squared_logarithmic_error(self):
        msle = MeanSquaredLogarithmicError()
        predictions = np.array([1e-7, 1e-6])
        labels = np.array([1e-6, 1e-7])
        loss = msle(predictions, labels)
        expected_loss = (
            np.mean(np.square(np.log1p(predictions) - np.log1p(labels))))
        self.assertAlmostEqual(loss, expected_loss)

    def test_cosine_similarity(self):
        csimilarity = CosineSimilarity()
        predictions = np.array([[1], [2]])
        labels = np.array([[2], [4]])
        loss = csimilarity(predictions, labels)
        similarity = (predictions*labels).sum() / \
            (np.linalg.norm(predictions)*np.linalg.norm(labels))
        expected_loss = -np.mean(similarity)
        self.assertAlmostEqual(loss, expected_loss)

    def test_huber_loss(self):
        hloss = HuberLoss(delta=1.)
        predictions = np.array([4., 5., 6., 7., 8., 9., 10., 11.])
        labels = np.array([4., 4., 4., 4., 4., 4., 4., 4.])
        loss = hloss(predictions, labels)
        diff = np.abs(predictions - labels)
        quadratic = np.minimum(diff, hloss.delta)
        linear = diff - quadratic
        expected_loss = (0.5 * np.square(quadratic) +
                         hloss.delta * linear).mean()
        self.assertAlmostEqual(loss, expected_loss)


if __name__ == "__main__":
    unittest.main()
