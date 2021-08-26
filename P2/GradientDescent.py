from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def normalize(x: pd.DataFrame) -> Tuple[np.ndarray, float, float]:
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma


class GradientDescent:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 200,
                 precision: float = 0.00001):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.precision = precision
        self.theta = None
        self.mu = None
        self.sigma = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        x, self.mu, self.sigma = normalize(x)
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.theta = np.zeros(x.shape[1])
        prev_cost = -1
        for _ in range(self.max_iterations):
            predictions = x.dot(self.theta)

            # TODO

            cost = mean_squared_error(y, predictions)
            if abs(cost - prev_cost) < self.precision:
                break
            prev_cost = cost

    def predict(self, x: pd.DataFrame):
        if self.theta is None or self.mu is None or self.sigma is None:
            raise Exception(
                "GradientDescent::predict() called before model was trained")

        x = (x - self.mu) / self.sigma
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        x.dot(self.theta)

        # TODO
