"""Linear regression from scratch with NumPy only (no sklearn inside the model)."""
import numpy as np


class LinearRegressionGD:
    """Linear regression trained with batch gradient descent.

    Hypothesis:   h(X) = X_b @ theta          (X_b = X with a leading column of ones)
    Cost (MSE):   J(theta) = (1 / 2m) * sum((h - y)^2)
    Gradient:     dJ/dtheta = (1 / m) * X_b.T @ (h - y)
    Update:       theta := theta - alpha * dJ/dtheta
    """

    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None
        self.cost_history = []

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:                      # single feature passed as a vector
            X = X.reshape(-1, 1)
        return self._add_bias(X) @ self.theta   # hypothesis h(X) = X theta

    def cost(self, X_b, y):
        errors = X_b @ self.theta - y
        return (errors @ errors) / (2 * len(y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = self._add_bias(X)
        m = X_b.shape[0]
        self.theta = np.zeros(X_b.shape[1])
        self.cost_history = []

        for _ in range(self.n_iters):
            errors = X_b @ self.theta - y
            grad = (X_b.T @ errors) / m
            self.theta -= self.lr * grad
            self.cost_history.append(self.cost(X_b, y))
        return self

    @property
    def intercept_(self):
        return self.theta[0]

    @property
    def coef_(self):
        return self.theta[1:]


def mse(y_true, y_pred):
    """Mean squared error, computed manually."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(d ** 2))


def r2_score_manual(y_true, y_pred):
    """R^2 = 1 - SS_res / SS_tot, computed manually."""
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)
