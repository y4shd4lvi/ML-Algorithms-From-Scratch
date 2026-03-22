"""
models.py
─────────────────────────────────────────────────────────────
Custom Perceptron (from scratch) + sklearn Logistic Regression.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# ──────────────────────────────────────────────────────────
# Custom Perceptron
# ──────────────────────────────────────────────────────────

class Perceptron:
    """
    Classic Perceptron learning algorithm (Rosenblatt, 1958).

    Assumes binary labels in {-1, +1}.
    Tracks the decision boundary as slope (m) and intercept (b)
    after every epoch so the UI can replay learning progression.
    """

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 50):
        self.lr = learning_rate
        self.n_epochs = n_epochs

        # Populated after fit()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []

        # Boundary snapshots: list of (m, b, accuracy) per epoch
        self.boundary_history: list[tuple[float | None, float, float]] = []

    # ── activation ────────────────────────────────────────
    @staticmethod
    def _step(z: np.ndarray) -> np.ndarray:
        """Step activation: +1 if z >= 0 else -1."""
        return np.where(z >= 0, 1, -1)

    # ── prediction ────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self._step(z)

    # ── decision boundary helpers ─────────────────────────
    def _boundary_params(self) -> tuple[float | None, float]:
        """
        Return (slope m, intercept b) of the decision line
        w[0]*x + w[1]*y + bias = 0  →  y = -(w[0]*x + bias) / w[1]
        Returns (None, 0) when w[1] ≈ 0 (vertical line).
        """
        w = self.weights
        if abs(w[1]) < 1e-10:
            return None, float(-self.bias / (w[0] + 1e-10))
        m = -w[0] / w[1]
        b = -self.bias / w[1]
        return m, b

    # ── training ──────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []
        self.boundary_history = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self._step(np.dot(xi, self.weights) + self.bias)
                if pred != yi:
                    delta = self.lr * yi
                    self.weights += delta * xi
                    self.bias += delta
                    errors += 1
            self.errors_per_epoch.append(errors)

            # Snapshot
            m, b = self._boundary_params()
            acc = np.mean(self.predict(X) == y)
            self.boundary_history.append((m, b, float(acc)))

        return self

    # ── convenience ───────────────────────────────────────
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ──────────────────────────────────────────────────────────
# Logistic Regression wrapper
# ──────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train an sklearn Logistic Regression model.

    Labels may be –1 / +1 or 0 / 1 – sklearn handles both.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def logistic_boundary_params(
    model: LogisticRegression,
) -> tuple[float | None, float]:
    """
    Extract slope (m) and intercept (b) from a fitted binary
    Logistic Regression model.

    Decision boundary: w[0]*x + w[1]*y + intercept = 0
    →  y = -(w[0]/w[1])*x - intercept/w[1]
    """
    coef = model.coef_[0]           # shape (2,)
    intercept = model.intercept_[0]
    if abs(coef[1]) < 1e-10:
        return None, float(-intercept / (coef[0] + 1e-10))
    m = -coef[0] / coef[1]
    b = -intercept / coef[1]
    return m, b