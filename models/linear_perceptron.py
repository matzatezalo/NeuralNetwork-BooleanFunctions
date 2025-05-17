"""
Single‑layer perceptron.
Learns any linearly separable Boolean gate (AND, OR).
"""
import numpy as np
from util import sigmoid, bce_loss

class Perceptron:
    def __init__(self, in_dim=2):
        # W shape (in_dim, 1),  b shape (1,)
        self.W = np.random.randn(in_dim, 1) * 0.1
        self.b = np.zeros((1,))

    def forward(self, X):
        return sigmoid(X @ self.W + self.b)

    def fit(self, X, t, lr=0.1, epochs=10_000, snapshots = 500, wd=0.0):
        n = len(X)
        for _ in range(epochs):
            p = self.forward(X)

            # Gradients (dL/dW, dL/db)
            dL_dz = (p - t) / n                 # because dσ inverse cancels in bce
            grad_W = X.T @ dL_dz
            grad_b = dL_dz.sum(axis=0)

            # Stochastic gradient descent
            self.W -= lr * grad_W
            self.b -= lr * grad_b
        return bce_loss(self.forward(X), t)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)