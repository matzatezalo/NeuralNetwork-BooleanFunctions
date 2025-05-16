"""
Two‑layer MLP with a sigmoid hidden activation.
Sufficient to learn all operations (AND, OR, XOR).
"""
import numpy as np
from util import sigmoid
from util import bce_loss
from util import sigmoid_deriv

class TinyMLP:
    def __init__(self, in_dim=2, hidden=3):
        self.W1 = np.random.randn(in_dim, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        
        # To store hidden activations over time
        self.hist = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1          
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.p  = sigmoid(self.z2)
        return self.p

    def fit(self, X, t, lr=0.1, epochs=10_000, snapshot=500):
        n = len(X)
        for epoch in range(1, epochs + 1):
            p = self.forward(X)

            # Backward pass
            dL_dz2 = (p - t) / n                 # shape (4,1)
            grad_W2 = self.a1.T @ dL_dz2         # (hidden,1)
            grad_b2 = dL_dz2.sum(axis=0, keepdims=True)

            dL_da1 = dL_dz2 @ self.W2.T          # (4,hidden)
            dL_dz1 = dL_da1 * sigmoid_deriv(self.a1)
            grad_W1 = X.T @ dL_dz1               # (in_dim,hidden)
            grad_b1 = dL_dz1.sum(axis=0, keepdims=True)

            # Stochastic gradient descent
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

            # Record hidden activations at every 'snapshot'
            if epoch % snapshot == 0 or epoch == epochs:
                # Store the copy so later mutations do not overwrite
                self.hist.append(self.a1.copy())

        return bce_loss(self.forward(X), t)
    
    # Return hidden layer activations σ(z₁) for given inputs X
    # To analyse what each neuron learns
    def hidden(self, X):
        z1 = X @ self.W1 + self.b1          # pre‑activation
        return sigmoid(z1)                  # post‑activation

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)