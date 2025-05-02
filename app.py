import numpy as np

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_deriv(y_true, y_pred):
    return y_pred - y_true

# Input (same for all logic gates)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Multi-output: [AND, OR, XOR]
y = np.array([
    [0, 0, 0],  # 0 AND 0, 0 OR 0, 0 XOR 0
    [0, 1, 1],  # 0 AND 1, 0 OR 1, 0 XOR 1
    [0, 1, 1],  # 1 AND 0, 1 OR 0, 1 XOR 0
    [1, 1, 0]   # 1 AND 1, 1 OR 1, 1 XOR 1
])

# Network architecture
input_dim = 2
hidden_dim = 2
output_dim = 3
lr = 0.1
epochs = 10000

# Weight initialization
np.random.seed(42)
W1 = np.random.randn(hidden_dim, input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.randn(output_dim, hidden_dim)
b2 = np.zeros((output_dim, 1))

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = X[i].reshape(-1, 1)
        target = y[i].reshape(-1, 1)

        # ---- Forward pass ----
        z1 = np.dot(W1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)

        # ---- Loss ----
        loss = mse_loss(target, a2)
        total_loss += loss

        # ---- Backward pass ----
        delta2 = mse_deriv(target, a2) * sigmoid_deriv(z2)
        dW2 = np.dot(delta2, a1.T)
        db2 = delta2

        delta1 = np.dot(W2.T, delta2) * sigmoid_deriv(z1)
        dW1 = np.dot(delta1, x.T)
        db1 = delta1

        # ---- Update ----
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")

# ---- Final predictions ----
print("\nFinal predictions (AND, OR, XOR):")
for i in range(len(X)):
    x = X[i].reshape(-1, 1)
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    print(f"Input: {X[i]} → Predicted: {a2.ravel()} | Target: {y[i]}")

