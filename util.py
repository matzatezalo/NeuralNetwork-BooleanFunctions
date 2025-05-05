import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    # derivative wrt *activation* value a = σ(z):  a(1‑a)
    return a * (1 - a)

def bce_loss(p, t):
    # Binary‑cross‑entropy averaged over samples
    p = np.clip(p, 1e-9, 1-1e-9)          # to avoid log(0)
    return -(t*np.log(p) + (1-t)*np.log(1-p)).mean()