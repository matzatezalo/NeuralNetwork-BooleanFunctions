import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    # derivative wrt *activation* value a = σ(z):  a(1‑a)
    return a * (1 - a)

def bce_loss(p, t):
    # Binary‑cross‑entropy averaged over samples
    p = np.clip(p, 1e-9, 1-1e-9)          # to avoid log(0)
    return -(t*np.log(p) + (1-t)*np.log(1-p)).mean()

def add_noise(X, prob_flip=0.1, seed=None):
        # Flip each bit with 10% probability - prob_flip
        rng = np.random.default_rng(seed)
        Xn = X.copy()
        flips = rng.random(X.shape) < prob_flip
        return np.where(flips, 1 - Xn, Xn)

def plot_boundary(model, title="", h=0.01, padding=0.5):
    """
    Visualise the model's 0/1 decision regions in 2D.

    Parameters
    ----------
    model   : object with .predict(X) -> {0,1}
    title   : str  figure title
    h       : float  grid step size
    padding : float  extra margin around data points
    """
    # --- gather the original XOR points to set axis limits ---------------
    X_orig = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)

    x_min, x_max = X_orig[:,0].min()-padding, X_orig[:,0].max()+padding
    y_min, y_max = X_orig[:,1].min()-padding, X_orig[:,1].max()+padding

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(4,3.5))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.1,0.1,1.1])
    plt.scatter(X_orig[:,0], X_orig[:,1], c=[0,1,1,0], s=80, edgecolor="k")
    plt.xticks([0,1]); plt.yticks([0,1])
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
    plt.title(title); plt.gca().set_aspect("equal", "box")
    plt.tight_layout(); plt.show()