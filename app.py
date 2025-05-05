import numpy as np
import matplotlib.pyplot as plt

from models.linear_perceptron import Perceptron
from models.tiny_mlp import TinyMLP

np.random.seed(42);

def main():
    X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)              # shape (4,2)

    # Target is 4x1 so we can feed it to binary cross-entropy
    y_and = np.array([[0], [0], [0], [1]], dtype=float)
    y_or  = np.array([[0], [1], [1], [1]], dtype=float)
    y_xor = np.array([[0], [1], [1], [0]], dtype=float)

    

