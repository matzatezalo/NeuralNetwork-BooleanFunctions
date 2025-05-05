import numpy as np
import matplotlib.pyplot as plt

from models.linear_perceptron import Perceptron
from models.tiny_mlp import TinyMLP

np.random.seed(42);

def run(X, y, model, gate_name, lr, epochs):
    network = model()
    loss = network.fit(X, y[gate_name], lr, epochs)
    
    preds = network.predict(X).ravel().tolist()
    acc = (np.array(preds) == y[gate_name].ravel()).mean()

    return loss, preds, acc

def main():
    X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)              # shape (4,2)

    # Target is 4x1 so we can feed it to binary cross-entropy
    y = {
        "AND": np.array([[0], [0], [0], [1]], dtype=float),
        "OR" : np.array([[0], [1], [1], [1]], dtype=float),
        "XOR": np.array([[0], [1], [1], [0]], dtype=float)
    }

    for gate in ["AND", "OR"]:
        for model in [Perceptron, TinyMLP]:
            loss, preds, acc = run(X, y, model, gate, 0.1, 10000)
            print(f"{model.__name__:10} on {gate}: "
                  f"accuracy = {acc}, predictions = {preds}")

    # XOR separated to see the difference in models
    print("\nXOR test")
    for model in [Perceptron, TinyMLP]:
        loss, preds, acc = run(X, y, model, "XOR", 0.1, 10000)
        print(f"{model.__name__:10} on XOR: "
                  f"accuracy = {acc}, predictions = {preds}")


if __name__ == "__main__":        
    main()

    

