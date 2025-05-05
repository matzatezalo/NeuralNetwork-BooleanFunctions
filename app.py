import numpy as np
import matplotlib.pyplot as plt

from models.linear_perceptron import Perceptron
from models.tiny_mlp import TinyMLP
from util import plot_boundary

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
            loss, preds, acc = run(X, y, model, gate, 0.2, 20000)
            print(f"{model.__name__:10} on {gate}: "
                  f"accuracy = {acc}, predictions = {preds}, loss = {loss}")

    # XOR separated to see the difference in models
    print("\nXOR test")
    for model in [Perceptron, TinyMLP]:
        loss, preds, acc = run(X, y, model, "XOR", 0.2, 20000)
        print(f"{model.__name__:10} on XOR: "
                  f"accuracy = {acc}, predictions = {preds}, loss = {loss}")
        
        
    print("-----------------------")
    # ---- Analyse representations ----

    # Display hidden-layer activations for XOR
    mlp = TinyMLP()
    mlp.fit(X, y["XOR"], lr = 0.2, epochs = 20000)
    hidden = mlp.hidden(X)       
    print("Hidden activations:\n", np.round(hidden, 3))

    print("-----------------------")
    # Display weights
    print("W1:", np.round(mlp.W1, 3))
    print("W2:", np.round(mlp.W2.ravel(), 3))

    print("-----------------------")
    # Plot decision regions
    # plot_boundary(perceptron, "Perceptron (fails on XOR)")
    plot_boundary(mlp, "MLP learned XOR boundary")


if __name__ == "__main__":        
    main()

    

