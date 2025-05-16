import numpy as np
import matplotlib.pyplot as plt

from models.linear_perceptron import Perceptron
from models.tiny_mlp import TinyMLP
from models.tiny_mlp import add_noise
from util import plot_boundary
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

np.random.seed(42);

def run(X, y, model, gate_name, lr, epochs):
    network = model()
    loss = network.fit(X, y[gate_name], lr, epochs)
    
    preds = network.predict(X).ravel().tolist()
    acc = (np.array(preds) == y[gate_name].ravel()).mean()

    return loss, preds, acc

def ablate_unit(network, unit_index, X, y):
    """
    Removes the contribution of one hidden neuron in W2,
    measures accuracy on (X, y), then restores the weight.
    """
    # Save original
    orig = network.W2[unit_index, 0].copy()

    # Ablate
    network.W2[unit_index, 0] = 0.0

    # Re-evaluate
    preds = network.predict(X).ravel()
    acc = (preds == y.ravel()).mean()

    # Restore
    network.W2[unit_index, 0] = orig

    return acc


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

    # Produce noisy data for analysis (10% flips)
    X_noisy = add_noise(X, prob_flip=0.1, seed=42)

    # Get models
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
        
    # Fit model for noisy data (for XOR)
    noisy_model = TinyMLP(hidden=3)
    loss_noisy, preds_noisy, acc_noisy = run(X_noisy, y["XOR"], noisy_model, lr=0.2, epochs=20000, wd=0.1)
    print(f"Noisy data with weight-decay=0.1  loss={loss_noisy:.3f}  acc={acc_noisy:.2f}")
        
    print("-----------------------")
    # ---- Analyse representations ----

    # Display hidden-layer activations for XOR
    mlp = TinyMLP()
    mlp.fit(X, y["XOR"], lr = 0.2, epochs = 20000, snapshot = 1000)
    hidden = mlp.hidden(X)       
    print("Hidden activations:\n", np.round(hidden, 3))
    H_final = mlp.hist[-1]

    print("-----------------------")
    # Display weights
    print("W1:", np.round(mlp.W1, 3))
    print("W2:", np.round(mlp.W2.ravel(), 3))

    print("-----------------------")
    # Plot decision regions
    # plot_boundary(perceptron, "Perceptron (fails on XOR)")
    plot_boundary(mlp, "MLP learned XOR boundary")

    # Run PCA 
    pca = PCA(n_components=2)
    proj = pca.fit_transform(H_final)

    # Plot the 4 points in latent space
    # This shows how mlp's 4 inputs are clustered in 2D PCA embedding of hidden layer
    labels = y["XOR"].ravel().astype(int)
    colors = ["tab:blue" if L==1 else "tab:red" for L in labels]

    plt.figure(figsize=(4,4))
    plt.scatter(proj[:,0], proj[:,1], c=colors, s=100, edgecolor="k")
    for i, coord in enumerate(proj):
        plt.text(coord[0]+0.02, coord[1]+0.02, f"{i:02b}", fontsize=12)
    plt.title("Hidden-layer PCA after XOR training")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()

    # How clusters separate over epochs
    fig, ax = plt.subplots(figsize=(4,4))

    def update_frame(frame_index):
        ax.clear()
        H = mlp.hist[frame_index]
        proj = pca.transform(H)
        ax.scatter(proj[:,0], proj[:,1], c=colors, s=80, edgecolor="k")
        ax.set_title(f"Epoch {(frame_index+1)*1000}")
        ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, update_frame, frames=len(mlp.hist), interval=400)
    plt.show()

    # Single-unit ablation - to see how crucial each hidden neuron is
    # print("\n=== Single-unit ablation on XOR ===")
    # for i in range(mlp.W1.shape[1]):  # number of hidden units
    #     acc_i = ablate_unit(mlp, i, X, y["XOR"])
    #     print(f" Ablate neuron {i}:  accuracy = {acc_i:.2f}")

    # Heat-map of W1 (input→hidden weights)
    """
        Shows which input features each hidden neuron is sensitive to
        (e.g. large positive on both inputs means "fires when either is 1"
    """
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(mlp.W1, cmap="coolwarm", aspect="auto", origin="lower")
    ax.set_xlabel("Hidden unit index")
    ax.set_ylabel("Input feature index")
    ax.set_title("Heat-map of W1 (input→hidden)")
    plt.colorbar(im, ax=ax, label="Weight value")
    plt.tight_layout()
    plt.show()

    # Heat-map of final hidden activations
    """
        Shows actual values of each hidden unit on the 4 XOR input patterns
        Clusters of high/low clearly map to the logic sub-functions
    """
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(H_final.T, cmap="YlGnBu", aspect="auto", origin="lower")
    ax.set_xlabel("Sample index (00,01,10,11)")
    ax.set_ylabel("Hidden unit index")
    ax.set_title("Heat-map of hidden activations after training")
    plt.colorbar(im, ax=ax, label="Activation (σ)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":        
    main()

    

