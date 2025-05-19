import numpy as np
import matplotlib.pyplot as plt

from models.linear_perceptron import Perceptron
from models.tiny_mlp import TinyMLP
from util import plot_boundary
from util import add_noise
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

np.random.seed(42);

def run(X, y, model, gate_name, lr, epochs, wd):
    network = model()
    loss = network.fit(X, y[gate_name], lr, epochs, 1000,  wd)
    
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

    # Get models
    for gate in ["AND", "OR"]:
        for model in [Perceptron, TinyMLP]:
            loss, preds, acc = run(X, y, model, gate, 0.1, 20000, 0.0)
            print(f"{model.__name__:10} on {gate}: "
                  f"accuracy = {acc}, predictions = {preds}, loss = {loss}")

    np.random.seed(42);
    # XOR separated to see the difference in models
    print("\nXOR test")
    for model in [Perceptron, TinyMLP]:
        loss, preds, acc = run(X, y, model, "XOR", 0.1, 20000, 0.0)
        print(f"{model.__name__:10} on XOR: "
                  f"accuracy = {acc}, predictions = {preds}, loss = {loss}")
        
    print("-----------------------")
    # ---- Analyse representations ----

    # Display hidden-layer activations for XOR
    mlp = TinyMLP()
    mlp.fit(X, y["XOR"], lr = 0.1, epochs = 20000, snapshot = 1000, wd = 0.0)
    hidden = mlp.hidden(X)       
    print("Hidden activations:\n", np.round(hidden, 3))
    H_final = mlp.hist[-1]

    print("-----------------------")
    # Display weights
    print("W1:", np.round(mlp.W1, 3))
    print("W2:", np.round(mlp.W2.ravel(), 3))

    print("-----------------------")
    # Plot decision regions
    perceptron_and = Perceptron()
    perceptron_and.fit(X, y["AND"])

    perceptron_or = Perceptron()
    perceptron_or.fit(X, y["OR"])

    perceptron_xor = Perceptron()
    perceptron_xor.fit(X, y["XOR"])

    plot_boundary(perceptron_and, "Perceptron on AND boundary")
    plot_boundary(perceptron_or, "Perceptron on OR boundary")
    plot_boundary(perceptron_xor, "Perceptron (fails on XOR)")
    plot_boundary(mlp, "MLP learned XOR boundary")

    # Run PCA 
    pca = PCA(n_components=2)
    proj = pca.fit_transform(H_final)

    # Plot the 4 points in latent space
    # This shows how mlp's 4 inputs are clustered in 2D PCA embedding of hidden layer
    labels = y["XOR"].ravel().astype(int)
    colors = ["tab:blue" if L==1 else "tab:red" for L in labels]

    fig, ax = plt.subplots(figsize=(4,4))

    var1, var2 = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var1:.1%} var)")
    ax.set_ylabel(f"PC2 ({var2:.1%} var)")
    ax.set_title("Hidden-space PCA (XOR)")

    # Scatter with bigger points and edgecolor
    ax.scatter(proj[:,0], proj[:,1],
            c=colors, s=120,
            edgecolor="k", linewidth=1.2)

    for i, (x, y_) in enumerate(proj):
        inp = X[i]           # e.g. [0,1]
        label = f"{int(inp[0])}{int(inp[1])}"  

        ax.annotate(
            label,
            xy=(x, y_), 
            xytext=(5, 5),              
            textcoords="offset points",
            ha="left", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6)
        )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(0.2)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Hidden-layer PCA after XOR training", pad=12)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()

    # Animation of how clusters separate over epochs
    fig, ax = plt.subplots(figsize=(4,4))

    def update_frame(frame_index):
        ax.clear()

        # Redraw the static labels
        ax.set_xlabel(f"PC1 ({var1:.1%} var)")
        ax.set_ylabel(f"PC2 ({var2:.1%} var)")
        ax.set_title(f"Epoch {(frame_index+1)*1000}")

        H = mlp.hist[frame_index]
        proj = pca.transform(H)
        ax.scatter(proj[:,0], proj[:,1], c=colors, s=80, edgecolor="k")
        ax.set_title(f"Epoch {(frame_index+1)*1000}")
        ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, update_frame, frames=len(mlp.hist), interval=300)
    plt.show()

    # Single-unit ablation - to see how crucial each hidden neuron is
    print("\n=== Single-unit ablation on XOR ===")
    for i in range(mlp.W1.shape[1]):  # number of hidden units
        acc_i = ablate_unit(mlp, i, X, y["XOR"])
        print(f" Ablate neuron {i}:  accuracy = {acc_i:.2f}")

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

    """
        2x2 grid to compare differences in PCA and W1 heat-map grids
        between clean and noisy XOR inputs
    """
    
    X_clean = X.copy()

    # Produce noisy data for analysis (10% flips)
    X_noisy = add_noise(X, prob_flip=0.1, seed=42)

    np.random.seed(42)
    # Loss, accuracy and predictions for noisy data (for XOR)
    loss_noisy, preds_noisy, acc_noisy = run(X_noisy, y, TinyMLP, "XOR", lr=0.1, epochs=20000, wd=0.1)
    print(f"Noisy data with weight-decay=0.1  loss={loss_noisy:.3f}, predictions_noisy={preds_noisy},  acc={acc_noisy:.2f}")

    # Train 4 models: (clean vs noisy) × (wd=0.0 vs wd=0.1)
    experiments = []
    for data_label, X_data in [("Clean", X_clean), ("Noisy", X_noisy)]:
        for wd in [0.0, 0.1]:
            #np.random.seed(42)
            net = TinyMLP(hidden=3)
            net.fit(
                X_data, y["XOR"],
                lr=0.1,
                epochs=20000,
                snapshot=1000,
                wd=wd
            )
            title = f"{data_label}, wd={wd:.1f}"
            experiments.append((title, net, X_data))

    # 3) Build a 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for ax, (title, net, X_data) in zip(axes.flatten(), experiments):
        # 3a) PCA on hidden activations
        H = net.hidden(X_data)                      # shape (4, hidden)
        proj = PCA(n_components=2).fit_transform(H) # shape (4, 2)

        # 3b) Scatter the 4 XOR points
        labels = y["XOR"].ravel().astype(int)
        colors = ["tab:blue" if L else "tab:red" for L in labels]
        ax.scatter(proj[:,0], proj[:,1],
                c=colors, s=80,
                edgecolor="k", zorder=2)

        # 3c) Overlay the W1 heat-map
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.imshow(net.W1,
                cmap="coolwarm",
                alpha=0.5,
                extent=[x0, x1, y0, y1],
                aspect="auto",
                origin="lower",
                zorder=1)

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Hidden-space PCA + W₁ heat-map (clean vs noisy)", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":        
    main()

    

