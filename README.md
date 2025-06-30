# Development of Neural Network: Learning AND, OR & XOR via Backpropagation and Analysis of Learned Representation

First clone our repository and navigate to the project root in console to install the requirements
(in a virtual environment) using pip -r install .\requirements.txt.

To run the program, run app.py. Results of the analysis will be shown in the terminal, and the
plots used in the analysis will appear.

A minimal, from-scratch NumPy implementation of a single-layer Perceptron and a two-layer MLP (“TinyMLP”) to learn and analyze the Boolean functions AND, OR, and XOR via backpropagation.  Alongside training and decision-boundary visualization, this project includes a suite of representation-analysis tools (PCA, animated trajectories, unit ablation, weight & activation heat-maps) and a noise/regularization robustness study.

---

## 🚀 Features

- **Perceptron & TinyMLP**  
  - Single-layer perceptron learns linearly separable gates (AND, OR)  
  - Two-layer MLP solves XOR with configurable hidden-unit size, learning rate, epochs, L2 weight-decay  

- **Decision Boundary Plots**  
  - Mesh-grid visualizations for Perceptron and TinyMLP on AND, OR, and XOR  

- **Representation Analysis**  
  - **Hidden-unit activations** table 
  - **PCA** of final hidden representations, with explained-variance annotations  
  - **Animated PCA trajectories** showing cluster separation over epochs  
  - **Single-unit ablation** to quantify each neuron's causal role  
  - **Weight & activation heat-maps** reveal logical sub-functions  

---
