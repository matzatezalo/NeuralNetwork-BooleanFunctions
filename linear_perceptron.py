"""
Single‑layer perceptron.
Learns any linearly separable Boolean gate (AND, OR).
"""
import torch.nn as nn
import torch

class Perceptron(nn.Module):
    def __init__(self, in_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid gives probabilities in [0,1]
        return torch.sigmoid(self.fc(x))