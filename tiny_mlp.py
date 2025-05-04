"""
Two‑layer MLP with a sigmoid hidden activation.
Sufficient to learn all operations (AND, OR, XOR).
"""
import torch.nn as nn
import torch

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))      # hidden representation
        return torch.sigmoid(self.fc2(h))