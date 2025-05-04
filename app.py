import numpy as np
import torch 
import matplotlib.pyplot as plt

torch.manual_seed(0);
np.random.seed(42);

def main():
    X = torch.tensor([[0.,0.],
                    [0.,1.],
                    [1.,0.],
                    [1.,1.]])              # shape (4,2)

    # Target is 4x1 so we can feed it to BCELoss
    y_and = torch.tensor([[0.],[0.],[0.],[1.]], dtype=torch.float32)
    y_or  = torch.tensor([[0.],[1.],[1.],[1.]], dtype=torch.float32)
    y_xor = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

