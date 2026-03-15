import torch
import torch.nn as nn
from torch import Tensor

class SimpleNN(nn.Module):
    """Simple feedforward neural network for XOR classification."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        return self.net(x)
