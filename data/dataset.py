from torch.utils.data import Dataset
import torch
from typing import Tuple

class XORdataset(Dataset):
    """Simple XOR dataset for neural network training."""

    def __init__(self) -> None:
        self.x = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], 
            dtype=torch.float32
        )
        self.y = torch.tensor(
            [[0], [1], [1], [0]], 
            dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
