from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    """Configuration settings for training and inference."""
    lr: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
