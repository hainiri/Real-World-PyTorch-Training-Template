import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
from config import Config

class Trainer:
    """Handles model training and checkpointing."""

    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        config: Config
    ) -> None:

        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.config = config

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.lr
        )

    def train_one_epoch(self, epoch: int) -> float:
        """
        Trains the model for a single epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.config.device)
            batch_y = batch_y.to(self.config.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}/{self.config.num_epochs} | Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves model weights.
        """
        path = f"{self.config.checkpoint_dir}/epoch_{epoch+1}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved at {path}")

    def fit(self) -> None:
        """Runs full training."""
        for epoch in range(self.config.num_epochs):
            self.train_one_epoch(epoch)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
