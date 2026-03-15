import torch
from torch.utils.data import DataLoader
from config import Config
from data.dataset import XORdataset
from models.simple_nn import SimpleNN
from engine.trainer import Trainer

def main() -> None:
    """Main training entry point."""
    config = Config()

    dataset = XORdataset()
    train_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )

    model = SimpleNN()
    trainer = Trainer(model, train_loader, config)
    
    trainer.fit()

if __name__ == "__main__":
    main()