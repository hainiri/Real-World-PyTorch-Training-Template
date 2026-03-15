import torch
from config import Config
from models.simple_nn import SimpleNN
from utils.common import predict

def main() -> None:
    """Runs inference using the trained model."""
    config = Config()
    model = SimpleNN()

    checkpoint = "checkpoints/epoch_50.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=config.device))

    test_inputs = torch.tensor(
        [[0,0],[0,1],[1,0],[1,1]], 
        dtype=torch.float32
    )

    preds = predict(model, test_inputs, config.device)
    print("Predictions:\n", preds.cpu())

if __name__ == "__main__":
    main()