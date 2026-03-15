import torch
from torch import Tensor
from torch.nn import Module
from typing import Union

def predict(model: Module, x: Tensor, device: str) -> Tensor:
    """
    Runs inference on a trained model.

    Args:
        model: PyTorch model.
        x: Input tensor.
        device: Device type ("cpu" or "cuda").

    Returns:
        Tensor: Model predictions.
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        return model(x)
