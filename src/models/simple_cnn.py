import torch
from torch import nn
from torchvision.models import resnet18
from .residual_block import ResidualBlock

class SimpleCNN(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.resnet = resnet18()
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.squeeze(1)

        return x