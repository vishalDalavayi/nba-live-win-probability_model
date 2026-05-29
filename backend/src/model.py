"""PyTorch neural network for live win probability."""

import torch
import torch.nn as nn


class WinProbabilityNet(nn.Module):
    """Simple feed-forward network: input -> 64 -> 32 -> sigmoid."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
