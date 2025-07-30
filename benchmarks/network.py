from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock()
            for _ in range(10)
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                16 * 20 * 20,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                4,
            ),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=91,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                # Include bias because we're not using batch norm.
                bias=True,
            ),
            nn.Flatten(),
        )

    def forward(self, occupancies):
        # Add an all-ones channel to the input for edge detection.
        x = occupancies
        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return (
            self.value_head(x),
            self.policy_head(x),
        )
