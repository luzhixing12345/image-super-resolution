
from torch import nn
import torch
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv21 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv23 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv24 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv25 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv26 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.gelu = GELU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.gelu(self.conv21(x))
        x = self.gelu(self.conv22(x))
        x = self.gelu(self.conv23(x))
        x = self.gelu(self.conv24(x))
        x = self.gelu(self.conv25(x))
        x = self.gelu(self.conv26(x))
        x = self.conv3(x)
        return x