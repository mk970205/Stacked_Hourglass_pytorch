import torch.nn as nn


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resSeq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels / 2, kernel_size=1),
            nn.BatchNorm2d(out_channels / 2),
            nn.ReLU(),
            nn.Conv2d(out_channels / 2, out_channels / 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels / 2),
            nn.ReLU(),
            nn.Conv2d(out_channels / 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            skip = self.conv_skip(x)
        else:
            skip = x

        return skip + self.resSeq(x)
