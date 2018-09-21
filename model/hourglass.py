import torch.nn as nn
from model.resModule import ResModule
from util.config import CONFIG


class Hourglass(nn.Module):
    def __init__(self, hg_depth, nFeatures):
        super(Hourglass, self).__init__()
        self.hg_depth = hg_depth
        self.nFeatures = nFeatures
        self.res1 = nn.ModuleList([ResModule(nFeatures, nFeatures) for i in range(CONFIG.nModules)])
        self.res2 = nn.ModuleList([ResModule(nFeatures, nFeatures) for i in range(CONFIG.nModules)])
        self.res3 = nn.ModuleList([ResModule(nFeatures, nFeatures) for i in range(CONFIG.nModules)])
        self.subHourglass = None
        self.resWaist = None
        if self.hg_depth > 1:
            self.subHourglass = Hourglass(self.hg_depth - 1, nFeatures)
        else:
            self.resWaist = nn.ModuleList([ResModule(nFeatures, nFeatures) for i in range(CONFIG.nModules)])

    def forward(self, x):
        up = self.res1(x)
        low1 = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        low1 = self.res2(low1)

        if self.hg_depth > 1:
            low2 = self.subHourglass(low1)
        else:
            low2 = self.resWaist(low1)

        low3 = self.res3(low2)

        low = nn.UpsamplingNearest2d(scale_factor=2)(low3)

        return up + low
