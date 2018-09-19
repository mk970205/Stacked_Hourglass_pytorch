import torch.nn as nn
from model.resModule import ResModule
from model.hourglass import Hourglass
from config import CONFIG


class MainModel(nn.Module):
    def __init__(self, in_channels=3):
        super(MainModel, self).__init__()

        self.beforeHourglass = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ResModule(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResModule(128, 128),
            ResModule(128, CONFIG.nFeatures)
        )

        self.hgArray = nn.ModuleList([])
        self.llArray = nn.ModuleList([])
        for i in range(CONFIG.nStacks):
            self.hgArray.append(Hourglass())
            self.llArray.append(
                nn.ModuleList([ResModule(CONFIG.nFeatures, CONFIG.nFeatures) for _ in range(CONFIG.nModules)]))

    def forward(self, x):
        inter = self.beforeHourglass(x)
        outHeatmap = []
        for i in range(CONFIG.nStacks):
            ll = self.hgArray[i](inter)
            for j in range(CONFIG.nModules):
                ll = self.llArray[i][j](ll)
            ll = self.lin(ll, CONFIG.nFeatures, CONFIG.nFeatures)
            htmap = nn.Conv2d(CONFIG.nFeatures, CONFIG.nJoints, kernel_size=1, stride=1, padding=0)(ll)
            outHeatmap.append(htmap)

            if i < CONFIG.nStacks:
                ll = nn.Conv2d(CONFIG.nFeatures, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0)(ll)
                htmap = nn.Conv2d(CONFIG.nJoints, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0)(htmap)
                inter = inter + ll + htmap

        return outHeatmap

    def lin(self, x, in_channels, out_channels):
        tmp = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)(x)
        return nn.ReLU()(nn.BatchNorm2d(out_channels)(tmp))
