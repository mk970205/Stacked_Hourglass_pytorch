import torch.nn as nn
import torch
import os
from model.resModule import ResModule
from model.hourglass import Hourglass
from util.config import CONFIG


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
        self.linArray = nn.ModuleList([])
        self.htmapArray = nn.ModuleList([])
        self.llBarArray = nn.ModuleList([])
        self.htmapBarArray = nn.ModuleList([])

        for i in range(CONFIG.nStacks):
            self.hgArray.append(Hourglass(CONFIG.nDepth, CONFIG.nFeatures))
            self.llArray.append(
                nn.ModuleList([ResModule(CONFIG.nFeatures, CONFIG.nFeatures) for _ in range(CONFIG.nModules)]))
            self.linArray.append(self.lin(CONFIG.nFeatures, CONFIG.nFeatures))
            self.htmapArray.append(nn.Conv2d(CONFIG.nFeatures, CONFIG.nJoints, kernel_size=1, stride=1, padding=0))

        for i in range(CONFIG.nStacks - 1):
            self.llBarArray.append(nn.Conv2d(CONFIG.nFeatures, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0))
            self.htmapBarArray.append(nn.Conv2d(CONFIG.nJoints, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        inter = self.beforeHourglass(x)
        outHeatmap = []

        for i in range(CONFIG.nStacks):
            ll = self.hgArray[i](inter)
            for j in range(CONFIG.nModules):
                ll = self.llArray[i][j](ll)
            ll = self.linArray[i](ll)
            htmap = self.htmapArray[i](ll)
            outHeatmap.append(htmap)

            if i < CONFIG.nStacks - 1:
                ll_ = self.llBarArray[i](ll)
                htmap_ = self.htmapBarArray(htmap)
                inter = inter + ll_ + htmap_

        return outHeatmap

    def lin(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
