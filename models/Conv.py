#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 14:52
# @Author  : jyl
# @File    : Conv.py
import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),  # Nx1x28x28 => Nx32x14X14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),  # Nx32x14X14 => Nx64x7x7

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),  # Nx64x7x7 => Nx128x4x4

            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=4, stride=2, padding=0, bias=True),
            nn.Sigmoid()  # Nx128x4x4 => Nx2x1x1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),  # Nx2x1x1 => Nx128x4x4

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),  # Nx128x4x4 => Nx64x7x7

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),  # Nx64x7x7 => Nx32x14X14

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Nx32x14X14 => Nx1x28x28
        )

    def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

