#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 11:20
# @Author  : jyl
# @File    : Basical.py
import torch
from torch import nn


class Basical(torch.nn.Module):
    def __init__(self):
        super(Basical, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=32, out_features=2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x




