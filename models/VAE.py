#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 11:20
# @Author  : jyl
# @File    : VAE.py
import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.linear1 = nn.Linear(in_features=28*28, out_features=400)
        self.linear2_m = nn.Linear(in_features=400, out_features=20)
        self.linear2_sigma = nn.Linear(in_features=400, out_features=20)
        self.linear3 = nn.Linear(in_features=20, out_features=400)
        self.linear4 = nn.Linear(in_features=400, out_features=28*28)

    def encoder(self, x):
        h1 = F.relu(self.linear1(x))
        return self.linear2_m(h1), self.linear2_sigma(h1)

    def reparameterize(self, m, sigma):
        if self.training:
            std = torch.exp(0.5 * sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(m)
        else:
            return m

    def decoder(self, z):
        h3 = F.relu(self.linear3(z))
        return F.sigmoid(self.linear4(h3))

    def forward(self, x):
        m, sigma = self.encoder(x)
        z = self.reparameterize(m, sigma)
        return self.decoder(z), m, sigma




