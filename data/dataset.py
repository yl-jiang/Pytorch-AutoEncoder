#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 9:36
# @Author  : jyl
# @File    : data.py
from config import opt
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader


def dataloader(opt, is_train=True):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.mean, std=opt.std)
    ])

    mnist_dataset = tv.datasets.MNIST('./data', train=is_train, transform=transform, download=True)
    if is_train:
        return DataLoader(mnist_dataset, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
    else:
        return DataLoader(mnist_dataset, batch_size=opt.batch_size, pin_memory=True, shuffle=False)








