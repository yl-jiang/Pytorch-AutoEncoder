#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 11:13
# @Author  : jyl
# @File    : configs.py
import os
import sys

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)
checkpoint = os.path.join(base_dir, 'checkpoints')


class Config:
    # W2V = 'Conv'
    # W2V = 'Basical'
    model = 'VAE'
    batch_size = 128
    img_size = 28
    max_epoch = 100
    lr = 1e-3
    weight_decay = 1e-6

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    plot_every = 100  # 每一百个batch画一次图
    print_every = 50  # 每50个batch打印一次
    save_every = 5  # 每隔5个epoch保存一次模型
    env = 'autoencoder'

    model_path = r'D:\Programs\Pytorch\AutoEncoder\checkpoints\model_best.pth.tar'
    save_path = checkpoint


opt = Config()
