#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 10:13
# @Author  : jyl
# @File    : Visuallizer.py
import torchvision as tv
from visdom import Visdom
import numpy as np
import time


class Visualizer:
    def __init__(self, env='default', **kwargs):
        self.vis = Visdom(env=env, **kwargs)
        self.index = {}
        self.log = ''

    def reinit(self, env='default', **kwargs):
        #  重新配置visdom
        self.vis = Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]),
                      Y=np.array([y]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append')
        self.index[name] = x + 1

    def img(self, name, img_):
        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.images(tensor=img_.cpu(),
                        win=name,
                        opts=dict(title=name))

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name=name,
                 img_=tv.utils.make_grid(
                    tensor=input_3d.cpu()[0].unsequeeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log'):
        self.log += '[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info)
        self.vis.text(text=self.log, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

