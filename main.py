#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 10:58
# @Author  : jyl
# @File    : autoencoder.py
import os
import torch
import models
import shutil
from tqdm import tqdm
from config import opt
from torchnet import meter
from utils import Visualizer
from data import dataloader
from torch.nn import functional as F


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, opt.model)().to(device)
    train_loader = dataloader(opt, is_train=True)
    test_loader = dataloader(opt, is_train=False)
    vis = Visualizer(opt.env)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if os.path.exists(opt.model_path):
        try:
            checkpoint = torch.load(opt.model_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoints '{}' (epoch {})".format(opt.model_path, checkpoint['epoch']))
        except RuntimeError:
            start_epoch = 1
            print("=> no checkpoints found at '{}'".format(opt.model_path))
    else:
        start_epoch = 1
        print("=> no checkpoints found at '{}'".format(opt.model_path))

    if opt.model == 'Basical':
        criterion = torch.nn.MSELoss().to(device)
        for epoch in range(start_epoch, opt.max_epoch + 1):
            train_basical(model, criterion, optimizer, train_loader, epoch, vis)
    elif opt.model == 'Conv':
        criterion = torch.nn.MSELoss().to(device)
        for epoch in range(start_epoch, opt.max_epoch + 1):
            train_conv(model, criterion, optimizer, train_loader, epoch, vis)
    elif opt.model == 'VAE':
        for epoch in range(start_epoch, opt.max_epoch + 1):
            train_vae(model, vae_loss_function, optimizer, train_loader, start_epoch, vis)
    else:
        print("No module nameed '{}', module's name must be '{} | {} | {}' please check again!".format(
            opt.model, 'Basical', 'Conv', 'VAE'))


def train_basical(model, criterion, optimizer, dataloader, epoch, vis):
    model.train()
    min_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_meter = meter.AverageValueMeter()
    for i, (input, target) in enumerate(tqdm(dataloader), 1):
        img = torch.Tensor(input).view(-1, opt.img_size * opt.img_size).to(device)
        output = model(img)
        loss = criterion(output, torch.Tensor(input).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        if i % opt.plot_every == 0:
            vis.plot(name='loss', y=loss_meter.value()[0])
            vis.images(output.data.cpu().numpy()[:32] * 0.5 + 0.5, win='output_imgs')

    if epoch % opt.save_every == 0:
        is_best = loss < min_loss
        state = dict(
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            min_loss=min_loss,
            optimizer=optimizer.state_dict()
        )
        file_path = opt.save_path
        save_checkpoint(state, is_best, file_path, filename='checkpoint_basical.pth.tar')


def train_conv(model, criterion, optimizer, dataloader, epoch, vis):
    model.train()
    min_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_meter = meter.AverageValueMeter()
    for i, (input, target) in enumerate(tqdm(dataloader), 1):
        img = torch.Tensor(input).to(device)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        if i % opt.plot_every == 0:
            vis.plot(name='loss', y=loss_meter.value()[0])
            vis.images(output.data.cpu().numpy()[:32] * 0.5 + 0.5, win='output_imgs')

    if epoch % opt.save_every == 0:
        is_best = loss < min_loss
        state = dict(
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            min_loss=min_loss,
            optimizer=optimizer.state_dict()
        )
        file_path = opt.save_path
        save_checkpoint(state, is_best, file_path, filename='checkpoint_conv.pth.tar')


def train_vae(model, criterion, optimizer, dataloader, epoch, vis):
    model.train()
    min_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_meter = meter.AverageValueMeter()
    for i, (input_, _) in enumerate(tqdm(dataloader), 1):
        img = torch.Tensor(input_).view(-1, opt.img_size * opt.img_size).to(device)
        output, m, sigma = model(img)
        loss = criterion(output, img, m, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        if i % opt.plot_every == 0:
            vis.plot(name='loss', y=loss_meter.value()[0])
            vis.images(output.view(-1, 1, opt.img_size, opt.img_size).cpu().detach().numpy()[:32] * 0.5 + 0.5, win='output_imgs')

    if epoch % opt.save_every == 0:
        is_best = loss < min_loss
        state = dict(
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            min_loss=min_loss,
            optimizer=optimizer.state_dict()
        )
        file_path = opt.save_path
        save_checkpoint(state, is_best, file_path, filename='checkpoint_vae.pth.tar')


def save_checkpoint(state, is_best, file_path, filename='checkpoints.pth.tar'):
    file = os.path.join(file_path, filename)
    torch.save(state, file)
    if is_best:
        best_file = os.path.join(file_path, 'model_best.pth.tar')
        shutil.copyfile(file, best_file)


def vae_loss_function(recon_x, x, m, sigma):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # MSE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + sigma - m.pow(2) - sigma.exp())
    return BCE + KLD
    # return MSE + KLD


def test(model, opt, test_loader, criterion, vis):
    model.eval()
    loss_meter = meter.AverageValueMeter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, (input, target) in enumerate(tqdm(test_loader), 1):
        if opt.model == 'Conv':
            img = torch.Tensor(input).to(device)
        else:
            img = torch.Tensor(input).view(-1, opt.img_size*opt.img_size).to(device)

        if opt.model == 'VAE':
            output, m, sigma = model(img)
            loss = criterion(output, img, m, sigma)
        else:
            output = model(img)
            loss = criterion(output, img)

        loss_meter.add(loss.item())
        if i % opt.plot_every == 0:
            vis.plot(name='loss_test', y=loss_meter.value()[0])
            vis.images(output.data.cpu().numpy()[:32] * 0.5 + 0.5, win='output_imgs')


if __name__ == '__main__':
    main(opt)





