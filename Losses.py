"""
    Name    : Losses.py 
    Author  : Yucheng Xing
"""
import torch
import numpy as np
import torch.nn as nn


def MaskedMAE(x, y, mask=None, cal_mean=True):
    mae = torch.abs(x - y) * mask if mask is not None else torch.abs(x - y)
    if cal_mean:
        mae = mae.sum(dim=(-3, -2, -1)).mean()
    else:
        mae = mae.sum()
    return mae


def MPJPE(x, y):
    mpjpe = torch.sqrt(((x - y) ** 2)).mean(dim=(-4, -3, -1))
    mpjpe_mean = mpjpe.mean()
    return mpjpe, mpjpe_mean


def PCKh(x, y, alpha=0.5):
    h = (torch.sqrt(((x[:, :, :, 1] - x[:, :, :, 0]) ** 2).sum(dim=-1)) * 2).unsqueeze(-1)
    d = torch.sqrt(((x - y) ** 2).sum(dim=-1))
    PCKh = torch.zeros_like(d)
    PCKh[d < h * alpha] = 1.0
    PCKh = PCKh.mean(dim=(-4, -3))
    PCKh_mean = PCKh.mean()
    return PCKh, PCKh_mean


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        return

    def forward(self, x, y, mask=None, cal_mean=True):
        mse = self.criterion(x, y) * mask if mask is not None else self.criterion(x, y)
        if cal_mean:
            mse = mse.sum(dim=(-4, -3, -2, -1)).mean()
        else:
            mse = mse.sum()
        return mse


class MaskedMSE1(nn.Module):
    def __init__(self):
        super(MaskedMSE1, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        return

    def forward(self, x, y, mask=None, cal_mean=True):
        mse = self.criterion(x, y) * mask if mask is not None else self.criterion(x, y)
        if cal_mean:
            mse = mse.sum(dim=(-3, -2, -1)).mean()
        else:
            mse = mse.sum()
        return mse


def GaussianNLL(x, mean, log_var, mask=None, cal_mean=True):
    std = torch.exp(0.5 * log_var)
    errors = (x - mean) / std
    constant = np.log(2 * np.pi)
    nll = 0.5 * (errors ** 2 + log_var + constant) * mask if mask \
        else 0.5 * (errors ** 2 + log_var + constant)
    if cal_mean:
        nll = nll.sum(dim=(-3, -2, -1)).mean()
    else:
        nll = nll.sum()
    return nll


def ParamCount(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()

