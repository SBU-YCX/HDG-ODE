"""
    Name    : Models.py 
    Author  : Yucheng Xing
"""
import torch
import numpy as np
import torch.nn as nn
from Losses import MaskedMSE
from Modules import DGNN, DGCGRUCell


class HDGODE(nn.Module):
    ''' Model : HDG-ODE '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t, choice='F'):
        super(HDGODE, self).__init__()
        self.GRU_1 = nn.ModuleList()
        for b in range(num_rnn[0]):
            self.GRU_1.append(DGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint[1] * 3, choice=choice))
        self.GRU_2 = nn.ModuleList()
        for b in range(num_rnn[1]):
            self.GRU_2.append(DGCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn, num_joint[2], choice=choice))
        self.GRU_3 = nn.ModuleList()
        for b in range(num_rnn[2]):
            self.GRU_3.append(DGCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn, num_joint[3], choice=choice))
        self.ODE_1 = DGNN(dim_rnn, dim_rnn, dim_hidden, num_hidden[0], num_joint[1] * 3)
        self.ODE_2 = DGNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[1], num_joint[2])
        self.ODE_3 = DGNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[2], num_joint[3])
        self.output = nn.Linear(dim_rnn, dim_out)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, 1, dim_rnn)))
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        h = self.h0.repeat(xs.size(0), xs.size(2), xs.size(3), 1)
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        dim_out = h.size(-1)
        gs = data_batch['group']
        g1 = gs[0][0].cuda() - 1
        g2 = gs[1][0].cuda() - 1
        g3 = gs[2][0].cuda() - 1
        s_adjs = data_batch['s_adj_h']
        s_adj1 = s_adjs[0].cuda()
        s_adj2 = s_adjs[1].repeat(1, 1, 1).cuda().repeat(3, 1, 1).cuda()
        s_adj3 = s_adjs[2].repeat(1, 1, 1, 1).cuda().repeat(3, 1, 1, 1).cuda() 
        d_adjs = data_batch['d_adj_h']
        d_adj1 = d_adjs[0].cuda()
        d_adj2 = d_adjs[1].reshape(-1, seq_len, g2.shape[0], g2.shape[0]).cuda()
        d_adj3 = d_adjs[2].reshape(-1, seq_len, g3.shape[0], g3.shape[1], g3.shape[1]).cuda()
        k = 2
        hs = []
        for n, t_n in enumerate(ts):
            m = ms[:, n]
            x = xs[:, n]
            h1 = h
            h1_1 = h1[:, :, g1]
            x_1 = x[:, :, g1]
            h1_2 = h1[:, :, g2]
            x_2 = x[:, :, g2]
            h1_3 = [h1[:, :, g3[g]] for g in range(g3.shape[0])]
            x_3 = [x[:, :, g3[g]] for g in range(g3.shape[0])]
            ## ODE
            # Lv1
            dh1_1 = self.ODE_1(h1_1.reshape(-1, num_p, dim_out), s_adj1).reshape(batch_size, num_p, -1, dim_out)
            h1_1 = h1_1 + dh1_1 * (2 * k * delta_t)
            # Lv2
            for i in range(1, k + 1):
                dh1_2 = self.ODE_2(torch.cat([h1_1.repeat(1, 1, h1_2.shape[2], 1), h1_2], -1).reshape(batch_size * num_p, -1, dim_out * 2), s_adj2).reshape(batch_size, num_p, -1, dim_out)
                h1_2 = h1_2 + dh1_2 * (k * delta_t) + h1_1
                # Lv3
                for j in range(1, k + 1):
                    for gi in range(len(h1_3)):
                        dh1_3 = self.ODE_3(torch.cat([h1_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, h1_3[gi].shape[2], 1), h1_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out * 2), s_adj3[:, gi]).reshape(batch_size, num_p, -1, dim_out)
                        h1_3[gi] = h1_3[gi] + dh1_3 * delta_t + h1_2[:, gi + 1].unsqueeze(1).repeat(1, h1_3[gi].shape[1], 1)
            h1[:, :, g1] = h1_1
            h1[:, :, g2] = h1_2
            for gi in range(g3.shape[0]):
                h1[:, :, g3[gi]] = h1_3[gi]
            hs.append(h1)
            ## GRU
            h2 = h1
            # Lv1
            h2_1 = x_1.reshape(-1, num_p, dim_in)
            h1_1 = h1_1.reshape(-1, num_p, dim_out)
            for block in self.GRU_1:
                h2_1 = block(h2_1, h1_1, d_adj1[:, n])
            h2_1 = h2_1.reshape(batch_size, num_p, -1, dim_out)
            # Lv2
            h2_2 = torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
            h1_2 = h1_2.reshape(batch_size * num_p, -1, dim_out)
            for block in self.GRU_2:
                h2_2 = block(h2_2, h1_2, d_adj2[:, n])
            h2_2 = h2_2.reshape(batch_size, num_p, -1, dim_out)
            # Lv3
            h2_3s = []
            for gi in range(len(h1_3)):
                h2_3 = torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
                for block in self.GRU_3:
                    h2_3 = block(h2_3, h1_3[gi].reshape(batch_size * num_p, -1, dim_out), d_adj3[:, n, gi])
                h2_3s.append(h2_3.reshape(batch_size, num_p, -1, dim_out))
            h2[:, :, g1] = h2_1
            h2[:, :, g2] = h2_2
            for gi in range(g3.shape[0]):
                h2[:, :, g3[gi]] = h2_3s[gi]
            ## Update
            h = h1 * (1. - m) + h2 * m
        hs = torch.stack(hs, 1)
        ys = self.output(hs)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss
