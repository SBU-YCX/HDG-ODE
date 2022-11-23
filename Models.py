"""
    Name    : Models.py 
    Author  : Yucheng Xing
"""
import torch
import numpy as np
import torch.nn as nn
from Losses import MaskedMSE
from Modules import DNN, GNN, DGNN, GRUCell, GCGRUCell, DGCGRUCell, STConvBlock, SemGNN, SemGCGRUCell


class STGCN(nn.Module):
    ''' Model : ST-GCN '''
    def __init__(self, dim_in, dim_out, dim_h, kernel_size, num_block, num_joint, activation=nn.ReLU):
        super(STGCN, self).__init__()
        self.net = nn.ModuleList()
        for b in range(num_block):
            self.net.append(STConvBlock(dim_h if b else dim_in, dim_h, dim_h, kernel_size, dim_h, num_joint, activation))
        self.output = nn.Linear(dim_h, dim_out)
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch):
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        assert not torch.isnan(xs).any()
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        s_adj = data_batch['s_adj_f'].cuda()
        for block in self.net:
            xs = block(xs, s_adj)
        y = self.output(xs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return y

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class GCGRU(nn.Module):
    ''' Model : GRU '''
    def __init__(self, dim_in, dim_out, dim_rnn, num_rnn, delta_t, choice='F'):
        super(GCGRU, self).__init__()
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(GCGRUCell(dim_rnn if b else dim_in, dim_rnn, choice=choice))
        self.output = nn.Linear(dim_rnn, dim_out)
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        s_adj = data_batch['s_adj_f'].cuda()
        for block in self.GRU:
            hs = []
            h = block.initiation().repeat(xs.size(0), xs.size(2), 1)
            for n in range(seq_len):
                h = block(xs[:, n], h, s_adj)
                hs.append(h)
            xs = torch.stack(hs, 1)
        ys = self.output(xs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class SemGCGRU(nn.Module):
    ''' Model : SemGRU '''
    def __init__(self, dim_in, dim_out, dim_rnn, num_rnn, num_joint, delta_t, choice='F'):
        super(SemGCGRU, self).__init__()
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(SemGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint, choice=choice))
            #self.GRU.append(DGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint, choice=choice))
        self.output = nn.Linear(dim_rnn, dim_out)
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        s_adj = data_batch['s_adj_f'].cuda()
        for block in self.GRU:
            hs = []
            h = block.initiation().repeat(xs.size(0), xs.size(2), 1)
            for n in range(seq_len):
                h = block(xs[:, n], h, s_adj)
                hs.append(h)
            xs = torch.stack(hs, 1)
        ys = self.output(xs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class DGCGRU(nn.Module):
    ''' Model : GRU (Dynamic Graph Conv) '''
    def __init__(self, dim_in, dim_out, dim_rnn, num_rnn, num_joint, delta_t, choice='F'):
        super(DGCGRU, self).__init__()
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(DGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint, choice='F'))
        self.output = nn.Linear(dim_rnn, dim_out)
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        d_adj = data_batch['d_adj_f'].cuda()
        for block in self.GRU:
            hs = []
            h = block.initiation().repeat(xs.size(0), xs.size(2), 1)
            for n in range(seq_len):
                h = block(xs[:, n], h, d_adj[:, n])
                hs.append(h)
            xs = torch.stack(hs, 1)
        ys = self.output(xs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class ODERNN(nn.Module):
    ''' Model : ODE-RNN '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t):
        super(ODERNN, self).__init__()
        #self.GRU = GRUCell(dim_in, dim_rnn)
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(GRUCell(dim_rnn if b else dim_in, dim_rnn))
        self.ODE = DNN(dim_rnn, dim_rnn, dim_hidden, num_hidden, num_joint)
        self.output = nn.Linear(dim_rnn, dim_out)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_rnn)))
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()#MaskedMSE1()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        ms = ms.reshape(batch_size, seq_len, -1, 1)
        h = self.h0.repeat(xs.size(0), xs.size(2), 1)
        s_adj = data_batch['s_adj_f'].cuda()
        d_adj = data_batch['d_adj_f'].cuda()
        k = int(4)#(ts[1] - ts[0]) / delta_t)
        hs = []
        for n, t_n in enumerate(ts):
            m = ms[:, n]
            x = xs[:, n]
            h1 = h
            for i in range(1, k + 1):
                dh = self.ODE(h1)
                h1 = h1 + dh * delta_t
            hs.append(h1)
            for block in self.GRU:
                h2 = block(x, h1)
                x = h2
            #h2 = self.GRU(x, h1)
            h = h1 * (1. - m) + h2 * m 
        hs = torch.stack(hs, 1)
        ys = self.output(hs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class GRUODE(nn.Module):
    ''' Model : GRU-ODE '''
    def __init__(self):
        super(ODERNN, self).__init__()
        return

class GCODERNN(nn.Module):
    ''' Model : ODE-RNN (Graph Conv) '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t, choice='F'):
        super(GCODERNN, self).__init__()
        #self.GRU = GCGRUCell(dim_in, dim_rnn, choice=choice)
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(GCGRUCell(dim_rnn if b else dim_in, dim_rnn, choice=choice))
        self.ODE = GNN(dim_rnn, dim_rnn, dim_hidden, num_hidden, num_joint, choice=choice)
        self.output = nn.Linear(dim_rnn, dim_out)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_rnn)))
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        ms = ms.reshape(batch_size, seq_len, -1, 1)
        h = self.h0.repeat(xs.size(0), xs.size(2), 1)
        dim_out = h.size(-1)
        adj = data_batch['s_adj_f'].cuda()
        d_adj = data_batch['d_adj_f'].cuda()
        k = 4#(ts[1] - ts[0]) / delta_t
        hs = []
        for n, t_n in enumerate(ts):
            m = ms[:, n]
            x = xs[:, n]
            h1 = h
            for i in range(1, k + 1):
                dh = self.ODE(h1, adj)
                h1 = h1 + dh * delta_t
            hs.append(h1)
            for block in self.GRU:
                h2 = block(x, h1, adj)
                x = h2
            #h2 = self.GRU(x, h1, adj)
            h = h1 * (1. - m) + h2 * m
        hs = torch.stack(hs, 1)
        ys = self.output(hs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class SemGCODERNN(nn.Module):
    ''' Model : ODE-RNN (Sem Graph Conv) '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t, choice='F'):
        super(SemGCODERNN, self).__init__()
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(SemGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint, choice=choice))
        #self.GRU = SemGCGRUCell(dim_in, dim_rnn, num_joint, choice=choice)
        self.ODE = SemGNN(dim_rnn, dim_rnn, dim_hidden, num_hidden, num_joint, choice=choice)
        #self.GRU = DGCGRUCell(dim_in, dim_rnn, num_joint, choice=choice)
        #self.ODE = DGNN(dim_rnn, dim_rnn, dim_hidden, num_hidden, num_joint, choice=choice)
        self.output = nn.Linear(dim_rnn, dim_out)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_rnn)))
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        ms = ms.reshape(batch_size, seq_len, -1, 1)
        h = self.h0.repeat(xs.size(0), xs.size(2), 1)
        dim_out = h.size(-1)
        adj = data_batch['s_adj_f'].cuda()
        d_adj = data_batch['d_adj_f'].cuda()
        k = 4#(ts[1] - ts[0]) / delta_t
        hs = []
        for n, t_n in enumerate(ts):
            m = ms[:, n]
            x = xs[:, n]
            h1 = h
            for i in range(1, k + 1):
                dh = self.ODE(h1, adj)
                h1 = h1 + dh * delta_t
            hs.append(h1)
            for block in self.GRU:
                h2 = block(x, h1, adj)
                x = h2
            #h2 = self.GRU(x, h1, adj)
            h = h1 * (1. - m) + h2 * m
        hs = torch.stack(hs, 1)
        ys = self.output(hs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class GCGRUODE(nn.Module):
    ''' Model : GRU-ODE (Graph Conv) '''
    def __init__(self):
        super(GCGRUODE, self).__init__()
        return

class DGCODERNN(nn.Module):
    ''' Model : ODE-RNN (Dynamic Graph Conv) '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t, choice='F'):
        super(DGCODERNN, self).__init__()
        #self.GRU = SemGCGRUCell(dim_in, dim_rnn, num_joint, choice=choice)
        self.GRU = nn.ModuleList()
        for b in range(num_rnn):
            self.GRU.append(DGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint, choice=choice))
        self.ODE = DGNN(dim_rnn, dim_rnn, dim_hidden, num_hidden, num_joint, choice=choice)
        self.output = nn.Linear(dim_rnn, dim_out)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_rnn)))
        self.delta_t = delta_t
        self.loss_func = MaskedMSE()
        return

    def forward(self, data_batch, delta_t=None):
        if delta_t is None:
            delta_t = self.delta_t
        ts = data_batch['t'][0].cuda()
        ms = data_batch['mask'].cuda()
        xs = data_batch['x2d'].cuda() * ms
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        xs = xs.reshape(batch_size, seq_len, -1, dim_in)
        ms = ms.reshape(batch_size, seq_len, -1, 1)
        h = self.h0.repeat(xs.size(0), xs.size(2), 1)
        dim_out = h.size(-1)
        s_adj = data_batch['s_adj_f'].cuda()
        d_adj = data_batch['d_adj_f'].cuda()
        k = 4#(ts[1] - ts[0]) / delta_t
        hs = []
        for n, t_n in enumerate(ts):
            m = ms[:, n]
            x = xs[:, n]
            a = d_adj[:, n]
            h1 = h
            for i in range(1, k + 1):
                dh = self.ODE(h1, s_adj)
                h1 = h1 + dh * delta_t
            hs.append(h1)
            for block in self.GRU:
                h2 = block(x, h1, a)
                x = h2
            #h2 = self.GRU(x, h1, a)
            h = h1 * (1. - m) + h2 * m
        hs = torch.stack(hs, 1)
        ys = self.output(hs).reshape(batch_size, seq_len, num_p, num_j, -1)
        return ys

    def get_loss(self, data_batch):
        x = self.forward(data_batch)[:, 1:]
        y = data_batch['x3d'][:, 1:].cuda()
        m = data_batch['mask'][:, 1:].cuda()
        loss = self.loss_func(x, y)
        return loss


class DGCGRUODE(nn.Module):
    ''' Model : GRU-ODE (Dynamic Graph Conv) '''
    def __init__(self):
        super(DGCGRUODE, self).__init__()
        return


class HNODE(nn.Module):
    ''' Model : H-ODE '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t):
        super(HNODE, self).__init__()
        self.GRU_1 = nn.ModuleList()
        for b in range(num_rnn[0]):
            self.GRU_1.append(GRUCell(dim_rnn if b else dim_in, dim_rnn))
        self.GRU_2 = nn.ModuleList()
        for b in range(num_rnn[1]):
            self.GRU_2.append(GRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn))
        self.GRU_3 = nn.ModuleList()
        for b in range(num_rnn[2]):
            self.GRU_3.append(GRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn))
        #self.GRU_1 = GRUCell(dim_in, dim_rnn)
        #self.GRU_2 = GRUCell(dim_in + dim_rnn, dim_rnn)
        #self.GRU_3 = GRUCell(dim_in + dim_rnn, dim_rnn)
        self.ODE_1 = DNN(dim_rnn, dim_rnn, dim_hidden, num_hidden[0], num_joint[1])
        self.ODE_2 = DNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[1], num_joint[2])
        self.ODE_3 = DNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[2], num_joint[3])
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
        batch_size, seq_len, num_p, num_j, dim_in = xs.size()
        h = self.h0.repeat(xs.size(0), xs.size(2), xs.size(3), 1)
        dim_out = h.size(-1)
        gs = data_batch['group']
        g1 = gs[0][0].cuda() - 1
        g2 = gs[1][0].cuda() - 1
        g3 = gs[2][0].cuda() - 1
        s_adjs = data_batch['s_adj_h']
        d_adjs = data_batch['d_adj_h']
        k = 2#(ts[1] - ts[0]) / delta_t
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
            dh1_1 = self.ODE_1(h1_1.reshape(batch_size, -1, dim_out)).reshape(batch_size, num_p, -1, dim_out)
            h1_1 = h1_1 + dh1_1 * (2 * k * delta_t)
            # Lv2
            for i in range(1, k + 1):
                dh1_2 = self.ODE_2(torch.cat([h1_1.repeat(1, 1, h1_2.shape[2], 1), h1_2], -1).reshape(batch_size, -1, dim_out * 2)).reshape(batch_size, num_p, -1, dim_out)
                h1_2 = h1_2 + dh1_2 * (k * delta_t)# + h1_1
                # Lv3
                for j in range(1, k + 1):
                    for gi in range(len(h1_3)):
                        dh1_3 = self.ODE_3(torch.cat([h1_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, h1_3[gi].shape[2], 1), h1_3[gi]], -1).reshape(batch_size, -1, dim_out * 2)).reshape(batch_size, num_p, -1, dim_out)
                        h1_3[gi] = h1_3[gi] + dh1_3 * delta_t# + h1_2[:, gi + 1].unsqueeze(1).repeat(1, h1_3[gi].shape[1], 1)
            h1[:, :, g1] = h1_1
            h1[:, :, g2] = h1_2
            for gi in range(g3.shape[0]):
                h1[:, :, g3[gi]] = h1_3[gi]
            hs.append(h1)
            ## GRU
            h2 = h1
            # Lv1
            h2_1 = x_1.reshape(batch_size, -1, dim_in)
            h1_1 = h1_1.reshape(batch_size, -1, dim_out)
            for block in self.GRU_1:
                h2_1 = block(h2_1, h1_1)
            h2_1 = h2_1.reshape(batch_size, num_p, -1, dim_out)
            #h2_1 = self.GRU_1(x_1.reshape(batch_size, -1, dim_in), h1_1.reshape(batch_size, -1, dim_out)).reshape(batch_size, num_p, -1, dim_out)
            # Lv2
            h2_2 = torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size, -1, dim_in + dim_out)
            h1_2 = h1_2.reshape(batch_size, -1, dim_out)
            for block in self.GRU_2:
                h2_2 = block(h2_2, h1_2)
            h2_2 = h2_2.reshape(batch_size, num_p, -1, dim_out)
            #h2_2 = self.GRU_2(torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size, -1, dim_in + dim_out), h1_2.reshape(batch_size, -1, dim_out)).reshape(batch_size, num_p, -1, dim_out)
            # Lv3
            h2_3s = []
            for gi in range(len(h1_3)):
                h2_3 = torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size, -1, dim_in + dim_out)
                for block in self.GRU_3:
                    h2_3 = block(h2_3, h1_3[gi].reshape(batch_size, -1, dim_out))
                h2_3s.append(h2_3.reshape(batch_size, num_p, -1, dim_out))
                #h2_3s.append(self.GRU_3(torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size, -1, dim_in + dim_out), h1_3[gi].reshape(batch_size, -1, dim_out)).reshape(batch_size, num_p, -1, dim_out))
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


class HGCODE(nn.Module):
    ''' Model : HG-ODE '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t):
        super(HGCODE, self).__init__()
        self.GRU_1 = nn.ModuleList()
        for b in range(num_rnn[0]):
            self.GRU_1.append(GCGRUCell(dim_rnn if b else dim_in, dim_rnn))
        self.GRU_2 = nn.ModuleList()
        for b in range(num_rnn[1]):
            self.GRU_2.append(GCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn))
        self.GRU_3 = nn.ModuleList()
        for b in range(num_rnn[2]):
            self.GRU_3.append(GCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn))
        #self.GRU_1 = GCGRUCell(dim_in, dim_rnn)
        #self.GRU_2 = GCGRUCell(dim_in + dim_rnn, dim_rnn)
        #self.GRU_3 = GCGRUCell(dim_in + dim_rnn, dim_rnn)
        self.ODE_1 = GNN(dim_rnn, dim_rnn, dim_hidden, num_hidden[0], num_joint[1] * 3)
        self.ODE_2 = GNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[1], num_joint[2])
        self.ODE_3 = GNN(dim_rnn * 2, dim_rnn, dim_hidden, num_hidden[2], num_joint[3])
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
        adjs = data_batch['s_adj_h']
        adj1 = adjs[0].cuda()
        adj2 = adjs[1].repeat(3, 1, 1).cuda()
        adj3 = adjs[2].repeat(3, 1, 1, 1).cuda()
        d_adjs = data_batch['d_adj_h']
        k = 2#(ts[1] - ts[0]) / delta_t
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
            dh1_1 = self.ODE_1(h1_1.reshape(-1, num_p, dim_out), adj1).reshape(batch_size, num_p, -1, dim_out)
            h1_1 = h1_1 + dh1_1 * (2 * k * delta_t)
            # Lv2
            for i in range(1, k + 1):
                dh1_2 = self.ODE_2(torch.cat([h1_1.repeat(1, 1, h1_2.shape[2], 1), h1_2], -1).reshape(batch_size * num_p, -1, dim_out * 2), adj2).reshape(batch_size, num_p, -1, dim_out)
                h1_2 = h1_2 + dh1_2 * (k * delta_t)# + h1_1
                # Lv3
                for j in range(1, k + 1):
                    for gi in range(len(h1_3)):
                        dh1_3 = self.ODE_3(torch.cat([h1_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, h1_3[gi].shape[2], 1), h1_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out * 2), adj3[:, gi]).reshape(batch_size, num_p, -1, dim_out)
                        h1_3[gi] = h1_3[gi] + dh1_3 * delta_t# + h1_2[:, gi + 1].unsqueeze(1).repeat(1, h1_3[gi].shape[1], 1)
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
                h2_1 = block(h2_1, h1_1, adj1)
            h2_1 = h2_1.reshape(batch_size, num_p, -1, dim_out)
            #h2_1 = self.GRU_1(x_1.reshape(-1, num_p, dim_in), h1_1.reshape(-1, num_p, dim_out), adj1).reshape(batch_size, num_p, -1, dim_out)
            # Lv2
            h2_2 = torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
            h1_2 = h1_2.reshape(batch_size * num_p, -1, dim_out)
            for block in self.GRU_2:
                h2_2 = block(h2_2, h1_2, adj2)
            h2_2 = h2_2.reshape(batch_size, num_p, -1, dim_out)
            #h2_2 = self.GRU_2(torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size * num_p, -1, dim_out + dim_in), h1_2.reshape(batch_size * num_p, -1, dim_out), adj2).reshape(batch_size, num_p, -1, dim_out)
            # Lv3
            h2_3s = []
            for gi in range(len(h1_3)):
                h2_3 = torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
                for block in self.GRU_3:
                    h2_3 = block(h2_3, h1_3[gi].reshape(batch_size * num_p, -1, dim_out), adj3[:, gi])
                h2_3s.append(h2_3.reshape(batch_size, num_p, -1, dim_out))
                #h2_3.append(self.GRU_3(torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out + dim_in), h1_3[gi].reshape(batch_size * num_p, -1, dim_out), adj3[:, gi]).reshape(batch_size, num_p, -1, dim_out))
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


class HDGODE(nn.Module):
    ''' Model : HDG-ODE '''
    def __init__(self, dim_in, dim_out, dim_rnn, dim_hidden, num_hidden, num_rnn, num_joint, delta_t, choice='F'):
        super(HDGODE, self).__init__()
        self.GRU_1 = nn.ModuleList()
        for b in range(num_rnn[0]):
            self.GRU_1.append(DGCGRUCell(dim_rnn if b else dim_in, dim_rnn, num_joint[1], choice=choice)) # * 3
        self.GRU_2 = nn.ModuleList()
        for b in range(num_rnn[1]):
            self.GRU_2.append(DGCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn, num_joint[2], choice=choice))
        self.GRU_3 = nn.ModuleList()
        for b in range(num_rnn[2]):
            self.GRU_3.append(DGCGRUCell(dim_rnn if b else dim_in + dim_rnn, dim_rnn, num_joint[3], choice=choice))
        #self.GRU_1 = SemGCGRUCell(dim_in, dim_rnn, num_joint[1] * 3)
        #self.GRU_2 = SemGCGRUCell(dim_in + dim_rnn, dim_rnn, num_joint[2])
        #self.GRU_3 = SemGCGRUCell(dim_in + dim_rnn, dim_rnn, num_joint[3])
        self.ODE_1 = DGNN(dim_rnn, dim_rnn, dim_hidden, num_hidden[0], num_joint[1])# * 3)
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
        #print(s_adjs[0].shape, s_adjs[1].shape, s_adjs[2].shape)
        #d_adjs = data_batch['d_adj_h']
        #print(d_adjs[0].shape, d_adjs[1].shape, d_adjs[2].shape)
        s_adj1 = s_adjs[0].cuda()
        s_adj2 = s_adjs[1].repeat(1, 1, 1).cuda()#.repeat(3, 1, 1).cuda()
        s_adj3 = s_adjs[2].repeat(1, 1, 1, 1).cuda()#.repeat(3, 1, 1, 1).cuda() #Multi
        d_adjs = data_batch['d_adj_h']
        d_adj1 = d_adjs[0].cuda()
        d_adj2 = d_adjs[1].reshape(-1, seq_len, g2.shape[0], g2.shape[0]).cuda()
        d_adj3 = d_adjs[2].reshape(-1, seq_len, g3.shape[0], g3.shape[1], g3.shape[1]).cuda()
        k = 2#(ts[1] - ts[0]) / delta_t
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
                h1_2 = h1_2 + dh1_2 * (k * delta_t)# + h1_1
                # Lv3
                for j in range(1, k + 1):
                    for gi in range(len(h1_3)):
                        dh1_3 = self.ODE_3(torch.cat([h1_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, h1_3[gi].shape[2], 1), h1_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out * 2), s_adj3[:, gi]).reshape(batch_size, num_p, -1, dim_out)
                        h1_3[gi] = h1_3[gi] + dh1_3 * delta_t# + h1_2[:, gi + 1].unsqueeze(1).repeat(1, h1_3[gi].shape[1], 1)
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
            #h2_1 = self.GRU_1(x_1.reshape(-1, num_p, dim_in), h1_1.reshape(-1, num_p, dim_out), d_adj1[:, n])
            # Lv2
            h2_2 = torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
            h1_2 = h1_2.reshape(batch_size * num_p, -1, dim_out)
            for block in self.GRU_2:
                h2_2 = block(h2_2, h1_2, d_adj2[:, n])
            h2_2 = h2_2.reshape(batch_size, num_p, -1, dim_out)
            #h2_2 = self.GRU_2(torch.cat([h2_1.repeat(1, 1, x_2.shape[2], 1), x_2], -1).reshape(batch_size * num_p, -1, dim_out + dim_in), h1_2.reshape(batch_size * num_p, -1, dim_out), d_adj2[:, n]).reshape(batch_size, num_p, -1, dim_out)
            # Lv3
            h2_3s = []
            for gi in range(len(h1_3)):
                h2_3 = torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out + dim_in)
                for block in self.GRU_3:
                    h2_3 = block(h2_3, h1_3[gi].reshape(batch_size * num_p, -1, dim_out), d_adj3[:, n, gi])
                h2_3s.append(h2_3.reshape(batch_size, num_p, -1, dim_out))
                #h2_3s.append(self.GRU_3(torch.cat([h2_2[:, :, gi + 1].unsqueeze(2).repeat(1, 1, x_3[gi].shape[2], 1), x_3[gi]], -1).reshape(batch_size * num_p, -1, dim_out + dim_in), h1_3[gi].reshape(batch_size * num_p, -1, dim_out), d_adj3[:, n, gi]).reshape(batch_size, num_p, -1, dim_out))
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
