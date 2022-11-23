"""
    Name    : Modules.py 
    Author  : Yucheng Xing
"""
import torch
import torch.nn as nn
from Utils import GCLayer, DGCLayer, ResTempConv, SemGCLayer


class DNN(nn.Module):
    ''' Dense Neural Network '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_joint, activation=nn.ReLU):
        super(DNN, self).__init__()
        self.network = nn.Sequential()
        for l in range(num_hidden):
            self.network.add_module('conv-{}'.format(l), nn.Linear(dim_hidden if l else dim_in, dim_hidden))
            self.network.add_module('bn-{}'.format(l), nn.BatchNorm1d(num_joint))
            self.network.add_module('act-{}'.format(l), activation())
        self.network.add_module('output', nn.Linear(dim_hidden, dim_out))
        return

    def forward(self, x, A=None):
        return self.network(x)


class GNN(nn.Module):
    ''' Graph Neural Network '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_joint, activation=nn.ReLU, choice='F'):
        super(GNN, self).__init__()
        self.network = nn.ModuleList()
        for l in range(num_hidden):
            self.network.append(GCLayer(dim_hidden if l else dim_in, dim_hidden, num_joint, activation))
        self.network.append(nn.Linear(dim_hidden, dim_out))
        self.choice = choice
        return

    def forward(self, x, A):
        if self.choice == 'F':
            tilde_A = A + torch.eye(A.size(-1)).cuda()
            D = torch.sqrt(tilde_A.sum(-1))
            L = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
            if len(L.size()) != len(x.size()):
                L = L.unsqueeze(1)
            h = x
            for block in self.network[0:-1]:
                h = block(h, L)
            h = self.network[-1](h)
        else:
            h = x
            for block in self.network[0:-1]:
                h = block(h, A)
            h = self.network[-1](h)
        return h


class DGNN(GNN):
    ''' Dynamic Graph Neural Network '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_joint, activation=nn.ReLU, choice='F'):
        super(DGNN, self).__init__(dim_in, dim_hidden, dim_out, num_hidden, num_joint, activation, choice)
        self.network = nn.Sequential()
        for l in range(num_hidden):
            self.network.add_module('dgc-{}'.format(l), DGCLayer(dim_hidden if l else dim_in, dim_hidden, num_joint, activation))
        self.network.add_module('output', nn.Linear(dim_hidden, dim_out))
        return


class SemGNN(GNN):
    ''' Sem Dynamic Graph Neural Network '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_joint, activation=nn.ReLU, choice='F'):
        super(SemGNN, self).__init__(dim_in, dim_hidden, dim_out, num_hidden, num_joint, activation, choice)
        self.network = nn.Sequential()
        for l in range(num_hidden):
            self.network.add_module('semgc-{}'.format(l), SemGCLayer(dim_hidden if l else dim_in, dim_hidden, num_joint, activation))
        self.network.add_module('output', nn.Linear(dim_hidden, dim_out))
        return


class GRUCell(nn.Module):
    ''' Dense GRU Cell '''
    def __init__(self, dim_in, dim_out, bias=True):
        super(GRUCell, self).__init__()
        self.lin_Z = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Tanh())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_out)))
        return

    def forward(self, x, h):
        X = torch.cat([x, h], dim=-1)
        Z = self.lin_Z(X)
        R = self.lin_R(X)
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self):
        return self.h0


class GCGRUCell(nn.Module):
    ''' Graph GRU Cell '''
    def __init__(self, dim_in, dim_out, bias=True, choice='F'):
        super(GCGRUCell, self).__init__()
        self.lin_Z = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Tanh())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_out)))
        self.choice = choice
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        if self.choice == 'F':
            tilde_A = A + torch.eye(A.size(-1)).cuda()
            D = torch.sqrt(tilde_A.sum(-1))
            L = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
            if len(L.size()) != len(X.size()):
                L = L.unsqueeze(1)
            Y = torch.matmul(L, X)
        else:
            Y = torch.matmul(A, X)
        Z = self.lin_Z(Y)
        R = self.lin_R(Y)
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self):
        return self.h0


class DGCGRUCell(nn.Module):
    ''' Dynamic Graph GRU Cell '''
    def __init__(self, dim_in, dim_out, num_joint, bias=True, choice='F'):
        super(DGCGRUCell, self).__init__()
        self.lin_Z = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Tanh())
        self.nn_W = nn.Sequential(nn.Linear(num_joint, num_joint), nn.ReLU())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_out)))
        self.choice = choice
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        if self.choice == 'F':
            tilde_A = A + torch.eye(A.size(-1)).cuda()
            D = torch.sqrt(tilde_A.sum(-1))
            L = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
            if len(L.size()) != len(X.size()):
                L = L.unsqueeze(1)
            W = self.nn_W(L)
            Y = torch.matmul(L, X)
        else:
            W = self.nn_W(A)
            Y = torch.matmul(A, X)
        Y = torch.matmul(W, Y)#
        Z = self.lin_Z(Y)
        R = self.lin_R(Y)
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self):
        return self.h0


class SemGCGRUCell(nn.Module):
    ''' Sem Dynamic Graph GRU Cell '''
    def __init__(self, dim_in, dim_out, num_joint, bias=True, choice='F'):
        super(SemGCGRUCell, self).__init__()
        self.lin_Z = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(nn.Linear(dim_in + dim_out, dim_out, bias=bias), nn.Tanh())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dim_out)))
        self.W = nn.Parameter(0.01 * torch.randn(size=(1, num_joint, num_joint)))
        self.choice = choice
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        if self.choice == 'F':
            tilde_A = A + torch.eye(A.size(-1)).cuda()
            D = torch.sqrt(tilde_A.sum(-1))
            L = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
            if len(L.size()) != len(X.size()):
                L = L.unsqueeze(1)
            #Y = torch.matmul(L, X)
            WA = nn.ReLU()(torch.matmul(self.W, L))
        else:
            #Y = torch.matmul(A, X)
            WA = nn.ReLU()(torch.matmul(self.W, A))
        #Y = nn.ReLU()(torch.matmul(self.W, Y))
        #Y = torch.softmax(torch.matmul(self.W, Y), dim=-1)
        Y = torch.matmul(WA, X)
        Z = self.lin_Z(Y)
        R = self.lin_R(Y)
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self):
        return self.h0


class STConvBlock(nn.Module):
    ''' ST-GCN Block '''
    def __init__(self, dim_in, dim_h, dim_out, kernel_size, dim_gcn, num_joint, activation=nn.ReLU, choice='F'):
        super(STConvBlock, self).__init__()
        self.t_conv1 = ResTempConv(dim_in, dim_h, kernel_size)
        self.s_conv = GCLayer(dim_h, dim_gcn, num_joint, False, activation)
        self.t_conv2 = ResTempConv(dim_gcn, dim_out, kernel_size)
        self.choice = choice
        return

    def forward(self, x, A):
        h = self.t_conv1(x)
        if self.choice == 'F':
            tilde_A = A + torch.eye(A.size(-1)).cuda()
            D = torch.sqrt(tilde_A.sum(-1))
            L = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
            if len(L.size()) != len(x.size()):
                L = L.unsqueeze(1)
            h = self.s_conv(h, L)
        else:
            h = self.s_conv(h, A)
        h = self.t_conv2(h)
        return h