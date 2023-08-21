"""
    Name    : Utils.py 
    Author  : Yucheng Xing
"""
import torch
import torch.nn as nn


class DGCLayer(nn.Module):
    ''' Dynamic Graph Convolutional Layer '''
    def __init__(self, dim_in, dim_out, num_joint, activation=nn.ReLU):
        super(DGCLayer, self).__init__()
        self.network1 = nn.Sequential()
        self.network1.add_module('conv', nn.Linear(num_joint, num_joint))
        self.network2 = nn.Sequential()
        self.network2.add_module('conv', nn.Linear(dim_in, dim_out))
        self.network2.add_module('bn', nn.BatchNorm1d(num_joint))
        if activation:
            self.network1.add_module('act', activation())
            self.network2.add_module('act', activation())
        return

    def forward(self, x, a):
        w = self.network1(a)
        w_a = torch.matmul(w, a)
        h = self.network2(torch.matmul(w_a, x))
        return h

