"""
    Name    : Utils.py 
    Author  : Yucheng Xing
"""
import torch
import torch.nn as nn


class GCLayer(nn.Module):
    ''' Graph Convolutional Layer '''
    def __init__(self, dim_in, dim_out, num_joint, bn=True, activation=nn.SiLU):
        super(GCLayer, self).__init__()
        self.network = nn.Sequential()
        self.network.add_module('conv', nn.Linear(dim_in, dim_out))
        if bn:
            self.network.add_module('bn', nn.BatchNorm1d(num_joint))
        if activation:
            self.network.add_module('act', activation())
        return

    def forward(self, x, a):
        h = self.network(torch.matmul(a, x))
        return h

"""
class DCLayer(nn.Module):
    ''' Diffusion Convolutional Layer '''
    def __init__(self, dim_in, dim_out, activation=nn.SiLU, k=3, cal_mean=True):
        super(DCLayer, self).__init__()
        self.network = nn.Sequential()

    def forward(self, x, a):
        return x
"""

class DGCLayer(nn.Module):
    ''' Dynamic Graph Convolutional Layer '''
    def __init__(self, dim_in, dim_out, num_joint, activation=nn.SiLU):
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
        w = self.network1(a)#torch.matmul(a, x))#
        x = torch.matmul(a, x)
        h = self.network2(torch.matmul(w, x))
        return h


class SemGCLayer(nn.Module):
    ''' Dynamic Graph Convolutional Layer '''
    def __init__(self, dim_in, dim_out, num_joint, activation=nn.SiLU):
        super(SemGCLayer, self).__init__()
        self.network2 = nn.Sequential()
        self.network2.add_module('conv', nn.Linear(dim_in, dim_out))
        self.network2.add_module('bn', nn.BatchNorm1d(num_joint))
        if activation:
            self.network2.add_module('act', activation())
        self.w = nn.Parameter(0.01 * torch.randn(size=(1, num_joint, num_joint)))
        self.activation = activation
        return

    def forward(self, x, a):
        wa = self.activation()(torch.matmul(self.w, a))
        #wa = torch.softmax(torch.matmul(self.w, a), dim=-1)
        h = self.network2(torch.matmul(wa, x))
        return h

        
class ResTempConv(nn.Module):
    ''' Gated Residual Temporal Convolutional Layer '''
    def __init__(self, dim_in, dim_out, kernel_size):
        super(ResTempConv, self).__init__()
        if dim_in != dim_out:
            self.conv_res_1d = nn.Conv1d(dim_in, dim_out, kernel_size)
        self.conv_1d = nn.Conv1d(dim_in, dim_out, kernel_size)
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        return

    def forward(self, x):
        batch_size, seq_len, num_j, dim_in = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, dim_in, seq_len)
        pad = torch.zeros(x.size(0), x.size(1), self.kernel_size - 1).cuda()
        x = torch.cat([pad, x], -1)
        h = self.conv_1d(x)
        r = self.conv_res_1d(x) if hasattr(self, 'conv_res_1d') else x[:, :, self.kernel_size - 1:]
        h = (h + r).reshape(batch_size, num_j, self.dim_out, seq_len).permute(0, 3, 1, 2)
        return h
