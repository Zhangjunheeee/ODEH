'张钧贺'
import sys
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torchdiffeq import odeint


class ODEFuncW(nn.Module):

    # currently requires in_features = out_features = 2*opt['hidden_dim']
    def __init__(self, in_features, out_features, adj):
        super(ODEFuncW, self).__init__()
        self.adj = adj  # [2708,2708]
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.95
        self.rnn = nn.RNNCell(256, 2)
        self.h = nn.Parameter(torch.ones(adj.shape[1], 2))
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))

        self.w = nn.Parameter(torch.eye(2 * 128))
        self.d = nn.Parameter(torch.zeros(2 * 128) + 1)

    def forward(self, t, x):
        self.nfe += 1

        #print("x的device:",x.device)
        #print("self.h的device:", self.h.device)
        alpha_new = self.rnn(x, self.h)
        alpha_new = alpha_new.squeeze(1)
        scale_alpha = alpha_new[:,0]
        shift_alpha = alpha_new[:,1]
        self.alpha_train = nn.Parameter(self.alpha_train * scale_alpha)
        self.alpha_train = nn.Parameter(self.alpha_train + shift_alpha)

        alph = torch.sigmoid(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.adj.cpu(), x)

        d = torch.clamp(self.d, min=0, max=1)  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.spmm(x, w)

        f = alph * 0.5 * (ax - x) + xw - x + self.x0

        return f


class ODEblockW(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblockW, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.nfe = 0

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        self.nfe += 1

        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t)[1]
        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"


# Define the GNN model.
class WGNN(nn.Module):
    def __init__(self,n_users,n_itmes):
        super(WGNN, self).__init__()
        self.n_users = n_users  # 邻接矩阵
        self.n_itmes = n_itmes

    def forward(self, x, adj):
        self.T = 0.9
        self.odeblock = ODEblockW(ODEFuncW(2 * 128, 2 * 128, adj),
                                  t=torch.tensor([0, self.T]))


        # Solve the initial value problem of the ODE.
        c_aux = torch.zeros(x.shape)
        x = torch.cat([x.cpu(), c_aux], dim=1)
        # print('x的shape:', x.shape)  #[2708,32]
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        # Activation.
        #z = F.relu(z)
        return z