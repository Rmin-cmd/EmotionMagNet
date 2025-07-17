import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.hermitian import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChebConvReal(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, use_attention=True):
        super(ChebConvReal, self).__init__()
        self.use_attention = use_attention
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)

        if self.use_attention:
            self.attn_fc = nn.Linear(2 * in_c, 1)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.softmax = nn.Softmax(dim=-1)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x, laplacian):
        B, N, C = x.shape
        laplacian = laplacian.real # Use only the real part of the Laplacian

        if self.use_attention:
            X_i = x.unsqueeze(2).expand(-1, -1, N, -1)
            X_j = x.unsqueeze(1).expand(-1, N, -1, -1)
            attn_input = torch.cat([X_i, X_j], dim=-1)
            scores = self.attn_fc(attn_input).squeeze(-1)
            scores = self.leaky_relu(scores)
            attn_weights = self.softmax(scores)
            laplacian = laplacian * attn_weights.unsqueeze(1)

        # Graph convolution
        sum_lxw = torch.zeros(B, N, self.weight.shape[2], device=x.device)
        for k in range(self.weight.shape[0]):
            Lk = laplacian[:, k, :, :]
            Wk = self.weight[k, :, :]
            lx = torch.matmul(Lk, x)
            lxw = torch.matmul(lx, Wk)
            sum_lxw += lxw

        output = sum_lxw
        if self.bias is not None:
            output += self.bias
        return output


class ChebNetReal(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(ChebNetReal, self).__init__()
        args = kwargs['args']
        self.K, self.q = args.K, args.q
        self.cheb_conv1 = ChebConvReal(in_c=in_c, out_c=args.num_filter, K=self.K, use_attention=args.simple_attention)
        self.cheb_conv2 = ChebConvReal(in_c=args.num_filter, out_c=args.num_filter, K=self.K, use_attention=args.simple_attention)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        # We only use the real part of the input
        x = real
        graph = torch.mean(graph, dim=1)
        her_mat = torch.stack([decomp(data, self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                            for data in graph])
        cheb_graph = torch.stack([cheb_poly(her_mat[i], self.K) for i in range(her_mat.shape[0])]).to(device)

        x = self.cheb_conv1(x, cheb_graph)
        x = self.relu(x)
        x = self.dropout(x)

        for l in range(1, layer):
            x = self.cheb_conv2(x, cheb_graph)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.lin(x).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1) # Add a dimension for Conv1d
        x = self.conv(x)
        return x.squeeze(-1)
