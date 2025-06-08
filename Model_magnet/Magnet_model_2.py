import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch
import torch.nn.functional as F
from utils.myBatch import *
from utils.hermitian import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, use_attention=True):
        super(ChebConv, self).__init__()
        self.use_attention = use_attention

        # Original weights
        self.weight_real = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))
        self.weight_imag = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))

        # Initialize weights
        stdv = 1. / math.sqrt(self.weight_real.size(-1))
        self.weight_real.data.uniform_(-stdv, stdv)
        self.weight_imag.data.uniform_(-stdv, stdv)

        # Normalize magnitude
        magnitude = torch.sqrt(self.weight_real ** 2 + self.weight_imag ** 2)
        self.weight_real.data /= magnitude
        self.weight_imag.data /= magnitude

        # Attention parameters
        if self.use_attention:
            # self.attn_fc = nn.Linear(4 * in_c, 1)  # Concatenates real+imag features
            self.attn_fc = compnn.CVLinear(2 * in_c, 1, bias=bias)
            self.cprelu = compnn.CPReLU()
            # self.cprelu = nn.PReLU()
            self.psoftmax = compnn.PhaseSoftMax(dim=1)
            # self.psoftmax = nn.Softmax(dim=1)

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(1, out_c))
            self.bias_imag = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, data, laplacian):
        X_real, X_imag = data[0], data[1]
        B, N, C = X_real.shape

        # Compute attention weights
        if self.use_attention:
            # Combine real and imaginary features
            # X_combined = torch.cat([X_real, X_imag], dim=-1)  # [B, N, 2C]
            X_combined = X_real + 1j*X_imag

            # Compute attention scores
            X_i = X_combined.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 2C]
            X_j = X_combined.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 2C]
            X = torch.cat([X_i, X_j], dim=-1)  # [B, N, N, 4C]
            # X = complextorch.CVTensor(X_i, X_j)

            # scores = self.attn_fc(X.complex).squeeze(-1)  # [B, N, N]
            scores = self.attn_fc(X).squeeze(-1)  # [B, N, N]
            # scores = self.attn_fc(concat).squeeze(-1)  # [B, N, N]
            scores = self.cprelu(scores)
            attn_weights = self.psoftmax(scores)
            # scores = F.leaky_relu(scores)
            # attn_weights = F.softmax(scores, dim=-1)  # [B, N, N]

            laplacian = laplacian * attn_weights.unsqueeze(1)
            L_real = laplacian.real
            L_imag = laplacian.imag

            # Apply attention to Laplacian
            # L_real = laplacian.real * attn_weights.unsqueeze(1)  # [B, N, N]
            # L_imag = laplacian.imag * attn_weights.unsqueeze(1)  # [B, N, N]
        else:
            L_real = laplacian.real
            L_imag = laplacian.imag

        # Process with attention-adjusted Laplacian
        # L_real, L_imag are from laplacian (B, K+1, N, N)
        # X_real, X_imag are (B, N, C_in)
        # self.weight_real, self.weight_imag are (K+1, C_in, C_out)

        processed_output = self.process(L_real, L_imag, self.weight_real, self.weight_imag, X_real, X_imag)
        # processed_output is torch.stack([sum_LXW_real, sum_LXW_imag]), shape (2, B, N, C_out)

        # Bias addition was commented out, but if it were to be added:
        # real_part = processed_output[0] + self.bias_real # self.bias_real is (1, C_out)
        # imag_part = processed_output[1] + self.bias_imag # self.bias_imag is (1, C_out)
        # return real_part, imag_part

        return processed_output[0], processed_output[1] # No bias addition as per original commented out line

    def process(self, L_real_poly, L_imag_poly, w_real_poly, w_imag_poly, X_node_real, X_node_imag):
        # L_real_poly, L_imag_poly: (B, K+1, N, N)
        # w_real_poly, w_imag_poly: (K+1, C_in, C_out)
        # X_node_real, X_node_imag: (B, N, C_in)
        # Output: sum_LXW_real, sum_LXW_imag of shape (B, N, C_out)

        B, K_plus_1, N, _ = L_real_poly.shape
        C_in = X_node_real.shape[-1]
        C_out = w_real_poly.shape[-1]

        sum_LXW_real = torch.zeros(B, N, C_out, device=X_node_real.device)
        sum_LXW_imag = torch.zeros(B, N, C_out, device=X_node_real.device)

        for k in range(K_plus_1):
            Lk_real = L_real_poly[:, k, :, :]  # (B, N, N)
            Lk_imag = L_imag_poly[:, k, :, :]  # (B, N, N)

            Wk_real = w_real_poly[k, :, :]    # (C_in, C_out)
            Wk_imag = w_imag_poly[k, :, :]    # (C_in, C_out)

            # Compute Lk @ X_node = (Lk_real + i*Lk_imag) @ (X_node_real + i*X_node_imag)
            # LX_real_k = Lk_real @ X_node_real - Lk_imag @ X_node_imag
            # LX_imag_k = Lk_real @ X_node_imag + Lk_imag @ X_node_real

            # Using torch.matmul which handles batching.
            LX_real_k = torch.matmul(Lk_real, X_node_real) - torch.matmul(Lk_imag, X_node_imag) # (B,N,C_in)
            LX_imag_k = torch.matmul(Lk_real, X_node_imag) + torch.matmul(Lk_imag, X_node_real) # (B,N,C_in)

            # Compute (Lk @ X_node) @ Wk = (LX_real_k + i*LX_imag_k) @ (Wk_real + i*Wk_imag)
            # LXW_real_k = LX_real_k @ Wk_real - LX_imag_k @ Wk_imag
            # LXW_imag_k = LX_real_k @ Wk_imag + LX_imag_k @ Wk_real

            sum_LXW_real += torch.matmul(LX_real_k, Wk_real) - torch.matmul(LX_imag_k, Wk_imag)
            sum_LXW_imag += torch.matmul(LX_real_k, Wk_imag) + torch.matmul(LX_imag_k, Wk_real)

        return torch.stack([sum_LXW_real, sum_LXW_imag])


class ChebNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        args = kwargs['args']

        self.K, self.q = args.K, args.q

        self.label_encoding = args.label_encoding

        self.cheb_conv1 = ChebConv(in_c=in_c, out_c=args.num_filter, K=self.K, use_attention=args.simple_attention)

        self.cheb_conv2 = ChebConv(in_c=args.num_filter, out_c=args.num_filter, K=self.K,
                                   use_attention=args.simple_attention)

        last_dim = 1
        self.dropout = compnn.CVDropout(args.dropout)
        # for the first loss function on label encoding
        # self.conv = compnn.CVConv1d(30 * last_dim, label_dim, kernel_size=1)
        # for the second loss function on class prototypes
        if args.label_encoding or args.simple_magnet:
            self.conv = compnn.CVConv1d(30 * last_dim, args.num_classes, kernel_size=1)
        else:
            print(type(args.proto_dim))
            self.conv = compnn.CVConv1d(30 * last_dim, args.proto_dim, kernel_size=1)

        #
        self.tanh = compnn.CVPolarTanh()
        self.bn = ComplexBatchNorm1d(30)

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, imag, graph, layer=2):

        graph = torch.mean(graph, dim=1)

        her_mat = torch.stack([decomp(data, self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                            for data in graph])

        cheb_graph = torch.stack([cheb_poly(her_mat[i], self.K) for i in range(her_mat.shape[0])]).to(device)

        real, imag = self.cheb_conv1((real, imag), cheb_graph)
        # print("cheb Conv1:",self.cheb_conv1.weight_real.requires_grad_())
        for l in range(1,layer):
            real, imag = self.cheb_conv2((real, imag), cheb_graph)
            # real, imag = self.complex_relu(real, imag)

        real, imag = torch.mean(real, dim=2), torch.mean(imag, dim=2)

        x = complextorch.CVTensor(real, imag).to(device)
        # x = self.bn(x)
        x = self.tanh(x)
        x = self.conv(x[:, :, None])
        # for the first loss function label encoding
        # if self.label_encoding:
        return x.squeeze(2)
        # else:
        #     return x

