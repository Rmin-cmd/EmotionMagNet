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
        mul_data = self.process(L_real, L_imag, self.weight_real, self.weight_imag, X_real, X_imag)
        result = torch.sum(mul_data, dim=2)  # Sum over polynomial orders
        # real = result[0] + self.bias_real
        # imag = result[1] + self.bias_imag
        return result[0], result[1]

    def process(self, L_real, L_imag, w_real, w_imag, X_real, X_imag):
        # Batched matrix multiplication
        def bmul(A, B):
            return torch.einsum('bijk,bqjp->bijp', A, B)

        # Real component calculations
        term1_real = bmul(L_real, X_real.unsqueeze(1))
        term1_real = torch.matmul(term1_real, w_real)

        term2_real = -1.0 * bmul(L_imag, X_imag.unsqueeze(1))
        term2_real = torch.matmul(term2_real, w_imag)
        real = term1_real + term2_real

        # Imaginary component calculations
        term1_imag = bmul(L_imag, X_real.unsqueeze(1))
        term1_imag = torch.matmul(term1_imag, w_real)

        term2_imag = bmul(L_real, X_imag.unsqueeze(1))
        term2_imag = torch.matmul(term2_imag, w_imag)
        imag = term1_imag + term2_imag

        return torch.stack([real, imag])


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

        self.num_heads = args.num_heads

        self.num_filters = args.num_filter

        self.label_encoding = args.label_encoding

        self.cheb_conv1 = ChebConv(in_c=in_c, out_c=args.num_filter, K=self.K, use_attention=args.simple_attention)

        self.cheb_conv2 = ChebConv(in_c=args.num_filter, out_c=args.num_filter, K=self.K)

        last_dim = 1
        self.dropout = compnn.CVDropout(args.dropout)
        # for the first loss function on label encoding
        # self.conv = compnn.CVConv1d(30 * last_dim, label_dim, kernel_size=1)
        # for the second loss function on class prototypes
        if args.label_encoding:
            self.conv = compnn.CVConv1d(30 * last_dim, 9, kernel_size=1)
        else:
            self.conv = compnn.CVConv1d(30 * last_dim, args.proto_dim, kernel_size=1)

        self.multi_head_attention = compnn.CVLinear(in_c, out_features=args.num_filter * self.num_heads)
        #
        self.tanh = compnn.CVPolarTanh()
        self.bn = ComplexBatchNorm1d(30, affine=False)

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def multi_head_attention_layer(self, real, imag, graph):

        x = complextorch.CVTensor(real, imag)

        x = self.multi_head_attention(x)

        B, N, _ = x.shape

        x = x.view(B, N, self.num_heads, self.num_filters)



    def forward(self, real, imag, graph, layer=2):

        for i in range(graph.shape[1]):


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
        if self.label_encoding:
            return x.squeeze(2)
        else:
            return x

