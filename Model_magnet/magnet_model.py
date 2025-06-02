import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process(mul_L_real, mul_L_imag, weight_real, weight_imag, X_real, X_imag):
    data = torch.matmul(mul_L_real, X_real.unsqueeze(1))
    real = torch.matmul(data, weight_real.unsqueeze(0))
    data = -1.0 * torch.matmul(mul_L_imag, X_imag.unsqueeze(1))
    real += torch.matmul(data, weight_imag.unsqueeze(0))

    data = torch.matmul(mul_L_imag, X_real.unsqueeze(1))
    imag = torch.matmul(data, weight_real.unsqueeze(0))
    data = torch.matmul(mul_L_real, X_imag.unsqueeze(1))
    imag += torch.matmul(data, weight_imag.unsqueeze(0))
    return torch.stack([real, imag])


class ChebConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super(ChebConv, self).__init__()

        self.weight_real = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        self.weight_imag = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight_real.size(-1))
        self.weight_real.data.uniform_(-stdv, stdv)
        self.weight_imag.data.uniform_(-stdv, stdv)

        magnitude = torch.sqrt(self.weight_real.data**2 + self.weight_imag.data**2)
        self.weight_real.data /= magnitude
        self.weight_imag.data /= magnitude

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data, laplacian):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        mul_data = process(laplacian.real, laplacian.imag, self.weight_real, self.weight_imag, X_real, X_imag)

        result = torch.sum(mul_data, dim=2)

        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias
        # return real , imag


class ChebNet(nn.Module):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=9,
                 dropout=None):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        self.cheb_conv1 = ChebConv(in_c=in_c, out_c=num_filter, K=K)

        self.cheb_conv2 = ChebConv(in_c=num_filter, out_c=num_filter, K=K)

        # self.cardioid = compnn.CVCardiod()
        #
        # self.sigmoid = compnn.CVSigLog()
        #
        # self.tanh = compnn.CVPolarTanh()

        last_dim = 1
        # self.fc = compnn.CVLinear(30 * last_dim, label_dim)
        self.dropout = compnn.CVDropout(dropout)
        self.conv = compnn.CVConv1d(30 * last_dim, label_dim, kernel_size=1)
        # self.fc = nn.Linear(30 * last_dim, label_dim)
        # self.fc = nn.Linear(30, label_dim)
        # complex_kaiming_uniform_(self.conv.weight)

    def forward(self, real, imag, laplacian, layer=2):
        real, imag = self.cheb_conv1((real, imag), laplacian)
        for l in range(1,layer):
            real, imag = self.cheb_conv2((real, imag), laplacian)

        real, imag = torch.mean(real, dim=2), torch.mean(imag, dim=2)

        # real, imag = self.fc(real), self.fc(imag)

        # x = real + 1j*imag

        # x = torch.cat((real, imag), dim=-1)
        #
        # x = self.fc(x)

        x = complextorch.CVTensor(real, imag).to(device)
        # x = self.cardioid(x)
        # x = self.dropout(x)
        x = self.conv(x[:, :, None])
        return x.squeeze(2)
        # return x