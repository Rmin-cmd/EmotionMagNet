import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch
import torch.nn.functional as F
import numpy as np
from utils.myBatch import *
from utils.myModReLU import *
from utils.hermition_2 import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FrequencySelfAttention(nn.Module):
    def __init__(self, input_dim, attention_hidden_dim=None):
        super(FrequencySelfAttention, self).__init__()
        self.input_dim = input_dim
        if attention_hidden_dim is None:
            attention_hidden_dim = input_dim // 2 if input_dim // 2 > 0 else 1

        # Scoring MLP
        self.attention_scorer = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1)
        )

    def forward(self, band_embeddings):
        # band_embeddings: (batch_size, num_bands, input_dim)
        batch_size, num_bands, _ = band_embeddings.shape

        # Flatten for MLP
        x = band_embeddings.reshape(batch_size * num_bands, self.input_dim)
        logits = self.attention_scorer(x.abs())  # (batch_size * num_bands, 1)
        logits = logits.view(batch_size, num_bands)  # (batch_size, num_bands)

        # Softmax over bands
        weights = F.softmax(logits, dim=1)  # (batch_size, num_bands)

        # Weighted sum
        weighted = band_embeddings * weights.unsqueeze(-1)  # broadcast
        fused = weighted.sum(dim=1)  # (batch_size, input_dim)
        return fused, weights


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
            self.attn_fc = compnn.CVLinear(2 * in_c, 1, bias=bias)
            self.cprelu = LearnableModReLU()
            self.psoftmax = compnn.CVSoftMax(dim=-1)

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
            X_combined = X_real + 1j * X_imag

            # Compute attention scores
            X_i = X_combined.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 2C]
            X_j = X_combined.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 2C]
            X = torch.cat([X_i, X_j], dim=-1)  # [B, N, N, 4C]

            X = complextorch.CVTensor(r=X.real, i=X.imag)
            scores = self.attn_fc(X).squeeze(-1)  # [B, N, N]
            scores = self.cprelu(scores)
            attn_weights = self.psoftmax(scores)

            laplacian = laplacian * attn_weights.unsqueeze(1)

            L_real = laplacian.real
            L_imag = laplacian.imag
        else:
            L_real = laplacian.real
            L_imag = laplacian.imag

        # Process with attention-adjusted Laplacian
        processed_output = self.process(L_real, L_imag, self.weight_real, self.weight_imag, X_real, X_imag)

        return processed_output[0], processed_output[1]

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

            Wk_real = w_real_poly[k, :, :]  # (C_in, C_out)
            Wk_imag = w_imag_poly[k, :, :]  # (C_in, C_out)

            # Compute Lk @ X_node = (Lk_real + i*Lk_imag) @ (X_node_real + i*X_node_imag)
            LX_real_k = torch.matmul(Lk_real, X_node_real) - torch.matmul(Lk_imag, X_node_imag)  # (B,N,C_in)
            LX_imag_k = torch.matmul(Lk_real, X_node_imag) + torch.matmul(Lk_imag, X_node_real)  # (B,N,C_in)

            # Compute (Lk @ X_node) @ Wk = (LX_real_k + i*LX_imag_k) @ (Wk_real + i*Wk_imag)
            sum_LXW_real += torch.matmul(LX_real_k, Wk_real) - torch.matmul(LX_imag_k, Wk_imag)
            sum_LXW_imag += torch.matmul(LX_real_k, Wk_imag) + torch.matmul(LX_imag_k, Wk_real)

        return torch.stack([sum_LXW_real, sum_LXW_imag])

    def PhaseSoftMax(self, z, eps=1e-6):
        # z: (...)x n complex tensor
        r = z.abs()  # magnitudes
        phi = z.angle()  # phases
        M = r.max(dim=-1, keepdim=True)[0]
        exp_shift = torch.exp(r - M)  # stable exp
        p = exp_shift / (exp_shift.sum(dim=-1, keepdim=True) + eps)
        return p * torch.exp(1j * phi)


class ChebNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        """
        Integrated ChebNet with frequency attention and multi-band processing
        :param in_c: int, number of input channels.
        """
        super(ChebNet, self).__init__()

        self.args = kwargs['args']
        self.K, self.q = self.args.K, self.args.q
        self.num_heads = getattr(self.args, 'num_heads', 1)
        self.num_filters = self.args.num_filter
        self.label_encoding = self.args.label_encoding
        self.concat_heads_chebconv = getattr(self.args, 'concat_heads_multi', False)

        # Graph convolutions
        self.cheb_conv1 = ChebConv(
            in_c=in_c,
            out_c=self.num_filters,
            K=self.K,
            use_attention=self.args.simple_attention,
        )
        self.cheb_conv2 = ChebConv(
            in_c=self.num_filters,
            out_c=self.num_filters,
            K=self.K,
            use_attention=self.args.simple_attention,
        )

        # Dropout
        self.dropout = compnn.CVDropout(self.args.dropout)

        # Output layers - determine output dimension based on loss type
        if self.args.label_encoding:
            output_dim = 9  # For label encoding
        elif hasattr(self.args, 'proto_dim'):
            output_dim = self.args.proto_dim  # For prototype-based losses
        else:
            output_dim = self.args.num_classes  # Fallback

        # Linear layers
        self.lin = compnn.CVLinear(self.num_filters, 1)
        self.tanh = compnn.CVPolarTanh()
        self.bn = ComplexBatchNorm1d(30)

        # Final output layer
        if self.args.label_encoding or getattr(self.args, 'simple_magnet', False):
            if getattr(self.args, 'concat', False):
                self.fc = nn.Conv1d(2 * 2, self.args.num_classes, kernel_size=1)
            elif getattr(self.args, 'simple_magnet', False):
                self.fc = compnn.CVConv1d(30, self.args.num_classes, kernel_size=1)
            else:
                # For multi-band processing, use fc_final
                self.fc_final = compnn.CVLinear(30, output_dim)
        else:
            # For prototype-based methods
            if hasattr(self, 'fc_final'):
                self.fc_final = compnn.CVLinear(30, output_dim)
            else:
                self.fc = compnn.CVConv1d(30, output_dim, kernel_size=1)

        # Frequency attention (for multi-band processing)
        self.freq_attn = FrequencySelfAttention(input_dim=output_dim)
        self.attention_weights_freq = None

        # Support both single-band and multi-band modes
        self.multi_band_mode = self.args.multi_head_attention

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, imag, graph, layer=2):
        # Handle both single-band and multi-band inputs
        if self.multi_band_mode and graph.dim() == 4:
            # Multi-band processing: graph shape (B, Nb, N, N)
            return self._forward_multiband(real, imag, graph, layer)
        else:
            # Single-band processing: original behavior
            return self._forward_singleband(real, imag, graph, layer)

    def _forward_singleband(self, real, imag, graph, layer=2):
        """Original single-band forward pass"""
        # Handle frequency band selection
        if self.args.Brain_frequency == "delta":
            graph = graph[:, 2]
        elif self.args.Brain_frequency == "theta":
            graph = graph[:, 4]
        elif self.args.Brain_frequency == "alpha":
            graph = graph[:, 0]
        elif self.args.Brain_frequency == "beta":
            graph = graph[:, 1]
        elif self.args.Brain_frequency == "gamma":
            graph = graph[:, 3]
        elif self.args.Brain_frequency == "average":
            graph = torch.mean(graph, dim=1)

        # Process Laplacian
        her_mat = decomp(graph, self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
        cheb_graph = cheb_poly(her_mat, self.K).to(device)

        # Graph convolutions
        real, imag = self.cheb_conv1((real, imag), cheb_graph)
        for l in range(1, layer):
            real, imag = self.cheb_conv2((real, imag), cheb_graph)

        # Output processing
        if self.args.concat:
            real, imag = torch.mean(real, dim=1), torch.mean(imag, dim=1)
            x = torch.cat((real, imag), dim=-1)
            x = self.fc(x[:, :, None])
        else:
            x = complextorch.CVTensor(real, imag).to(device)
            x = self.lin(x).squeeze()
            x = self.bn(x)
            x = self.tanh(x)
            x = self.fc(x[:, :, None])

        return x.squeeze(2)

    def _forward_multiband(self, real, imag, graph, layer=2):
        """New multi-band forward pass with frequency attention"""
        batch_size, num_bands, N, _ = graph.shape

        # Reshape graph for simultaneous band processing
        graph_reshaped = graph.reshape(batch_size * num_bands, N, N)

        # Vectorized Laplacian calculation
        her_mat = decomp(graph_reshaped, self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
        lap = cheb_poly(her_mat, self.K).to(device)

        # Expand features for all bands
        C_in = real.shape[-1]
        real_expanded = real.unsqueeze(1).expand(-1, num_bands, -1, -1).reshape(batch_size * num_bands, N, C_in)
        imag_expanded = imag.unsqueeze(1).expand(-1, num_bands, -1, -1).reshape(batch_size * num_bands, N, C_in)

        # Apply graph convolutions
        r, im = self.cheb_conv1((real_expanded, imag_expanded), lap)
        for l in range(1, layer):
            r, im = self.cheb_conv2((r, im), lap)

        # Process through network
        x = complextorch.CVTensor(r, im).to(device)
        x = self.lin(x).squeeze()
        x = self.bn(x)
        x = self.tanh(x)

        # Final output layer
        if hasattr(self, 'fc_final'):
            x = self.fc_final(x)
        else:
            x = self.fc(x[:, :, None]).squeeze(2)

        # Reshape output back to (B, Nb, D_out)
        output_dim = x.shape[-1] if x.dim() > 1 else 1
        all_feats = x.reshape(batch_size, num_bands, output_dim)

        # Apply frequency attention if enabled
        if hasattr(self, 'freq_attn') and self.freq_attn is not None:
            fused_output, self.attention_weights_freq = self.freq_attn(all_feats)
            return fused_output
        else:
            return all_feats