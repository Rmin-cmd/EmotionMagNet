import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch
import torch.nn.functional as F
from Model_magnet.Magnet_model_2 import ChebConv
from utils.myBatch import *  # Keep for ChebNet
from utils.hermitian import *  # Keep for ChebNet
from utils.myModReLU import LearnableModReLU

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
        logits = self.attention_scorer(x.abs())             # (batch_size * num_bands, 1)
        logits = logits.view(batch_size, num_bands)   # (batch_size, num_bands)

        # Softmax over bands
        weights = F.softmax(logits, dim=1)            # (batch_size, num_bands)

        # Weighted sum
        weighted = band_embeddings * weights.unsqueeze(-1)  # broadcast
        fused = weighted.sum(dim=1)                   # (batch_size, input_dim)
        return fused, weights


class ChebNet(nn.Module):
    def __init__(self, in_c, args):
        super(ChebNet, self).__init__()
        self.args = args
        self.K = args.K
        self.q = args.q
        self.num_heads = args.num_heads
        self.num_filters = args.num_filter
        self.label_encoding = args.label_encoding
        self.concat_heads_chebconv = args.concat_heads_multi

        # Graph convolutions
        self.cheb_conv1 = ChebConv(
            in_c=in_c,
            out_c=self.num_filters,
            K=self.K,
            use_attention=args.simple_attention,
        )
        self.cheb_conv2 = ChebConv(
            in_c=self.num_filters,
            out_c=self.num_filters,
            K=self.K,
            use_attention=args.simple_attention,
        )

        # Final complex linear and polarization
        output_dim = 9 if args.label_encoding else args.proto_dim
        self.lin = compnn.CVLinear(self.num_filters, 1)
        self.tanh = compnn.CVPolarTanh()
        self.fc_final = compnn.CVLinear(30, output_dim)

        # Frequency attention (on magnitudes)
        self.freq_attn = FrequencySelfAttention(input_dim=output_dim)
        self.bn = ComplexBatchNorm1d(30)
        # self.dropout = compnn.CVDropout(args.dropout)
        self.attention_weights_freq = None

    def forward(self, real, imag, graph, layer=2):
        batch_size, num_bands, N, _ = graph.shape
        band_feats = []

        # Process each band graph
        for b in range(num_bands):
            G = graph[:, b, :, :].to(device)
            her_list = [decomp(G[i], self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                        for i in range(batch_size)]
            her_stack = torch.stack(her_list)
            cheb_list = [cheb_poly(her_stack[i], self.K) for i in range(batch_size)]
            lap = torch.stack(cheb_list)

            r, im = self.cheb_conv1((real, imag), lap)
            if layer > 1:
                r, im = self.cheb_conv2((r, im), lap)

            x = complextorch.CVTensor(r, im).to(device)
            x = self.lin(x).squeeze()    # (B, nodes?) -> (B, output_dim)
            x = self.bn(x)
            x = self.tanh(x)
            x = self.fc_final(x) # (B, output_dim)

            # Magnitude for real attention MLP
            band_feats.append(x.complex)

        # Stack and attend
        all_feats = torch.stack(band_feats, dim=1)  # (B, num_bands, output_dim)
        # fused_complex, weights = self.freq_attn(all_feats)
        # self.attention_weights_freq = weights

        # Back to complex for any further layers
        # zeros = torch.zeros_like(fused_real, device=fused_real.device)
        # fused_complex = complextorch.CVTensor(fused_real, zeros)

        # Optional dropout and return
        # fused_complex = self.dropout(fused_complex)
        return all_feats
