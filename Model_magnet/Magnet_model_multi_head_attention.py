import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch
import torch.nn.functional as F
from utils.myBatch import * # Keep for ChebNet
from utils.hermitian import * # Keep for ChebNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, num_heads=1, bias=True, use_attention=True, concat_heads=True): # Added concat_heads
        super(ChebConv, self).__init__()
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.out_c_per_head = out_c
        self.concat_heads = concat_heads

        if self.concat_heads:
            self.actual_out_c = out_c * num_heads
        else:
            self.actual_out_c = out_c # Heads will be averaged

        self.weight_real = nn.Parameter(torch.Tensor(K + 1, in_c, self.actual_out_c))
        self.weight_imag = nn.Parameter(torch.Tensor(K + 1, in_c, self.actual_out_c))

        stdv = 1. / math.sqrt(self.weight_real.size(-1))
        self.weight_real.data.uniform_(-stdv, stdv)
        self.weight_imag.data.uniform_(-stdv, stdv)

        # Optional: Normalize magnitude (if desired for initial weights)
        # magnitude = torch.sqrt(self.weight_real ** 2 + self.weight_imag ** 2) + 1e-9 # avoid div by zero
        # self.weight_real.data /= magnitude
        # self.weight_imag.data /= magnitude

        if self.use_attention:
            self.attn_fc = compnn.CVLinear(2 * in_c, self.num_heads, bias=bias)
            self.cprelu = compnn.CPReLU()
            # self.psoftmax = compnn.PhaseSoftMax(dim=-1) # User code used F.softmax on magnitude

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(1, self.actual_out_c))
            self.bias_imag = nn.Parameter(torch.Tensor(1, self.actual_out_c))
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, data, laplacian_tuple): # laplacian_tuple is (L_real_poly, L_imag_poly)
        # L_real_poly, L_imag_poly should be [B, K+1, N, N]
        X_real, X_imag = data[0], data[1]
        B, N, C_in = X_real.shape
        K_plus_1 = self.weight_real.shape[0] # laplacian_tuple[0].shape[1]

        L_real_poly_full, L_imag_poly_full = laplacian_tuple[0], laplacian_tuple[1]


        if self.use_attention:
            X_complex = X_real + 1j * X_imag
            X_i = X_complex.unsqueeze(2)
            X_j = X_complex.unsqueeze(1)
            attn_input = torch.cat([X_i.expand(-1, -1, N, -1), X_j.expand(-1, N, -1, -1)], dim=-1)
            scores = self.attn_fc(attn_input)
            scores = self.cprelu(scores)

            scores_perm = scores.permute(0, 3, 1, 2) # [B, num_heads, N_i, N_j]

            # Use a simple adjacency mask from the sum of absolute values of K polynomials of the real part of Laplacian
            # This assumes L_real_poly_full is [B, K+1, N, N]
            adj_mask = (torch.abs(L_real_poly_full).sum(dim=1, keepdim=True) > 1e-9).float() # [B, 1, N, N]
            adj_mask_expanded = adj_mask.expand(-1, self.num_heads, -1, -1) # [B, num_heads, N, N]

            # Mask scores
            # Ensure scores_perm.real and scores_perm.imag are used with torch.where
            # Create a very small negative number for masking instead of float('-inf') to avoid potential NaN with some ops
            mask_value = -1e9
            scores_re_masked = torch.where(adj_mask_expanded.bool(), scores_perm.real, torch.tensor(mask_value, device=X_real.device))
            scores_im_masked = torch.where(adj_mask_expanded.bool(), scores_perm.imag, torch.tensor(mask_value, device=X_real.device)) # Not used if softmax on magnitude

            attention_magnitude_sq = scores_re_masked**2 + scores_im_masked**2 # Using masked real part for magnitude
            attn_weights_magnitude = F.softmax(attention_magnitude_sq, dim=-1) # [B, num_heads, N, N]

            # attn_weights_phase = scores_perm.imag.atan2(scores_perm.real) # Phase if needed
            # For now, only magnitude-based attention weights are used for weighting Laplacian

            # Unsqueeze for broadcasting over K+1 polynomials and N_i, N_j
            # attn_weights_magnitude is [B, num_heads, N_i, N_j]
            # We need to make it [B, 1, N_j, N_i, num_heads] if we want to use it as L_att = L * att
            # Or [B, K+1, N_j, N_i, num_heads] after expansion

            # The user code expanded laplacian and then multiplied.
            # L_real_orig_expanded = L_real_poly_full.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_heads) # [B, K+1, N, N, num_heads]
            # L_imag_orig_expanded = L_imag_poly_full.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_heads) # [B, K+1, N, N, num_heads]
            # attn_weights_to_mult = attn_weights_magnitude.permute(0,2,3,1).unsqueeze(1) # [B, 1, N, N, num_heads]

            # L_real_att = L_real_orig_expanded * attn_weights_to_mult # Element-wise
            # L_imag_att = L_imag_orig_expanded * attn_weights_to_mult # Element-wise

            # Simpler: laplacian_output_real/imag will be [B, K+1, N, N, num_heads]
            # attn_weights_magnitude is [B, num_heads, N, N] -> permute to [B, N, N, num_heads] then unsqueeze for K
            attn_for_lap = attn_weights_magnitude.permute(0,2,3,1).unsqueeze(1) # [B, 1, N, N, num_heads]

            L_real_processed = L_real_poly_full.unsqueeze(-1) * attn_for_lap # Broadcast L, multiply by attention
            L_imag_processed = L_imag_poly_full.unsqueeze(-1) * attn_for_lap
        else:
            L_real_processed = L_real_poly_full.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_heads)
            L_imag_processed = L_imag_poly_full.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_heads)

        processed_output_stacked = self.process(L_real_processed, L_imag_processed, self.weight_real, self.weight_imag, X_real, X_imag)

        real_part_raw = processed_output_stacked[0] # [B, N, actual_out_c]
        imag_part_raw = processed_output_stacked[1] # [B, N, actual_out_c]

        if not self.concat_heads and self.num_heads > 1:
             # Average over heads if not concatenating
             real_part_raw = real_part_raw.view(B, N, self.out_c_per_head, self.num_heads).mean(dim=-1)
             imag_part_raw = imag_part_raw.view(B, N, self.out_c_per_head, self.num_heads).mean(dim=-1)

        if self.bias_real is not None:
            # Bias should match the final output dimension (either actual_out_c if concat, or out_c_per_head if averaged)
            bias_real_to_add = self.bias_real
            bias_imag_to_add = self.bias_imag
            # The __init__ already handles bias shape based on concat_heads.
            real_part_raw = real_part_raw + bias_real_to_add
            imag_part_raw = imag_part_raw + bias_imag_to_add

        return real_part_raw, imag_part_raw

    def process(self, L_real_poly_att, L_imag_poly_att, w_real_poly, w_imag_poly, X_node_real, X_node_imag):
        # L_real_poly_att, L_imag_poly_att: (B, K+1, N, N, num_heads)
        # w_real_poly, w_imag_poly: (K+1, C_in, actual_out_c) where actual_out_c = out_c_per_head * num_heads if concat
        # X_node_real, X_node_imag: (B, N, C_in)
        # Output: sum_LXW_real, sum_LXW_imag of shape (B, N, actual_out_c)

        B, K_plus_1, N, _, num_h = L_real_poly_att.shape
        C_in = X_node_real.shape[-1]
        actual_out_c = w_real_poly.shape[-1]

        sum_LXW_real = torch.zeros(B, N, actual_out_c, device=X_node_real.device)
        sum_LXW_imag = torch.zeros(B, N, actual_out_c, device=X_node_imag.device)

        for k in range(K_plus_1):
            Lk_real_att = L_real_poly_att[:, k, :, :, :]    # (B, N, N, num_heads)
            Lk_imag_att = L_imag_poly_att[:, k, :, :, :]    # (B, N, N, num_heads)

            Wk_real = w_real_poly[k, :, :]      # (C_in, actual_out_c)
            Wk_imag = w_imag_poly[k, :, :]      # (C_in, actual_out_c)

            # Expand X_node for per-head processing if num_heads > 1 for LX_real_k/LX_imag_k
            # X_node_real_exp/imag_exp: (B,N,C_in,1) to enable broadcasting with Lk_att (B,N,N,num_h)
            X_node_real_exp = X_node_real.unsqueeze(-1)
            X_node_imag_exp = X_node_imag.unsqueeze(-1)

            # LX_real_k should be [B, N, C_in, num_heads]
            LX_real_k = torch.einsum('bnjh,bjch->bnch', Lk_real_att, X_node_real_exp) - \
                        torch.einsum('bnjh,bjch->bnch', Lk_imag_att, X_node_imag_exp)
            LX_imag_k = torch.einsum('bnjh,bjch->bnch', Lk_real_att, X_node_imag_exp) + \
                        torch.einsum('bnjh,bjch->bnch', Lk_imag_att, X_node_real_exp)

            if self.concat_heads and self.num_heads > 1:
                Wk_real_p = Wk_real.view(C_in, self.out_c_per_head, num_h)
                Wk_imag_p = Wk_imag.view(C_in, self.out_c_per_head, num_h)

                current_real = torch.einsum('bnch,cph->bnph', LX_real_k, Wk_real_p) - \
                               torch.einsum('bnch,cph->bnph', LX_imag_k, Wk_imag_p)
                current_imag = torch.einsum('bnch,cph->bnph', LX_real_k, Wk_imag_p) + \
                               torch.einsum('bnch,cph->bnph', LX_imag_k, Wk_real_p)

                sum_LXW_real += current_real.reshape(B, N, actual_out_c)
                sum_LXW_imag += current_imag.reshape(B, N, actual_out_c)
            else: # num_heads = 1 or not concatenating (averaging)
                if not self.concat_heads and self.num_heads > 1: # Averaging case
                    LX_real_k_avg = LX_real_k.mean(dim=-1) # [B, N, C_in]
                    LX_imag_k_avg = LX_imag_k.mean(dim=-1) # [B, N, C_in]
                     # Wk_real/Wk_imag are already (C_in, out_c_per_head) in this case from __init__
                else: # num_heads == 1 (concat_heads doesn't matter or is true)
                    LX_real_k_avg = LX_real_k.squeeze(-1) # [B, N, C_in]
                    LX_imag_k_avg = LX_imag_k.squeeze(-1) # [B, N, C_in]
                     # Wk_real/Wk_imag are (C_in, out_c_per_head * 1)

                sum_LXW_real += torch.matmul(LX_real_k_avg, Wk_real) - torch.matmul(LX_imag_k_avg, Wk_imag)
                sum_LXW_imag += torch.matmul(LX_real_k_avg, Wk_imag) + torch.matmul(LX_imag_k_avg, Wk_real)

        return torch.stack([sum_LXW_real, sum_LXW_imag])


class FrequencySelfAttention(nn.Module):
    def __init__(self, input_dim, attention_hidden_dim=None):
        super(FrequencySelfAttention, self).__init__()
        self.input_dim = input_dim
        if attention_hidden_dim is None:
            attention_hidden_dim = input_dim // 2 if input_dim // 2 > 0 else 1 # Ensure hidden_dim is at least 1

        # Simple scoring: a small MLP to compute an attention score (logit) for each band embedding
        # This MLP will be applied independently to each of the 5 band embeddings
        self.attention_scorer = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(), # Using Tanh as an activation
            nn.Linear(attention_hidden_dim, 1) # Outputs a single score per band embedding
        )

    def forward(self, band_embeddings):
        # input: band_embeddings - shape (batch_size, num_bands, input_dim)
        # Note: This module expects real-valued inputs for nn.Linear.
        # If band_embeddings are complex, they need to be handled (e.g. take abs, or process real/imag separately)
        # Assuming real-valued band_embeddings for now.
        batch_size, num_bands, _ = band_embeddings.shape

        # Compute scores for each band embedding
        # To apply attention_scorer to each band embedding independently, reshape and then reshape back
        # (batch_size * num_bands, input_dim)
        reshaped_embeddings = band_embeddings.reshape(batch_size * num_bands, self.input_dim)

        # attention_logits will be (batch_size * num_bands, 1)
        attention_logits = self.attention_scorer(reshaped_embeddings)

        # Reshape back to (batch_size, num_bands, 1) and then squeeze to (batch_size, num_bands)
        attention_logits = attention_logits.view(batch_size, num_bands, 1).squeeze(-1) # Shape: (batch_size, num_bands)

        # Apply softmax to get attention weights
        # These weights indicate the importance of each band
        attention_weights = F.softmax(attention_logits, dim=1) # Shape: (batch_size, num_bands)

        # Multiply original band_embeddings by their attention_weights
        # attention_weights need to be unsqueezed to match dimensions for broadcasting: (batch_size, num_bands, 1)
        weighted_embeddings = band_embeddings * attention_weights.unsqueeze(-1)

        # Sum the weighted embeddings across the bands to get the final context vector
        # final_combined_embedding shape: (batch_size, input_dim)
        final_combined_embedding = torch.sum(weighted_embeddings, dim=1)

        return final_combined_embedding, attention_weights # Return weights for potential analysis/logging


# Existing ChebNet class follows (modified for new ChebConv and FrequencySelfAttention)
class ChebNet(nn.Module):
    def __init__(self, in_c, args):
        super(ChebNet, self).__init__()
        self.args = args # Store args
        self.K = args.K
        self.q = args.q
        self.num_heads = args.num_heads
        self.num_filters = args.num_filter # This is out_c_per_head for ChebConv
        self.label_encoding = args.label_encoding

        # Assuming concat_heads=True for ChebConv layers
        self.concat_heads_chebconv = True # Can be made an arg if needed

        self.cheb_conv1 = ChebConv(
            in_c=in_c,
            out_c=self.num_filters,
            K=self.K,
            num_heads=self.num_heads,
            use_attention=args.simple_attention,
            concat_heads=self.concat_heads_chebconv
        )

        in_c_for_conv2 = self.num_filters * self.num_heads if self.concat_heads_chebconv else self.num_filters

        self.cheb_conv2 = ChebConv(
            in_c=in_c_for_conv2,
            out_c=self.num_filters,
            K=self.K,
            num_heads=self.num_heads,
            use_attention=True,
            concat_heads=self.concat_heads_chebconv
        )

        gcn_output_dim = self.num_filters * self.num_heads if self.concat_heads_chebconv else self.num_filters

        self.frequency_attention = FrequencySelfAttention(input_dim=gcn_output_dim)

        self.dropout = compnn.CVDropout(args.dropout)

        output_dim_final_layer = 9 if args.label_encoding else args.proto_dim
        self.fc_final = compnn.CVLinear(gcn_output_dim, output_dim_final_layer)

        self.tanh = compnn.CVPolarTanh()
        self.attention_weights_freq = None # To store attention weights

    def forward(self, real, imag, graph, layer=2): # graph is A_pdc (B, num_bands, N, N)

        band_embeddings_list = []
        num_bands = graph.shape[1]

        for i in range(num_bands): # Iterate over frequency bands
            current_band_graph = graph[:, i, :, :] # Shape (B, N, N)

            # Ensure current_band_graph is on the correct device before passing to decomp
            current_band_graph_device = current_band_graph.to(device if her_mat_list_needs_device else current_band_graph.device)


            her_mat_list_for_band = [decomp(data_item, self.q, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                                     for data_item in current_band_graph_device]
            her_mat = torch.stack(her_mat_list_for_band)

            cheb_graph_complex_list = [cheb_poly(her_mat[j], self.K) for j in range(her_mat.shape[0])]
            cheb_graph_complex = torch.stack(cheb_graph_complex_list)

            laplacian_tuple = (cheb_graph_complex.real, cheb_graph_complex.imag)

            # Pass initial real, imag features for each band's graph structure
            # These features (real, imag) are (B, N, C_in)
            band_real, band_imag = self.cheb_conv1((real, imag), laplacian_tuple)

            if layer > 1:
                band_real, band_imag = self.cheb_conv2((band_real, band_imag), laplacian_tuple)

            # Global Graph Pooling: Spatial mean pooling over nodes
            # band_real, band_imag are (B, N, gcn_output_dim)
            pooled_real = torch.mean(band_real, dim=1) # Shape (B, gcn_output_dim)
            pooled_imag = torch.mean(band_imag, dim=1) # Shape (B, gcn_output_dim)

            # Convert complex pooled embedding to real magnitude for FrequencySelfAttention
            complex_pooled_embedding = pooled_real + 1j * pooled_imag
            magnitude_embedding = torch.abs(complex_pooled_embedding) # Shape (B, gcn_output_dim)
            band_embeddings_list.append(magnitude_embedding)

        # Stack band embeddings: (B, num_bands, gcn_output_dim)
        all_band_embeddings = torch.stack(band_embeddings_list, dim=1)

        # Pass through frequency attention
        # combined_embedding_real is real, shape (B, gcn_output_dim)
        combined_embedding_real, self.attention_weights_freq = self.frequency_attention(all_band_embeddings)

        # Convert real combined_embedding to complex for subsequent CV layers (imaginary part is zero)
        # Ensure device consistency
        zeros_for_imag = torch.zeros_like(combined_embedding_real, device=combined_embedding_real.device)
        combined_embedding_complex = complextorch.CVTensor(combined_embedding_real, zeros_for_imag)

        # Pass through Tanh and final CVLinear layer
        # x = self.dropout(combined_embedding_complex) # Optional dropout
        x = self.tanh(combined_embedding_complex)
        x = self.fc_final(x)

        return x
