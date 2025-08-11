import torch
import torch.nn as nn
import torch.nn.functional as F
import complextorch
import complextorch.nn as compnn

class ComplexCrossAttention(nn.Module):
    """
    Cross-attention where learnable complex prototypes query complex band embeddings.
    Outputs attended embeddings and attention weights.
    """
    def __init__(self, num_classes, dim):
        super().__init__()
        self.C, self.d = num_classes, dim
        # one Parameter of shape (2, C, d) for real & imag parts
        self.prototypes_param = nn.Parameter(torch.randn(2, self.C, self.d))

    def forward(self, band_feats: complextorch.CVTensor):
        # band_feats.real/.imag: (B, Bn, d)
        B, Bn, d = band_feats.real.shape
        dev = band_feats.real.device

        # move prototypes to same device as inputs
        proto = self.prototypes_param.to(dev)       # (2, C, d)
        Pr    = proto[0].unsqueeze(0)               # (1, C, d)
        Pi    = proto[1].unsqueeze(0)               # (1, C, d)

        # expand band embeddings for key/value
        Zr = band_feats.real.unsqueeze(1)           # (B, 1, Bn, d)
        Zi = band_feats.imag.unsqueeze(1)           # (B, 1, Bn, d)

        # 1) Hermitian inner-product scores: Re(conj(P)·Z)
        scores = (Pr[:,:,None,:] * Zr + Pi[:,:,None,:] * Zi).sum(-1)  # (B, C, Bn)

        # 2) attention weights over bands
        alpha = F.softmax(scores, dim=2)            # (B, C, Bn)

        # 3) weighted sum to get attended embeddings
        alpha_e = alpha.unsqueeze(-1)               # (B, C, Bn, 1)
        hat_r   = (alpha_e * Zr).sum(2)             # (B, C, d)
        hat_i   = (alpha_e * Zi).sum(2)             # (B, C, d)

        return complextorch.CVTensor(hat_r, hat_i), alpha


class PrototypeAttentionLoss(nn.Module):
    """
    Cross-attention prototype loss with selectable distance metric (L1, L2 or orthogonal)
    and soft-MSE regularizer.
    """
    def __init__(self, num_classes, dist_features,
                 distance_metric='orth',
                 temperature=1.0, gmm_lambda=0.01, criterion=None):
        super().__init__()
        self.num_classes     = num_classes
        self.distance_metric = distance_metric
        self.temperature     = temperature
        self.gmm_lambda      = gmm_lambda
        self.cross_attn      = ComplexCrossAttention(num_classes, dist_features)
        self.criterion = criterion

    def forward(self, band_feats: complextorch.CVTensor, labels):
        """
        band_feats: CVTensor of shape (B, num_bands, dim)
        labels:     LongTensor of shape (B,)
        """
        dev = band_feats.real.device
        labels = labels.to(dev).long()

        # 1) Cross-attention: prototypes query bands
        Z, attn = self.cross_attn(band_feats)      # CVTensor (B, C, d)
        Zr, Zi = Z.real, Z.imag                    # each (B, C, d)

        # 2) Re-extract prototypes on correct device
        proto = self.cross_attn.prototypes_param.to(dev)  # (2, C, d)
        Pr    = proto[0].unsqueeze(0)                     # (1, C, d)
        Pi    = proto[1].unsqueeze(0)                     # (1, C, d)

        # 3) Soft-MSE regularizer: uses squared distances
        dr = Zr - Pr
        di = Zi - Pi
        D2 = (dr.pow(2) + di.pow(2)).sum(dim=2)           # (B, C)
        r  = F.softmax(-D2 / self.temperature, dim=1)     # (B, C)
        loss_softmse = (r * D2).sum(dim=1).mean()

        # 4) Classification distances
        if self.distance_metric == 'L1':
            D = (dr.abs() + di.abs()).sum(dim=2)          # L1
        elif self.distance_metric == 'L2':
            D = torch.sqrt(D2 + 1e-8)                     # Euclidean
        elif self.distance_metric == 'orth':
            # orthogonal distance per feature
            cross_mag = torch.abs(Zr * Pi - Zi * Pr)     # (B,C,d)
            mag_z     = torch.sqrt(Zr.pow(2) + Zi.pow(2)) + 1e-8
            orth      = cross_mag / mag_z
            dot       = Zr * Pr + Zi * Pi
            mask      = dot < 0
            signed    = torch.where(mask, mag_z + (mag_z - orth), orth)
            # add angle difference term
            proto_mag = torch.sqrt(Pr.pow(2) + Pi.pow(2))
            D = (signed + torch.abs(mag_z - proto_mag)).sum(dim=2)
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        # 5) Compute logits and total loss
        logits   = -self.temperature * D               # (B, C)
        loss_cls = self.criterion(logits, labels)
        loss     = loss_cls + self.gmm_lambda * loss_softmse
        preds    = logits.argmax(dim=1)

        return loss, preds, attn
