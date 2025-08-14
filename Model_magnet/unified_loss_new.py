import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import complextorch
import complextorch.nn as compnn
from typing import Optional, Tuple


class UnifiedLoss(nn.Module):
    def __init__(self, loss_type, num_classes, distance_metric='L1',
                 dist_features=128, temperature=1.0, gmm_lambda=0.01,
                 criterion=None, num_heads=1):
        super(UnifiedLoss, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.distance_metric = distance_metric
        self.criterion = criterion or nn.CrossEntropyLoss()

        if self.loss_type == 'label_encoding':
            self.temperature = float(temperature)
            angles_deg = [105, 165, 135, 225, 0, 75, 15, 45, 315]
            if self.num_classes != len(angles_deg):
                angles_rad = [2 * np.pi * (i / self.num_classes)
                              for i in range(self.num_classes)]
            else:
                angles_rad = [2 * np.pi * (deg / 360) for deg in angles_deg]
            buf = torch.tensor([np.exp(1j * a) for a in angles_rad],
                               dtype=torch.complex64)
            self.register_buffer('label_en', buf)

        elif self.loss_type == 'prototype':
            if dist_features <= 0:
                raise ValueError("dist_features must be positive")
            self.dist_features = dist_features
            # prototypes_param: real+imag channels, shape (2,1,num_classes,dist_features)
            self.prototypes_param = nn.Parameter(
                torch.randn(2, 1, num_classes, dist_features)
            )
            self.temp_param = nn.Parameter(torch.tensor(float(temperature)))
            self.gmm_lambda = float(gmm_lambda)

        elif self.loss_type == 'prototype_attention':
            if dist_features <= 0:
                raise ValueError("dist_features must be positive")
            assert dist_features % num_heads == 0, "dist_features must be divisible by num_heads"

            self.C = num_classes
            self.d = dist_features
            self.h = num_heads
            self.head_dim = dist_features // num_heads
            self.scale = (self.head_dim) ** -0.5

            self.temperature = float(temperature)
            self.gmm_lambda = float(gmm_lambda)

            # prototypes parameter: real + imag
            self.prototypes_param = nn.Parameter(torch.randn(2, self.C, self.d))

            # real projectors (operate on last dim) used in pre()
            self.projector_rep = nn.Sequential(
                compnn.CVLinear(self.d, self.d),
                compnn.CVLayerNorm(self.d),
                compnn.CPReLU(),
                compnn.CVLinear(self.d, self.d),
            )
            self.projector_prototype = nn.Sequential(
                compnn.CVLinear(self.d, self.d),
                compnn.CVLayerNorm(self.d),
                compnn.CPReLU(),
                compnn.CVLinear(self.d, self.d),
            )

            # complex linear projections for q/k/v and output
            self.q_proj = compnn.CVLinear(self.d, self.d)
            self.k_proj = compnn.CVLinear(self.d, self.d)
            self.v_proj = compnn.CVLinear(self.d, self.d)
            self.out_proj = compnn.CVLinear(self.d, self.d)

        elif self.loss_type == 'simple':
            pass

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    @staticmethod
    def complex_normalize(cv: complextorch.CVTensor):
        r, i = cv.real, cv.imag
        mag = torch.sqrt(r.pow(2) + i.pow(2) + 1e-8)
        return complextorch.CVTensor(r / mag, i / mag)

    def pre(self, band_real, band_imag, proto_real, proto_imag):
        """Preprocessing for prototype_attention loss type"""
        # don't L2-normalize bands — preserve magnitude differences
        brn = band_real
        if band_imag is not None:
            if band_imag.shape != band_real.shape:
                raise ValueError("band_imag must have same shape as band_real")
            bin_ = band_imag
        else:
            bin_ = torch.zeros_like(brn)

        band_cv_in = complextorch.CVTensor(brn, bin_)
        band_cv_out = self.projector_rep(band_cv_in)

        # residual to prevent collapse
        if not hasattr(self, "proj_alpha"):
            self.proj_alpha = nn.Parameter(torch.tensor(0.5))
        alpha = torch.sigmoid(self.proj_alpha)
        brp = alpha * band_cv_out.real + (1 - alpha) * brn
        bip = alpha * band_cv_out.imag + (1 - alpha) * bin_

        # prototypes: normalize less aggressively or not at all
        prp = proto_real
        pip = proto_imag

        return brp, bip, prp, pip

    def prototype_query_attention(self, band_feats: complextorch.CVTensor, P_cv: complextorch.CVTensor) -> Tuple[
        complextorch.CVTensor, torch.Tensor]:
        """
        Multi-head attention for prototype_attention loss type
        band_feats: CVTensor (.real/.imag) shape (B, Bn, D)
        P_cv: CVTensor of prototypes (.real/.imag) shape (C, D)
        Returns: attended CVTensor (B, C, D), attn (B, C, Bn)
        """
        dev = band_feats.real.device
        B, Bn, d = band_feats.real.shape
        assert d == self.d

        # complex projections
        q = self.q_proj(P_cv)  # (C, d)
        k = self.k_proj(band_feats)  # (B, Bn, d)
        v = self.v_proj(band_feats)  # (B, Bn, d)

        # split into heads
        q_r = q.real.view(self.C, self.h, self.head_dim)
        q_i = q.imag.view(self.C, self.h, self.head_dim)

        k_r = k.real.view(B, Bn, self.h, self.head_dim)
        k_i = k.imag.view(B, Bn, self.h, self.head_dim)

        v_r = v.real.view(B, Bn, self.h, self.head_dim)
        v_i = v.imag.view(B, Bn, self.h, self.head_dim)

        # Hermitian per-head scores via einsum and permutation
        head_scores_real = torch.einsum("chd,bnhd->bnch", q_r, k_r)
        head_scores_imag = torch.einsum("chd,bnhd->bnch", q_i, k_i)
        head_scores = head_scores_real + head_scores_imag  # (B, Bn, C, h)
        head_scores = head_scores.permute(0, 2, 1, 3)  # (B, C, Bn, h)

        # sum heads scaled
        scores = head_scores.sum(-1) * self.scale  # (B, C, Bn)

        s_mean = scores.mean(dim=2, keepdim=True)
        s_std = scores.std(dim=2, keepdim=True) + 1e-6
        scores_norm = (scores - s_mean) / s_std  # now mean=0, std=1 per sample/class

        # apply gain (learnable)
        if not hasattr(self, "attn_gain"):
            self.attn_gain = nn.Parameter(torch.tensor(1.0))
        scores_scaled = scores_norm * F.softplus(self.attn_gain)

        attn = F.softmax(scores_scaled, dim=2)  # (B, C, Bn)

        # weighted sum of v over bands
        v_r_full = v_r.contiguous().view(B, Bn, self.d)
        v_i_full = v_i.contiguous().view(B, Bn, self.d)
        hat_r = torch.einsum("bcn,bnd->bcd", attn, v_r_full)  # (B, C, d)
        hat_i = torch.einsum("bcn,bnd->bcd", attn, v_i_full)  # (B, C, d)

        return complextorch.CVTensor(hat_r, hat_i), attn

    def forward(self, preds, labels, gmm_lambda=None):
        labels = labels.long()
        device = preds.device

        if self.loss_type == 'label_encoding':
            _label_en = self.label_en.to(device)
            preds_c = preds.complex if isinstance(preds, complextorch.CVTensor) else preds
            if preds_c.ndim > 2 and preds_c.shape[-1] == 1:
                preds_c = preds_c.squeeze(-1)
            preds_exp = preds_c.unsqueeze(1)  # [B,1,C]
            label_exp = _label_en.unsqueeze(0)  # [1,C]

            # classification loss: distance to each prototype
            if self.distance_metric == 'L1':
                dists = torch.abs(preds_exp - label_exp)
            elif self.distance_metric == 'L2':
                dists = torch.sqrt(
                    (preds_exp.unsqueeze(1).real - label_exp.unsqueeze(0).real) ** 2 +
                    (preds_exp.unsqueeze(1).imag - label_exp.unsqueeze(0).imag) ** 2 + 1e-8
                )
            elif self.distance_metric == 'orth':
                pred_real_b = preds_exp.real
                pred_imag_b = preds_exp.imag
                proto_real_b = label_exp.real
                proto_imag_b = label_exp.imag

                P_numerator_b = torch.abs(pred_real_b * proto_imag_b - pred_imag_b * proto_real_b)
                P_b = P_numerator_b / (preds_exp.abs() + 1e-8)

                dot_product_b = pred_real_b * proto_real_b + pred_imag_b * proto_imag_b
                aligned_mask_b = dot_product_b < 0

                final_term_b = torch.zeros_like(P_b)
                current_mag_preds_b = preds_exp.abs()

                final_term_b[aligned_mask_b] = current_mag_preds_b.expand_as(P_b)[aligned_mask_b] + \
                                               (current_mag_preds_b.expand_as(P_b)[aligned_mask_b] - P_b[
                                                   aligned_mask_b])
                final_term_b[~aligned_mask_b] = P_b[~aligned_mask_b]
                dists = final_term_b + torch.abs(current_mag_preds_b - label_exp.abs())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric} for label_encoding loss")

            logits = -dists.squeeze() / self.temperature
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds_lbl = torch.argmax(probs, dim=1)
            return loss, preds_lbl

        elif self.loss_type == 'prototype':
            # build complex prototypes [C, F]
            real = self.prototypes_param[0, 0]  # [C, F]
            imag = self.prototypes_param[1, 0]  # [C, F]
            prototypes = real + 1j * imag  # [C, F]

            preds_c = preds.complex if isinstance(preds, complextorch.CVTensor) else preds
            B, F = preds_c.shape
            C = self.num_classes

            # squared distances [B,C,F]
            delta = (preds_c.unsqueeze(1) - prototypes.unsqueeze(0)).abs() ** 2
            # sum over features → [B,C]
            D2 = delta.sum(dim=2)

            # compute soft assignments r [B,C]
            t = 1.0
            r = torch.softmax(-D2 / t, dim=1)  # [B,C]

            # soft-MSE regularizer
            loss_softmse = (r * D2).sum(dim=1).mean()

            # classification loss: distance to each prototype
            if self.distance_metric == 'L1':
                dists = torch.abs(preds_c.unsqueeze(1) - prototypes.unsqueeze(0))
            elif self.distance_metric == 'L2':
                dists = torch.sqrt(
                    (preds_c.unsqueeze(1).real - prototypes.unsqueeze(0).real) ** 2 +
                    (preds_c.unsqueeze(1).imag - prototypes.unsqueeze(0).imag) ** 2 + 1e-8
                )
            elif self.distance_metric == 'orth':
                preds_c_u, prototypes_u = preds_c.unsqueeze(1), prototypes.unsqueeze(0)
                pred_real_b = preds_c_u.real
                pred_imag_b = preds_c_u.imag
                proto_real_b = prototypes_u.real
                proto_imag_b = prototypes_u.imag

                P_numerator_b = torch.abs(pred_real_b * proto_imag_b - pred_imag_b * proto_real_b)
                P_b = P_numerator_b / (prototypes_u.abs() + 1e-8)

                dot_product_b = pred_real_b * proto_real_b + pred_imag_b * proto_imag_b
                aligned_mask_b = dot_product_b < 0

                final_term_b = torch.zeros_like(P_b)
                current_mag_preds_b = preds_c_u.abs()

                final_term_b[aligned_mask_b] = current_mag_preds_b.expand_as(P_b)[aligned_mask_b] + \
                                               (current_mag_preds_b.expand_as(P_b)[aligned_mask_b] - P_b[
                                                   aligned_mask_b])
                final_term_b[~aligned_mask_b] = P_b[~aligned_mask_b]
                dists = final_term_b + torch.abs(current_mag_preds_b - prototypes_u.abs())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric} for prototype loss")

            logits = -self.temp_param * dists.mean(dim=2)  # [B,C]
            class_loss = self.criterion(logits, labels)

            # combine
            λ = gmm_lambda if gmm_lambda is not None else self.gmm_lambda
            total_loss = class_loss + λ * loss_softmse

            # predictions
            probs = torch.softmax(logits, dim=1)
            preds_lbl = torch.argmax(probs, dim=1)
            return total_loss, preds_lbl

        elif self.loss_type == 'prototype_attention':
            """
            Integrated PrototypeAttentionLoss functionality
            preds: CVTensor (B, Bn, D) OR real Tensor (B, Bn, D) - band features
            labels: LongTensor (B,)
            """
            # convert to complex if real provided
            if isinstance(preds, torch.Tensor):
                band_feats = complextorch.CVTensor(preds, torch.zeros_like(preds))
            else:
                band_feats = preds

            br = band_feats.real
            bi = band_feats.imag

            dev = br.device
            labels = labels.to(dev).long()
            B, Bn, d = br.shape

            # reconstruct prototypes on device
            proto = self.prototypes_param.to(dev)  # (2, C, d)
            proto_r = proto[0]  # (C, d)
            proto_i = proto[1]  # (C, d)

            # build projected complex band_feats and prototypes
            band_feats_proj = complextorch.CVTensor(br, bi)
            proto_cv_proj = complextorch.CVTensor(proto_r, proto_i)

            # 1) Cross-attention: prototypes query bands
            Z, attn = self.prototype_query_attention(band_feats_proj, proto_cv_proj)  # CVTensor (B, C, d)
            Zr, Zi = Z.real, Z.imag  # each (B, C, d)

            # 2) Use prototypes for loss calculation
            Pr = proto_r.unsqueeze(0)  # (1, C, d)
            Pi = proto_i.unsqueeze(0)  # (1, C, d)

            # 3) Soft-MSE regularizer: uses squared distances
            dr = Zr - Pr
            di = Zi - Pi
            D2 = (dr.pow(2) + di.pow(2)).sum(dim=2)  # (B, C)
            r_soft = F.softmax(-D2 / self.temperature, dim=1)
            loss_softmse = (r_soft * D2).sum(dim=1).mean()

            # 4) Classification distances
            if self.distance_metric == "L1":
                D = (dr.abs() + di.abs()).sum(dim=2)
            elif self.distance_metric == "L2":
                D = torch.sqrt(D2 + 1e-8)
            elif self.distance_metric == "orth":
                cross_mag = torch.abs(Zr * Pi - Zi * Pr)
                mag_z = torch.sqrt(Zr.pow(2) + Zi.pow(2)) + 1e-8
                orth = cross_mag / mag_z
                dot = Zr * Pr + Zi * Pi
                mask = dot < 0
                signed = torch.where(mask, mag_z + (mag_z - orth), orth)
                proto_mag = torch.sqrt(Pr.pow(2) + Pi.pow(2))
                D = (signed + torch.abs(mag_z - proto_mag)).sum(dim=2)
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

            # 5) Compute logits and total loss
            logits = -self.temperature * D
            loss_cls = self.criterion(logits, labels)

            λ = gmm_lambda if gmm_lambda is not None else self.gmm_lambda
            loss = loss_cls + λ * loss_softmse
            preds_lbl = logits.argmax(dim=1)

            return loss, preds_lbl, attn  # Return attention weights for analysis

        elif self.loss_type == 'simple':
            if preds.is_complex():
                logits = preds.abs()
            else:
                logits = preds
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds_lbl = torch.argmax(probs, dim=1)
            return loss, preds_lbl

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")