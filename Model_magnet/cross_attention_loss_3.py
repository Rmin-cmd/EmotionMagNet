# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import complextorch
# import complextorch.nn as compnn
# from typing import Optional, Tuple
#
#
# class PrototypeAttentionLoss(nn.Module):
#     """
#     Prototype -> bands multi-head cross-attention + loss, with pre() applied in forward.
#     - Inputs (band_feats): CVTensor with .real/.imag shapes (B, Bn, D) OR real Tensor (B, Bn, D)
#     - pre() will normalize & project band real/imag and prototypes before attention.
#     """
#
#     def __init__(
#             self,
#             num_classes: int,
#             dist_features: int,
#             num_heads: int = 1,
#             distance_metric: str = "orth",
#             temperature: float = 1.0,
#             gmm_lambda: float = 0.01,
#             criterion: Optional[nn.Module] = None,
#     ):
#         super().__init__()
#         assert dist_features % num_heads == 0, "dim must be divisible by num_heads"
#         self.C = num_classes
#         self.d = dist_features
#         self.h = num_heads
#         self.head_dim = dist_features // num_heads
#         self.scale = (self.head_dim) ** -0.5
#
#         self.distance_metric = distance_metric
#         self.temperature = float(temperature)
#         self.gmm_lambda = float(gmm_lambda)
#         self.criterion = criterion or nn.CrossEntropyLoss()
#
#         # prototypes parameter: real + imag
#         self.prototypes_param = nn.Parameter(torch.randn(2, self.C, self.d))
#
#         # real projectors (operate on last dim) used in pre()
#         self.projector_rep = nn.Sequential(
#             compnn.CVLinear(self.d, self.d),
#             compnn.CVLayerNorm(self.d),
#             compnn.CPReLU(),
#             compnn.CVLinear(self.d, self.d),
#         )
#         self.projector_prototype = nn.Identity()
#         # complex linear projections for q/k/v and output
#         self.q_proj = compnn.CVLinear(self.d, self.d)
#         self.k_proj = compnn.CVLinear(self.d, self.d)
#         self.v_proj = compnn.CVLinear(self.d, self.d)
#         self.out_proj = compnn.CVLinear(self.d, self.d)
#
#     @staticmethod
#     def complex_normalize(cv: complextorch.CVTensor):
#         r, i = cv.real, cv.imag
#         mag = torch.sqrt(r.pow(2) + i.pow(2) + 1e-8)
#         return complextorch.CVTensor(r / mag, i / mag)
#
#     # -------------------------
#     # pre: normalize & project real tensors (band real/imag and prototypes)
#     # -------------------------
#     def pre(
#             self,
#             band_real: torch.Tensor,
#             band_imag: Optional[torch.Tensor],
#             proto_real: torch.Tensor,
#             proto_imag: torch.Tensor,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
#         """
#         Normalize (L2 on last dim) and project band_real, band_imag (if provided),
#         and prototype real/imag.
#
#         Shapes:
#           - band_real: (B, Bn, D) or (B, D)
#           - band_imag: same shape as band_real or None
#           - proto_real: (C, D)
#           - proto_imag: (C, D)
#
#         Returns projected (band_real_proj, band_imag_proj_or_None, proto_real_proj, proto_imag_proj)
#         """
#         # project band real
#         if band_real.dim() == 3:
#             brn = F.normalize(band_real, p=2, dim=-1)  # (B, Bn, D)
#             brp = self.projector_rep(brn)  # (B, Bn, D)
#         elif band_real.dim() == 2:
#             brn = F.normalize(band_real, p=2, dim=-1)  # (B, D)
#             brp = self.projector_rep(brn)  # (B, D)
#         else:
#             raise ValueError("band_real must have dim 2 or 3")
#
#         # project band imag if present
#         if band_imag is not None:
#             if band_imag.shape != band_real.shape:
#                 raise ValueError("band_imag must have same shape as band_real")
#             bin_ = F.normalize(band_imag, p=2, dim=-1)
#             bip = self.projector_rep(bin_)
#         else:
#             bip = None
#
#         # prototypes: (C, D)
#         prn = F.normalize(proto_real, p=2, dim=-1)
#         prp = self.projector_prototype(prn)
#         pin = F.normalize(proto_imag, p=2, dim=-1)
#         pip = self.projector_prototype(pin)
#
#         return brp, bip, prp, pip
#
#     # -------------------------
#     # prototype -> bands multi-head attention (complex)
#     # -------------------------
#     def prototype_query_attention(self, band_feats: complextorch.CVTensor, P_cv: complextorch.CVTensor) -> Tuple[
#         complextorch.CVTensor, torch.Tensor]:
#         """
#         band_feats: CVTensor (.real/.imag) shape (B, Bn, D)
#         P_cv: CVTensor of prototypes (.real/.imag) shape (C, D)
#         Returns: attended CVTensor (B, C, D), attn (B, C, Bn)
#         """
#         dev = band_feats.real.device
#         B, Bn, d = band_feats.real.shape
#         assert d == self.d
#
#         # complex projections
#         q = self.q_proj(P_cv)  # (C, d)
#         k = self.k_proj(band_feats.complex)  # (B, Bn, d)
#         v = self.v_proj(band_feats.complex)  # (B, Bn, d)
#
#         # split into heads
#         q_r = q.real.view(self.C, self.h, self.head_dim)
#         q_i = q.imag.view(self.C, self.h, self.head_dim)
#
#         k_r = k.real.view(B, Bn, self.h, self.head_dim)
#         k_i = k.imag.view(B, Bn, self.h, self.head_dim)
#
#         v_r = v.real.view(B, Bn, self.h, self.head_dim)
#         v_i = v.imag.view(B, Bn, self.h, self.head_dim)
#
#         # Hermitian per-head scores via einsum and permutation
#         head_scores_real = torch.einsum("chd,bnhd->bnch", q_r, k_r)
#         head_scores_imag = torch.einsum("chd,bnhd->bnch", q_i, k_i)
#         head_scores = head_scores_real + head_scores_imag  # (B, Bn, C, h)
#         head_scores = head_scores.permute(0, 2, 1, 3)  # (B, C, Bn, h)
#
#         # sum heads scaled
#         scores = head_scores.sum(-1) * self.scale  # (B, C, Bn)
#
#         attn = F.softmax(scores, dim=2)  # (B, C, Bn)
#
#         # weighted sum of v over bands
#         v_r_full = v_r.contiguous().view(B, Bn, self.d)
#         v_i_full = v_i.contiguous().view(B, Bn, self.d)
#         hat_r = torch.einsum("bcn,bnd->bcd", attn, v_r_full)  # (B, C, d)
#         hat_i = torch.einsum("bcn,bnd->bcd", attn, v_i_full)  # (B, C, d)
#
#         # out projection (complex)
#         hat_flat = complextorch.CVTensor(hat_r.view(B * self.C, d), hat_i.view(B * self.C, d))
#         hat_proj = self.out_proj(hat_flat)
#         hr = hat_proj.real.view(B, self.C, d)
#         hi = hat_proj.imag.view(B, self.C, d)
#
#         return complextorch.CVTensor(hr, hi), attn
#
#     # -------------------------
#     # forward: integrate pre() into pipeline
#     # -------------------------
#     def forward(self, band_feats_in, labels: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         band_feats_in: CVTensor (B, Bn, D) OR real Tensor (B, Bn, D)
#         labels: LongTensor (B,)
#         """
#         # convert to complex if real provided
#         if isinstance(band_feats_in, torch.Tensor):
#             band_feats = complextorch.CVTensor(band_feats_in, torch.zeros_like(band_feats_in))
#         else:
#             band_feats = band_feats_in
#
#         br = band_feats.real
#         bi = band_feats.imag
#
#         dev = br.device
#         labels = labels.to(dev).long()
#         B, Bn, d = br.shape
#
#         # reconstruct prototypes on device
#         proto = self.prototypes_param.to(dev)  # (2, C, d)
#         proto_r = proto[0]  # (C, d)
#         proto_i = proto[1]  # (C, d)
#
#         # === APPLY pre() HERE ===
#         # project & normalize band_real, band_imag, proto_real, proto_imag
#         br_p, bi_p, pr_p, pi_p = self.pre(br, bi, proto_r, proto_i)
#         # br_p/bi_p shapes: (B, Bn, D); pr_p/pi_p shapes: (C, D)
#
#         # build projected complex band_feats and prototypes
#         band_feats_proj = complextorch.CVTensor(br_p, bi_p)
#         proto_cv_proj = complextorch.CVTensor(pr_p, pi_p)
#
#         # 1) Cross-attention: prototypes query bands (using projected features)
#         Z, attn = self.prototype_query_attention(band_feats_proj, proto_cv_proj)  # CVTensor (B, C, d)
#         Zr, Zi = Z.real, Z.imag  # each (B, C, d)
#
#         # 2) Use PROJECTED prototypes for loss calculation
#         Pr = pr_p.unsqueeze(0)  # (1, C, d)
#         Pi = pi_p.unsqueeze(0)  # (1, C, d)
#
#         # 3) Soft-MSE regularizer: uses squared distances
#         dr = Zr - Pr
#         di = Zi - Pi
#         D2 = (dr.pow(2) + di.pow(2)).sum(dim=2)  # (B, C)
#         r_soft = F.softmax(-D2 / self.temperature, dim=1)
#         loss_softmse = (r_soft * D2).sum(dim=1).mean()
#
#         # 4) Classification distances
#         if self.distance_metric == "L1":
#             D = (dr.abs() + di.abs()).sum(dim=2)
#         elif self.distance_metric == "L2":
#             D = torch.sqrt(D2 + 1e-8)
#         elif self.distance_metric == "orth":
#             cross_mag = torch.abs(Zr * Pi - Zi * Pr)
#             mag_z = torch.sqrt(Zr.pow(2) + Zi.pow(2)) + 1e-8
#             orth = cross_mag / mag_z
#             dot = Zr * Pr + Zi * Pi
#             mask = dot < 0
#             signed = torch.where(mask, mag_z + (mag_z - orth), orth)
#             proto_mag = torch.sqrt(Pr.pow(2) + Pi.pow(2))
#             D = (signed + torch.abs(mag_z - proto_mag)).sum(dim=2)
#         else:
#             raise ValueError(f"Unknown distance_metric: {self.distance_metric}")
#
#         # 5) Compute logits and total loss
#         logits = -self.temperature * D
#         loss_cls = self.criterion(logits, labels)
#         loss = loss_cls + self.gmm_lambda * loss_softmse
#         preds = logits.argmax(dim=1)
#
#         return loss, preds, attn
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import complextorch
import complextorch.nn as compnn
from typing import Optional, Tuple


class PrototypeAttentionLoss(nn.Module):
    """
    Prototype -> bands multi-head cross-attention + loss, with pre() applied in forward.
    - Inputs (band_feats): CVTensor with .real/.imag shapes (B, Bn, D) OR real Tensor (B, Bn, D)
    - pre() will normalize & project band real/imag and prototypes before attention.
    """

    def __init__(
            self,
            num_classes: int,
            dist_features: int,
            num_heads: int = 1,
            distance_metric: str = "orth",
            temperature: float = 1.0,
            gmm_lambda: float = 0.01,
            criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dist_features % num_heads == 0, "dim must be divisible by num_heads"
        self.C = num_classes
        self.d = dist_features
        self.h = num_heads
        self.head_dim = dist_features // num_heads
        self.scale = (self.head_dim) ** -0.5

        self.distance_metric = distance_metric
        self.temperature = float(temperature)
        self.gmm_lambda = float(gmm_lambda)
        self.criterion = criterion or nn.CrossEntropyLoss()

        # prototypes parameter: real + imag
        self.prototypes_param = nn.Parameter(torch.randn(2, self.C, self.d))

        # real projectors (operate on last dim) used in pre()
        self.projector_rep = nn.Sequential(
            compnn.CVLinear(self.d, self.d),
            compnn.CVLayerNorm(self.d),
            compnn.CPReLU(),
            compnn.CVLinear(self.d, self.d),
        )
        # self.projector_prototype = nn.Identity()
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

    @staticmethod
    def complex_normalize(cv: complextorch.CVTensor):
        r, i = cv.real, cv.imag
        mag = torch.sqrt(r.pow(2) + i.pow(2) + 1e-8)
        return complextorch.CVTensor(r / mag, i / mag)

    # -------------------------
    # pre: normalize & project real tensors (band real/imag and prototypes)
    # -------------------------
    # def pre(
    #         self,
    #         band_real: torch.Tensor,
    #         band_imag: Optional[torch.Tensor],
    #         proto_real: torch.Tensor,
    #         proto_imag: torch.Tensor,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    #     """
    #     Normalize (L2 on last dim) and project band_real, band_imag (if provided),
    #     and prototype real/imag.
    #
    #     Shapes:
    #       - band_real: (B, Bn, D) or (B, D)
    #       - band_imag: same shape as band_real or None
    #       - proto_real: (C, D)
    #       - proto_imag: (C, D)
    #
    #     Returns projected (band_real_proj, band_imag_proj_or_None, proto_real_proj, proto_imag_proj)
    #     """
    #     # Normalize band features
    #     brn = F.normalize(band_real, p=2, dim=-1)
    #
    #     if band_imag is not None:
    #         if band_imag.shape != band_real.shape:
    #             raise ValueError("band_imag must have same shape as band_real")
    #         bin_ = F.normalize(band_imag, p=2, dim=-1)
    #     else:
    #         bin_ = torch.zeros_like(brn)
    #
    #     # Create complex tensor and project
    #     band_cv_in = complextorch.CVTensor(brn, bin_)
    #     band_cv_out = self.projector_rep(band_cv_in)
    #     brp = band_cv_out.real
    #     bip = band_cv_out.imag
    #
    #     # Normalize and project prototypes (projector is Identity)
    #     pr_cv_in = complextorch.CVTensor(proto_real, proto_imag)
    #     pr_cv_in = self.complex_normalize(pr_cv_in)
    #     pr_cv_out = self.projector_prototype(pr_cv_in)
    #     prp, pip = pr_cv_out.real, pr_cv_out.imag
    #     # prn = F.normalize(proto_real, p=2, dim=-1)
    #     # prp = self.projector_prototype(prn)
    #     # pin = F.normalize(proto_imag, p=2, dim=-1)
    #     # pip = self.projector_prototype(pin)
    #
    #     return brp, bip, prp, pip

    def pre(self, band_real, band_imag, proto_real, proto_imag):
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
        prp = proto_real  # or F.normalize(proto_real, p=2, dim=-1) if you want
        pip = proto_imag

        return brp, bip, prp, pip

    # -------------------------
    # prototype -> bands multi-head attention (complex)
    # -------------------------
    def prototype_query_attention(self, band_feats: complextorch.CVTensor, P_cv: complextorch.CVTensor) -> Tuple[
        complextorch.CVTensor, torch.Tensor]:
        """
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
            self.attn_gain = nn.Parameter(torch.tensor(1.0))  # put in __init__ ideally
        scores_scaled = scores_norm * F.softplus(self.attn_gain)

        attn = F.softmax(scores_scaled, dim=2)  # (B, C, Bn)

        # weighted sum of v over bands
        v_r_full = v_r.contiguous().view(B, Bn, self.d)
        v_i_full = v_i.contiguous().view(B, Bn, self.d)
        hat_r = torch.einsum("bcn,bnd->bcd", attn, v_r_full)  # (B, C, d)
        hat_i = torch.einsum("bcn,bnd->bcd", attn, v_i_full)  # (B, C, d)

        # out projection (complex)
        # hat_flat = complextorch.CVTensor(hat_r.view(B * self.C, d), hat_i.view(B * self.C, d))
        # hat_proj = self.out_proj(hat_flat)
        # hr = hat_proj.real.view(B, self.C, d)
        # hi = hat_proj.imag.view(B, self.C, d)

        return complextorch.CVTensor(hat_r, hat_i), attn

    # -------------------------
    # forward: integrate pre() into pipeline
    # -------------------------
    def forward(self, band_feats_in, labels: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        band_feats_in: CVTensor (B, Bn, D) OR real Tensor (B, Bn, D)
        labels: LongTensor (B,)
        """
        # convert to complex if real provided
        if isinstance(band_feats_in, torch.Tensor):
            band_feats = complextorch.CVTensor(band_feats_in, torch.zeros_like(band_feats_in))
        else:
            band_feats = band_feats_in

        br = band_feats.real
        bi = band_feats.imag

        dev = br.device
        labels = labels.to(dev).long()
        B, Bn, d = br.shape

        # reconstruct prototypes on device
        proto = self.prototypes_param.to(dev)  # (2, C, d)
        proto_r = proto[0]  # (C, d)
        proto_i = proto[1]  # (C, d)

        # === APPLY pre() HERE ===
        # project & normalize band_real, band_imag, proto_real, proto_imag
        # br_p, bi_p, pr_p, pi_p = self.pre(br, bi, proto_r, proto_i)
        # br_p/bi_p shapes: (B, Bn, D); pr_p/pi_p shapes: (C, D)

        # build projected complex band_feats and prototypes
        band_feats_proj = complextorch.CVTensor(br, bi)
        proto_cv_proj = complextorch.CVTensor(proto_r, proto_i)

        # 1) Cross-attention: prototypes query bands (using projected features)
        Z, attn = self.prototype_query_attention(band_feats_proj, proto_cv_proj)  # CVTensor (B, C, d)
        Zr, Zi = Z.real, Z.imag  # each (B, C, d)

        # Diagnostic: attention entropy (per-example, per-class -> mean printed)
        # attn shape: (B, C, Bn)
        # with torch.no_grad():
        #     attn_clamped = attn.clamp(min=1e-12)
        #     ent = -(attn_clamped * attn_clamped.log()).sum(dim=2)
        #     ent_mean = ent.mean().item()
        #     print(
        #         f"[PrototypeAttentionLoss] attn entropy mean: {ent_mean:.4f} (log(Bn) ~ {torch.log(torch.tensor(Bn)).item():.4f})")
        # # 2) Use PROJECTED prototypes for loss calculation
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
        loss = loss_cls + self.gmm_lambda * loss_softmse
        preds = logits.argmax(dim=1)

        return loss, preds, attn




