import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import complextorch


class UnifiedLoss(nn.Module):
    def __init__(self, loss_type, num_classes, distance_metric='L1',
                 dist_features=128, temperature=1.0, gmm_lambda=0.01,
                 criterion=None):
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

        elif self.loss_type == 'simple':
            pass

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

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
            if self.distance_metric == 'L1':
                d = torch.abs(preds_exp - label_exp)
            else:  # L2
                d = torch.sqrt((preds_exp.real - label_exp.real)**2 +
                               (preds_exp.imag - label_exp.imag)**2 + 1e-8)
            logits = -d.squeeze() / self.temperature
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds_lbl = torch.argmax(probs, dim=1)
            return loss, preds_lbl

        elif self.loss_type == 'prototype':
            # build complex prototypes [C, F]
            real = self.prototypes_param[0,0]  # [C, F]
            imag = self.prototypes_param[1,0]  # [C, F]
            prototypes = real + 1j * imag      # [C, F]

            preds_c = preds.complex if isinstance(preds, complextorch.CVTensor) else preds
            B, F = preds_c.shape
            C = self.num_classes

            # squared distances [B,C,F]
            delta = (preds_c.unsqueeze(1) - prototypes.unsqueeze(0)).abs()**2
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
            elif self.distance_metric == 'L2':  # L2
                dists = torch.sqrt(
                    (preds_c.unsqueeze(1).real - prototypes.unsqueeze(0).real)**2 +
                    (preds_c.unsqueeze(1).imag - prototypes.unsqueeze(0).imag)**2 + 1e-8
                )  # [B,C,F]
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

        elif self.loss_type == 'simple':
            logits = preds.abs()
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds_lbl = torch.argmax(probs, dim=1)
            return loss, preds_lbl

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
