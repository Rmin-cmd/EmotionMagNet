# Test comment
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import complextorch

class UnifiedLoss(nn.Module):
    def __init__(self, loss_type, num_classes, distance_metric='L1',
                 dist_features=128, temperature=1.0, gmm_lambda=0.01,
                 criterion=None):
        super(UnifiedLoss, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.distance_metric = distance_metric
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        if self.loss_type == 'label_encoding':
            self.temperature = float(temperature) # Store as float
            angles_deg = [105, 165, 135, 225, 0, 75, 15, 45, 315]
            # Ensure num_classes is used consistently
            if self.num_classes != len(angles_deg):
                print(f"Warning: num_classes ({self.num_classes}) for label_encoding does not match standard 9. Using evenly spaced angles.")
                angles_rad = [2 * np.pi * (i / self.num_classes) for i in range(self.num_classes)]
            else:
                angles_rad = [2 * np.pi * (deg / 360) for deg in angles_deg]
            # Register as buffer as it's fixed but needs to be on device
            self.register_buffer('label_en', torch.tensor([np.exp(1j * angle) for angle in angles_rad], dtype=torch.complex64))

        elif self.loss_type == 'prototype':
            if dist_features <= 0:
                raise ValueError("dist_features must be positive for prototype loss.")
            self.dist_features = dist_features
            # Prototypes: shape (2, 5, dist_features, num_classes) - 5 is a hardcoded dimension
            self.prototypes_param = nn.Parameter(torch.randn(2, 5, self.dist_features, self.num_classes))
            self.temp_param = nn.Parameter(torch.tensor(float(temperature)))
            self.log_sigma_param = nn.Parameter(torch.zeros(1, 5, self.num_classes))
            self.gmm_lambda = float(gmm_lambda)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, preds, labels):
        labels_long = labels.to(torch.long)
        current_device = preds.device

        if self.loss_type == 'label_encoding':
            # self.label_en is registered as a buffer, should be on the correct device
            # if the model is moved to a device. Ensure it for safety.
            _label_en = self.label_en.to(current_device)

            preds_complex = preds.complex if isinstance(preds, complextorch.CVTensor) else preds

            # Assuming model output for 'label_encoding' is (B) or (B,1) complex numbers.
            if preds_complex.ndim > 1 and preds_complex.shape[-1] == 1:
                preds_complex = preds_complex.squeeze(-1)

            if preds_complex.ndim > 1 :
                 raise ValueError(f"Predictions for label_encoding loss have unexpected shape: {preds_complex.shape}. Expected (B) or (B,1).")

            # preds_complex is (B), _label_en is (C)
            # Unsqueeze for broadcasting: preds (B,1), label_en (1,C) -> distances (B,C)
            preds_exp = preds_complex.unsqueeze(1)
            label_exp = _label_en.unsqueeze(0)

            if self.distance_metric == 'L1':
                distances = torch.abs(preds_exp - label_exp)
            elif self.distance_metric == 'L2':
                distances = torch.sqrt((preds_exp.real - label_exp.real)**2 +
                                       (preds_exp.imag - label_exp.imag)**2 + 1e-8) # Added epsilon for stability
            elif self.distance_metric == 'orth':
                pred_real_bc = preds_exp.real
                pred_imag_bc = preds_exp.imag
                label_real_bc = label_exp.real
                label_imag_bc = label_exp.imag

                P_numerator = torch.abs(pred_real_bc * label_imag_bc - pred_imag_bc * label_real_bc)
                P = P_numerator / (label_exp.abs() + 1e-8)

                dot_product = pred_real_bc * label_real_bc + pred_imag_bc * label_imag_bc
                aligned_mask = dot_product < 0

                final_term = torch.zeros_like(P)
                current_mag_preds = preds_exp.abs()

                final_term[aligned_mask] = current_mag_preds.expand_as(P)[aligned_mask] + \
                                           (current_mag_preds.expand_as(P)[aligned_mask] - P[aligned_mask])
                final_term[~aligned_mask] = P[~aligned_mask]

                distances = final_term + torch.abs(current_mag_preds - label_exp.abs())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric} for label_encoding")

            logits = -distances / self.temperature
            loss = self.criterion(logits, labels_long)
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            return loss, predicted_labels

        elif self.loss_type == 'prototype':
            # Parameters (prototypes_param, temp_param, log_sigma_param) are already on device
            # due to nn.Parameter and model.to(device) call.
            prototypes_cv = self.prototypes_param[0] + 1j * self.prototypes_param[1] # Shape (5, Features, Classes)

            preds_complex = preds.complex if isinstance(preds, complextorch.CVTensor) else preds
            # Expected preds_complex shape: (Batch, dist_features)
            if preds_complex.ndim != 2 or preds_complex.shape[1] != self.dist_features:
                raise ValueError(f"Predictions for prototype loss have shape {preds_complex.shape}, expected (Batch, {self.dist_features})")

            # Broadcasting: preds (B,1,F,1), protos (1,5,F,C) -> distances (B,5,F,C)
            _preds_exp = preds_complex.unsqueeze(1).unsqueeze(3)
            _prototypes_exp = prototypes_cv.unsqueeze(0) # Add batch dimension for broadcasting

            if self.distance_metric == 'L1':
                distances = torch.abs(_preds_exp - _prototypes_exp)
            elif self.distance_metric == 'L2':
                distances = torch.sqrt((_preds_exp.real - _prototypes_exp.real)**2 +
                                       (_preds_exp.imag - _prototypes_exp.imag)**2 + 1e-8)
            elif self.distance_metric == 'orth':
                pred_real_b = _preds_exp.real
                pred_imag_b = _preds_exp.imag
                proto_real_b = _prototypes_exp.real
                proto_imag_b = _prototypes_exp.imag

                P_numerator_b = torch.abs(pred_real_b * proto_imag_b - pred_imag_b * proto_real_b)
                P_b = P_numerator_b / (_prototypes_exp.abs() + 1e-8)

                dot_product_b = pred_real_b * proto_real_b + pred_imag_b * proto_imag_b
                aligned_mask_b = dot_product_b < 0

                final_term_b = torch.zeros_like(P_b)
                current_mag_preds_b = _preds_exp.abs()

                final_term_b[aligned_mask_b] = current_mag_preds_b.expand_as(P_b)[aligned_mask_b] + \
                                               (current_mag_preds_b.expand_as(P_b)[aligned_mask_b] - P_b[aligned_mask_b])
                final_term_b[~aligned_mask_b] = P_b[~aligned_mask_b]
                distances = final_term_b + torch.abs(current_mag_preds_b - _prototypes_exp.abs())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric} for prototype loss")

            # distances is (B, 5, F, C), reduce over F (dim=2)
            logits = -self.temp_param * distances.mean(dim=2) # (B, 5, C)

            # Classification loss: mean logits over 5 views, then CE
            class_loss = self.criterion(logits.mean(dim=1), labels_long) # Input (B,C), Target (B)

            # GMM Regularization (simplified placeholder)
            loss_proto_gmm = torch.tensor(0.0, device=current_device)
            if self.gmm_lambda > 0:
                # Original GMM logic from loss_fucntion_2:
                # preds_complex is (B, F)
                # prototypes_cv is (5, F, C)
                # squared_diff_gmm: (B, 1, F, 1) vs (1, 5, F, C) -> (B, 5, F, C)
                squared_diff_gmm = torch.pow(torch.abs(preds_complex.unsqueeze(1).unsqueeze(3) - prototypes_cv.unsqueeze(0)), 2)

                distance2_gmm = squared_diff_gmm.sum(dim=2) # Sum over F: (B, 5, C)

                sigma_gmm = torch.exp(self.log_sigma_param.to(current_device)) # (1, 5, C)
                # likelihoods: (B, 5, C)
                likelihoods_gmm = torch.exp(-distance2_gmm / (2 * sigma_gmm**2 + 1e-8))

                # Normalize responsibilities per prototype set (over batch B)
                responsibilities_gmm = likelihoods_gmm / (likelihoods_gmm.sum(dim=0, keepdim=True) + 1e-8)

                # weighted_sum: (5, F, C)
                # einsum: 'bac,bf->afc' (b=batch, a=num_models_5, c=classes, f=features)
                weighted_sum_gmm = torch.einsum('bac,bf->afc', responsibilities_gmm, preds_complex)

                sum_resp_gmm = responsibilities_gmm.sum(dim=0) # (5, C)

                new_prototypes_gmm = weighted_sum_gmm / (sum_resp_gmm.unsqueeze(1) + 1e-8) # (5,F,C) / (5,1,C)

                loss_proto_gmm = torch.mean(torch.abs(new_prototypes_gmm - prototypes_cv)**2)

            total_loss = class_loss + self.gmm_lambda * loss_proto_gmm

            probabilities = torch.softmax(logits.mean(dim=1), dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            return total_loss, predicted_labels

        else:
            raise ValueError(f"Unknown loss_type in forward: {self.loss_type}")

# Old code (loss_function, loss_fucntion_2, label_encoding, and device variable) is removed.