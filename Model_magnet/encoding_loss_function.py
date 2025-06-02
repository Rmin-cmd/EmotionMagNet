import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import complextorch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def loss_function(loss_func, preds, labels, temperature=1, distance_metric = 'L1'):

    # angs = preds.angle()
    # abs_lbl = preds.abs()
    label_en = torch.tensor([np.exp(2j*np.pi*(105/360)), np.exp(2j*np.pi*(165/360)), np.exp(2j*np.pi*(135/360)),
               np.exp(2j*np.pi*(225/360)), np.exp(0j), np.exp(2j*np.pi*(75/360)), np.exp(2j*np.pi*(15/360)),
               np.exp(2j*np.pi*(45/360)), np.exp(2j*np.pi*(315/360))]).to(device=device)
    # label_ang = label_en.angle()
    # label_abs = label_en.abs()
    if distance_metric == 'L1':
        distances = torch.abs(preds.complex - label_en)
    elif distance_metric == 'L2':
        distances = torch.sqrt((preds.real - label_en.real)**2+(preds.imag - label_en.imag)**2)
    elif distance_metric == 'orth':
        mag_input = label_en.abs()
        mag_preds = preds.abs()
        angle = torch.atan2(label_en.imag, label_en.real) - torch.atan2(preds.imag, preds.real)
        P = torch.abs((preds.real * label_en.imag) - (preds.imag * label_en.real))/mag_input
        aligned_mask = (torch.cos(angle) < 0).bool()
        final_term = torch.zeros_like(P)
        final_term[aligned_mask] = mag_preds[aligned_mask] + (mag_preds[aligned_mask] - P[aligned_mask])
        final_term[~aligned_mask] = P[~aligned_mask]
        distances = final_term + torch.abs(mag_preds - mag_input)

    logits = - distances/temperature

    loss = loss_func(logits, labels.to(torch.int64))

    probabilities = torch.softmax(logits, dim=1)

    predicted_labels = torch.argmax(probabilities, dim=1)

    return loss, predicted_labels


class loss_fucntion_2(nn.Module):

    def __init__(self, distance_metric='L1', dist_features=128):
        super(loss_fucntion_2, self).__init__()
        self.distance_metric = distance_metric
        self.loss_function = nn.CrossEntropyLoss()
        # self.prototypes = torch.rand(2, dist_features, 9).to(device)
        self.prototypes = torch.rand(2, 5, dist_features, 9).to(device)
        self.prototypes = nn.Parameter(data=self.prototypes, requires_grad=True)
        self.temp = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        # self.log_sigma = nn.Parameter(data=torch.zeros([1, 9])).to(device)
        self.log_sigma = nn.Parameter(data=torch.zeros([1, 5, 9])).to(device)
        self.gmm_lambda = 0.01

    def forward(self, preds, labels):

        self.prototypes_cv = self.prototypes[0] + 1j * self.prototypes[1]

        if self.distance_metric == 'L1':
            distances = torch.abs(preds.complex - self.prototypes_cv)
        elif self.distance_metric == 'L2':
            distances = torch.sqrt((preds.real.unsqueeze(1) - self.prototypes_cv.real)**2+(preds.imag.unsqueeze(1) - self.prototypes_cv.imag)**2)
        elif self.distance_metric == 'orth':
            mag_input = self.prototypes_cv.abs()
            mag_preds = preds.abs()
            angle = torch.atan2(self.prototypes_cv.imag, self.prototypes_cv.real) - torch.atan2(preds.imag, preds.real)
            P = torch.abs((preds.real * self.prototypes_cv.imag) - (preds.imag * self.prototypes_cv.real))/mag_input
            aligned_mask = (torch.cos(angle) < 0).bool()
            final_term = torch.zeros_like(P)
            final_term[aligned_mask] = mag_preds[aligned_mask] + (mag_preds[aligned_mask] - P[aligned_mask])
            final_term[~aligned_mask] = P[~aligned_mask]
            distances = final_term + torch.abs(mag_preds - mag_input)

        # logits = - self.temp * distances.mean(dim=1)
        logits = - self.temp * distances.mean(dim=2)

        loss = self.loss_function(logits.mean(dim=1), labels.to(torch.int64))

        # ----------------- GMM-inspired Prototype Update -----------------
        # We use the Euclidean (squared) distance to define a Gaussian kernel.
        # Our aim: for each sample, compute a soft-assignment (responsibilities) to each prototype,
        # and then compute a weighted average of the sample features.
        #
        # First, compute the squared Euclidean distance between each sample and each prototype.
        # We already have: preds_complex shape: (batch, dist_features, 1)
        squared_diff = torch.pow(torch.abs(preds.complex.unsqueeze(1) - self.prototypes_cv.unsqueeze(0)), 2)  # (batch, dist_features, num_prototypes)
        # Sum over the feature dimension to get a scalar squared distance per prototype.
        bc, ppc, dist, plen = squared_diff.size()
        # distance2 = squared_diff.sum(dim=1)  # shape: (batch, num_prototypes)
        distance2 = squared_diff.sum(dim=2)

        # Compute the unnormalized "likelihood" using a Gaussian kernel:
        sigma = torch.exp(self.log_sigma)
        # likelihoods = torch.exp(-distance2 / (2 * sigma**2 + 1e-8))  # (batch, num_prototypes)
        likelihoods = torch.exp(-distance2.view(bc, ppc * plen) / (2 * sigma.view(sigma.shape[0], -1) ** 2 + 1e-8))
        # Normalize to get soft-assignment weights (responsibilities):
        # responsibilities = likelihoods / (likelihoods.sum(dim=1, keepdim=True) + 1e-8)  # shape: (batch, num_prototypes)
        responsibilities = likelihoods / (likelihoods.sum(dim=1, keepdim=True) + 1e-8)

        # Now, compute the new prototype estimates as a weighted average of the sample features.
        # preds.complex has shape (batch, dist_features).
        # We compute weighted sums along the batch dimension.
        # Use Einstein summation: for each prototype k, new_proto[:, k] = sum_{i} responsibilities[i,k] * preds.complex[i,:] / sum_{i} responsibilities[i,k]
        weighted_sum = torch.einsum('bk,bd->kd', responsibilities+1j*torch.zeros_like(responsibilities), preds.complex.squeeze())  # (num_prototypes, dist_features)
        sum_resp = responsibilities.sum(dim=0).unsqueeze(1)  # (num_prototypes, 1)
        new_prototypes = weighted_sum / (sum_resp + 1e-8)  # (num_prototypes, dist_features)
        # Transpose to match our prototype shape: (dist_features, num_prototypes)
        new_prototypes = new_prototypes.transpose(0, 1)
        new_prototypes = new_prototypes.reshape([ppc, dist, plen])

        # Define a prototype regularization loss that encourages current prototypes to be close to the updated values.
        loss_proto = torch.mean(torch.abs(new_prototypes - self.prototypes_cv)**2)

        # Total loss: classification loss plus the weighted GMM regularization term.
        loss_total = loss + self.gmm_lambda * loss_proto
        # loss_total = loss

        probabilities = torch.softmax(logits.mean(dim=1), dim=1)

        predicted_labels = torch.argmax(probabilities, dim=1)

        return loss_total, predicted_labels



def label_encoding(labels):

    label_en = torch.zeros_like(labels).to(torch.complex64)

    label_comp = [np.exp(2j*np.pi*(105/360)), np.exp(2j*np.pi*(165/360)), np.exp(2j*np.pi*(135/360)),
              np.exp(2j*np.pi*(225/360)), np.exp(0j), np.exp(2j*np.pi*(75/360)), np.exp(2j*np.pi*(15/360)),
              np.exp(2j*np.pi*(45/360)), np.exp(2j*np.pi*(315/360))]
    # label_comp = [np.exp(2j * np.pi * (i / 9)) for i in range(9)]

    for lbl in torch.unique(labels):

        label_en[labels == lbl] = label_comp[lbl]

    return label_en