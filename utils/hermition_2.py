import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix
import torch


def cheb_poly(A, K):
    """
    Computes Chebyshev polynomials of a batch of graph Laplacians.
    A: (B, N, N)
    K: int
    Returns: (B, K+1, N, N)
    """
    K += 1
    B, N, _ = A.shape
    multi_order_laplacian = torch.zeros([B, K, N, N], dtype=torch.complex64, device=A.device)

    eye_batch = torch.eye(N, dtype=torch.complex64, device=A.device).unsqueeze(0).expand(B, -1, -1)
    multi_order_laplacian[:, 0, :, :] = eye_batch

    if K == 1:
        return multi_order_laplacian

    multi_order_laplacian[:, 1, :, :] = A
    if K > 2:
        for k in range(2, K):
            # A is (B, N, N), multi_order_laplacian[:, k-1, :, :] is (B, N, N)
            term1 = torch.matmul(A, multi_order_laplacian[:, k - 1, :, :])
            multi_order_laplacian[:, k, :, :] = 2 * term1 - multi_order_laplacian[:, k - 2, :, :]

    return multi_order_laplacian


def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
    """
    Computes the graph Laplacian for a batch of adjacency matrices.
    A: (B, N, N)
    Returns: (B, N, N)
    """
    if A.dim() == 2:  # Handle non-batch case for backward compatibility
        A = A.unsqueeze(0)

    B, N, _ = A.shape
    A = 1.0 * A
    if gcn_appr:
        eye_batch = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        A = A + 1.0 * eye_batch

    A_sym = 0.5 * (A + A.transpose(1, 2))

    if norm:
        d = torch.sum(A_sym, dim=2)
        d[d == 0] = 1
        d = torch.pow(d, -0.5)
        D = torch.diag_embed(d)
        A_sym = torch.matmul(torch.matmul(D, A_sym), D)

    if laplacian:
        Theta = 2 * torch.pi * q * 1j * (A - A.transpose(1, 2))
        if norm:
            D = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        else:
            d = torch.sum(A_sym, dim=2)
            D = torch.diag_embed(d)
        L = D - torch.exp(Theta) * A_sym
    else:
        # This part was commented out, keeping it that way.
        # It's also not vectorized.
        raise NotImplementedError("Non-laplacian decomposition is not implemented for batches.")

    if norm:
        eye_batch = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        L = (2.0 / max_eigen) * L - eye_batch

    return L.squeeze(0) if B == 1 and L.dim() > 2 else L
