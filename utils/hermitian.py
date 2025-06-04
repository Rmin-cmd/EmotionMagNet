import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix
import torch
###########################################
####### Dense implementation ##############
###########################################


def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    # multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = torch.zeros([K, N, N], dtype=torch.complex64, device=A.device)  # [K, N, N]
    # multi_order_laplacian[0] += np.eye(N, dtype=np.float32)
    multi_order_laplacian[0] += torch.eye(N, dtype=torch.complex64, device=A.device) # Match dtype

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] += A
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                # multi_order_laplacian[k] += 2 * np.dot(A, multi_order_laplacian[k-1]) - multi_order_laplacian[k-2]
                multi_order_laplacian[k] += 2 * torch.matmul(A, multi_order_laplacian[k-1]) - multi_order_laplacian[k-2]

    return multi_order_laplacian


def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
    # A = 1.0*np.array(A)
    A = 1.0*A
    if gcn_appr:
        # A += 1.0*np.eye(A.shape[0])
        A += 1.0*torch.eye(A.shape[0], device=A.device)

    A_sym = 0.5*(A + A.T) # symmetrized adjacency
    # A_sym = 0.7 * A + 0.3 * A.T

    if norm:
        # d = np.sum(np.array(A_sym), axis = 0)
        d = torch.sum(A_sym, dim = 0)
        d[d == 0] = 1
        # d = np.power(d, -0.5)
        d = torch.pow(d, -0.5)
        # D = np.diag(d)
        D = torch.diag(d)
        # A_sym = np.dot(np.dot(D, A_sym), D)
        A_sym = torch.matmul(torch.matmul(D, A_sym), D)

    if laplacian:
        # Theta = 2*np.pi*q*1j*(A - A.T) # phase angle array
        Theta = 2*torch.pi*q*1j*(A - A.T) # phase angle array
        if norm:
            # D = np.diag([1.0]*len(d))
            D = torch.eye(A.shape[0], device=A.device) # Assuming A.shape[0] is num_nodes, same as len(d)
        else:
            # d = np.sum(np.array(A_sym), axis = 0) # diag of degree array
            d = torch.sum(A_sym, dim = 0) # diag of degree array
            # D = np.diag(d)
            D = torch.diag(d)
        # L = D - np.exp(Theta)*A_sym
        L = D - torch.exp(Theta)*A_sym
    '''
    else:
        #transition matrix
        d_out = np.sum(np.array(A), axis = 1)
        d_out[d_out==0] = -1
        d_out = 1.0/d_out
        d_out[d_out<0] = 0
        D = np.diag(d_out)
        L = np.eye(len(d_out)) - np.dot(D, A)
    '''
    if norm:

        # L = (2.0/max_eigen)*L - np.diag([1.0]*len(A))
        L = (2.0/max_eigen)*L - torch.eye(A.shape[0], device=A.device)

    return L

# This function is unused in the current pipeline and may be incomplete.
# def hermitian_decomp(As, q = 0.25, norm = False, laplacian = True, max_eigen = None, gcn_appr = False):
#     ls, ws, vs = [], [], []
#     if len(As.shape)>2:
#         for i, A in enumerate(As):
#             l, w, v = decomp(A, q, norm, laplacian, max_eigen, gcn_appr)
#             vs.append(v)
#             ws.append(w)
#             ls.append(l)
#     else:
#         ls, ws, vs = decomp(As, q, norm, laplacian, max_eigen, gcn_appr)
#     return np.array(ls), np.array(ws), np.array(vs)
