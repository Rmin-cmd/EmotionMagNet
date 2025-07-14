import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import ChebConv, global_mean_pool, SAGPooling
# from utils.geometric_baselines import GCNModel, SAGEModel, GATModel, GIN_Model
# from utils.geometric_baselines_pooling import GCNModel, SAGEModel, GATModel, GINModel
from utils.geometric_baselines_new import GCNBatch
# from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import spectral_norm
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
# from torch_geometric.utils import add_self_loops, degree
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from utils import load_data
from utils.utils_loss import Metrics
from sklearn import svm, model_selection, metrics, preprocessing


def preprocess_pdc(raw_pdc,
                   clip_percentile: float = 99.9,
                   zero_diag: bool = True,
                   trials_per_subject: int = None):
    """
    Clean and reshape raw PDC data.

    :param raw_pdc: np.ndarray, shape (n_subs, n_clips, n_windows, N, N)
    :param clip_percentile: float in (0,100), upper percentile for clipping
    :param zero_diag: if True, force diagonal to zero in each matrix
    :param trials_per_subject: if provided, used for logging/intra-trial checks
    :return:
        pdc_clean: np.ndarray, shape (n_graphs, N, N)
    """
    # ---- 1. Basic sanity checks ----
    assert raw_pdc.ndim == 5, f"Expected 5D raw_pdc, got {raw_pdc.shape}"
    n_subs, n_clips, n_windows, N, _ = raw_pdc.shape
    n_graphs = n_subs * n_clips * n_windows

    # ---- 2. Zero out diagonals ----
    if zero_diag:
        for i in range(n_subs):
            for j in range(n_clips):
                for k in range(n_windows):
                    np.fill_diagonal(raw_pdc[i, j, k], 0.0)

    # ---- 3. Clip extreme outliers ----
    # Compute global upper bound from all off-diagonal entries
    all_vals = raw_pdc[:, :, :, ~np.eye(N, dtype=bool)].ravel()
    ub = np.percentile(all_vals, clip_percentile)
    raw_pdc = np.clip(raw_pdc, 0, ub)

    # ---- 4. Normalize each matrix to [0,1] ----
    # Avoid divide-by-zero by checking max > 0
    for i in range(n_subs):
        for j in range(n_clips):
            for k in range(n_windows):
                mat = raw_pdc[i, j, k]
                m = mat.max()
                if m > 0:
                    raw_pdc[i, j, k] = mat / m

    # ---- 5. Remove any NaN/Inf (shouldn’t be any after clipping) ----
    assert np.isfinite(raw_pdc).all(), "Non-finite values detected after clipping/normalization!"

    # ---- 6. Reshape to [n_graphs, N, N] ----
    pdc_clean = raw_pdc.reshape(n_graphs, N, N)

    # # Optional: report intra‑subject variability
    # if trials_per_subject is not None:
    #     flat = pdc_clean.reshape(n_graphs, -1)
    #     n_subjects = n_graphs // trials_per_subject
    #     for s in range(n_subjects):
    #         block = flat[s * trials_per_subject:(s + 1) * trials_per_subject]
    #         var = np.var(block, axis=0).mean()
    #         print(f"Subject {s} avg intra-trial variance: {var:.3e}")

    return pdc_clean


# -------------------- Model --------------------
class GraphDataset(Dataset):
    def __init__(self, features, adjs, labels, transform=None, sparsity=0.1):
        super().__init__(None, transform)
        self.features = features
        self.adjs = adjs
        self.labels = labels
        self.sparsity = sparsity

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x = self.features[idx]
        adj = self.adjs[idx]

        thr = np.percentile(adj, 100 * (1 - self.sparsity))
        mask = adj >= thr
        edge_index = torch.nonzero(torch.from_numpy(mask), as_tuple=False).t().contiguous()
        weights = adj[mask]
        edge_attr = torch.FloatTensor(weights)

        return Data(x=torch.FloatTensor(x),
                    edge_index=edge_index,
                    edge_weight=edge_attr,
                    y=torch.LongTensor([self.labels[idx]]))

# -------------------- Graph Normalization --------------------

# def normalize_edge(edge_index, edge_weight, num_nodes):
#     edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)
#     row, col = edge_index
#     deg = degree(row, num_nodes, edge_weight.dtype)
#     deg_inv_sqrt = deg.pow(-0.5)
#     norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#     return edge_index, norm_weight

# -------------------- Model --------------------
# class ChebGNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  K=3, num_layers=2, dropout=0.2):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.prelu = nn.PReLU()
#         self.dropout = dropout
#
#         # Build ChebConv layers with spectral normalization on weight parameter
#         for i in range(num_layers):
#             in_c = in_channels if i == 0 else hidden_channels
#             out_c = hidden_channels
#             conv = ChebConv(in_c, out_c, K)
#
#             for j in range(len(conv.lins)):  # Access the ModuleList of linear layers
#                 conv.lins[j] = spectral_norm(conv.lins[j], name='weight')  # Apply to 'weight' of each Linear layer
#
#             self.convs.append(conv)
#             self.bns.append(nn.BatchNorm1d(out_c))
#
#         # Final projection layer
#         out_conv = ChebConv(hidden_channels, out_channels, K)
#         # Apply spectral_norm to each internal linear layer of the final ChebConv
#         for j in range(len(out_conv.lins)):
#             out_conv.lins[j] = spectral_norm(out_conv.lins[j], name='weight')
#         self.out_conv = out_conv
#         self.linear = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, edge_weight, batch):
#         # edge_index, edge_weight = normalize_edge(edge_index, edge_weight, x.size(0))
#
#         # edge_index = edge_index.to(x.device)
#         # edge_weight = edge_weight.to(x.device)
#
#         for conv, bn in zip(self.convs, self.bns):
#             conv = conv.to(device)
#             x = conv(x, edge_index, edge_weight)
#             # x = bn(x)
#             x = self.prelu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         x = global_mean_pool(x, batch)
#         x = self.linear(x)
#         # x = self.out_conv(x, edge_index, edge_weight)
#
#         return x


# -------------------- Training --------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    n_folds = 10
    num_subs = 123
    batch_size = 256
    lr = 1e-2
    weight_decay = 1e-5
    categories = 9
    epochs = 100
    sparsity = 0.2 # top 10% edges

    met_calc = Metrics(num_class=categories)

    # Load and preprocess adjacency
    data_path = 'data/processed_conn_30_mod_4.mat'
    A_pdc = sio.loadmat(data_path)['data']
    A_pdc = np.ones_like(A_pdc) * 0.1
    A_pdc = np.mean(A_pdc, axis=1)

    # A_pdc = preprocess_pdc(A_pdc, trials_per_subject=28*11).reshape(A_pdc.shape)

    # Standardize features across all nodes
    # placeholder scaler, fit later per fold

    # CV folds
    results = {'acc': [], 'rec': [], 'prec': [], 'f1': []}
    n_per = num_subs // n_folds

    for fold in range(n_folds):
        # load features & labels
        feat_mat = sio.loadmat(f'data/features/de_lds_fold{fold}.mat')['de_lds']
        labels_rep = load_data.load_srt_de(feat_mat, True, 'cls9', 11)

        # split
        start = fold * n_per
        end = (fold+1)*n_per if fold < n_folds-1 else num_subs
        val_ids = np.arange(start, end)
        train_ids = np.setdiff1d(np.arange(num_subs), val_ids)

        # reshape data
        X_train = feat_mat[train_ids].reshape(-1, 30, 5)
        X_val   = feat_mat[val_ids].reshape(-1, 30, 5)
        y_train = np.repeat(labels_rep, len(train_ids))
        y_val   = np.repeat(labels_rep, len(val_ids))

        # standardize features per node-feature dim
        # scaler = StandardScaler()
        # X_flat = X_train.reshape(-1, X_train.shape[-1])
        # scaler.fit(X_flat)
        # X_train = scaler.transform(X_flat).reshape(X_train.shape)
        # X_val   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        # adjacency
        A_train = A_pdc[train_ids].reshape(-1, 30, 30)
        A_val   = A_pdc[val_ids].reshape(-1, 30, 30)

        # datasets
        # ds_tr = GraphDataset(X_train, A_train, y_train, sparsity=sparsity)
        ds_tr = TensorDataset(torch.tensor(X_train).to(device),
                              torch.tensor(A_train).to(device),
                              torch.tensor(y_train).to(device))
        # ds_va = GraphDataset(X_val,   A_val,   y_val, sparsity=sparsity)
        ds_va = TensorDataset(torch.tensor(X_val).to(device),
                              torch.tensor(A_val).to(device),
                              torch.tensor(y_val).to(device))
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        # model, opt, sched
        model = GCNBatch(5, 2, 9, dropout=0.2).to(device)
        # model = SAGENet(5, 64, 9, num_layers=3, dropout=0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        # training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            for data, adj, label in loader_tr:
                data = data.to(device)
                label = label.to(torch.long)
                optimizer.zero_grad()
                out = model(data.to(torch.float32), adj)
                # loss = criterion(out, data.y.view(-1))
                loss = criterion(out, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                # total_loss += loss.item() * data.num_graphs
                total_loss += loss.item()
                # correct += (out.argmax(1) == data.y.view(-1)).sum().item()
                correct += (out.argmax(1) == label).sum().item()
            train_acc = correct / len(ds_tr)

            # validation
            model.eval()
            val_loss = 0
            preds, labs = [], []
            with torch.no_grad():
                # for data in loader_va:
                for data, adj, label in loader_va:
                    data = data.to(device)
                    label = label.to(torch.long)
                    out = model(data.to(torch.float32), adj)
                    # val_loss += criterion(out, data.y.view(-1)).item()
                    val_loss += criterion(out, label).item()
                    preds.append(out.argmax(1).cpu())
                    # labs.append(data.y.view(-1).cpu())
                    labs.append(label.cpu())
            val_loss /= len(loader_va)
            # scheduler.step()

            preds = torch.cat(preds)
            labs = torch.cat(labs)
            acc, rec, prec, f1 = met_calc.compute_metrics(preds, labs)

            # if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Fold {fold} Epoch {epoch:03d}  TrAcc={train_acc:.3f}  ValLoss={val_loss:.4f}  ValAcc={acc:.3f}  F1={f1:.3f}")

        # final metrics
        results['acc'].append(acc)
        results['rec'].append(rec)
        results['prec'].append(prec)
        results['f1'].append(f1)

    # summary
    print("=== Final CV Results ===")
    for k,v in results.items():
        arr = np.array(v)
        print(f"{k}: {arr.mean():.3f} ± {arr.std():.3f}")
