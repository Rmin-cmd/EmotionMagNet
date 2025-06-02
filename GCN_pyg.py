import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import torch.optim as optim
import time
from tqdm import tqdm
from utils import load_data

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.data import Data, Dataset
from torch.nn import Linear
import os
from utils.utils_loss import *


class GraphDataset(Dataset):
    def __init__(self, feature_matrices, adjacency_matrices, labels, transform=None):
        super(GraphDataset, self).__init__(None, transform)
        self.feature_matrices = feature_matrices
        self.adjacency_matrices = adjacency_matrices
        self.labels = labels

    def len(self):
        return len(self.labels)

    def get(self, idx):
        feature_matrix = self.feature_matrices[idx]
        adj_matrix = self.adjacency_matrices[idx]
        label = self.labels[idx]

        # Convert adjacency matrix to edge index
        # edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        # print(self.adjacency_matrices.shape)
        edge_index = torch.nonzero(adj_matrix).t().contiguous()

        # Create the Data object
        data = Data(x=feature_matrix, edge_index=edge_index, y=label)
        return data


class GCN(nn.Module):
    """
    GCN with PyTorch Geometric's GCNConv.

    :param in_channels: int, number of input features.
    :param out_channels: int, number of output classes.
    :param hidden_channels: int, number of hidden channels.
    :param num_layers: int, number of GCN layers.
    :param dropout: float, dropout rate.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # self.convs.append(GCNConv(hidden_channels, out_channels))
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        # x = self.convs[-1](x, edge_index)
        x = self.lin(x)
        return x


# Load data
data_path = 'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_3.mat'
# feature_path = './feature_data/feature_preprocess/DE/feature_preprocess.mat'
root_dir = r'preprocessed_data\preprocessed_feature\smooth_preprocessed_28'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
n_folds = 10
num_subs = 123
batch_size = 32
lr = 5e-4
l2 = 5e-4
categories = 9
epochs = 50
n_per = num_subs // n_folds
label_type = 'cls2' if categories == 2 else 'cls9'

met_calc = Metrics(num_class=9)

# Load adjacency and feature matrices
A_pdc = sio.loadmat(data_path)['data']
A_pdc = np.mean(A_pdc, axis=1)

# Initialize metric storage
acc_fold, re_fold, pre_fold, f1_fold = [], [], [], []

# Define loss and metrics
criterion = nn.CrossEntropyLoss()

# Cross-validation
for fold in tqdm(range(n_folds)):
    data_dir = os.path.join(root_dir,'de_lds_fold%d.mat' % (fold))
    feature_pdc = sio.loadmat(data_dir)['de_lds']
    # feature_pdc = sio.loadmat(feature_path)['de_fold' + str(fold)]

    label_repeat = load_data.load_srt_de(feature_pdc, True, label_type, 10)

    # Split data
    if fold < n_folds - 1:
        val_sub = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_sub = np.arange(n_per * fold, n_per * (fold + 1) - 1)

    train_sub = list(set(np.arange(num_subs)) - set(val_sub))

    data_train = feature_pdc[list(train_sub), :, :].reshape(-1, 30, 5)
    data_val = feature_pdc[list(val_sub), :, :].reshape(-1, 30, 5)

    label_train = np.tile(label_repeat, len(train_sub))
    label_val = np.tile(label_repeat, len(val_sub))

    A_pdc_train, A_pdc_valid = A_pdc[train_sub].reshape([-1, 30, 30]), \
                               A_pdc[val_sub].reshape([-1, 30, 30])

    feature_train = torch.FloatTensor(data_train).to(device)
    feature_valid = torch.FloatTensor(data_val).to(device)
# Graph edges for validation set

    dataset_train = GraphDataset(feature_train, torch.from_numpy(A_pdc_train).to(device), torch.LongTensor(label_train).to(device))
    dataset_valid = GraphDataset(feature_valid, torch.from_numpy(A_pdc_valid).to(device), torch.LongTensor(label_val).to(device))

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = GCN(in_channels=5, hidden_channels=64, out_channels=categories, num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    met = []
    # Training and validation loop
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        total_loss_train = 0.0
        correct_train = 0

        for data in train_loader:

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            correct_train += (out.argmax(dim=1) == data.y).sum().item()

        # Validation
        model.eval()
        total_loss_valid = 0.0
        correct_valid = 0

        pred_, label_ = [], []

        with torch.no_grad():
            for data in valid_loader:
                # edge_index_valid = [torch.nonzero(torch.from_numpy(A)) for A in graph]
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss_valid += loss.item()
                pred_label = out.max(dim=1)[1].tolist()
                pred_ += pred_label
                label_ += data.y.tolist()
                # correct_valid += (out.argmax(dim=1) == data.y).sum().item()

        final_metrics = met_calc.compute_metrics(torch.tensor([pred_]).to(device),
                                                 torch.tensor([label_]).to(device))
        met.append(final_metrics)

        outstrtrain = 'epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % (
        epoch,
        total_loss_valid / len(valid_loader), final_metrics[0], final_metrics[1], final_metrics[2], final_metrics[3])

        print(outstrtrain)

    acc_fold.append(np.mean(met, axis=0)[0]), re_fold.append(np.mean(met, axis=0)[1]), pre_fold.append(
        np.mean(met, axis=0)[2]),
    f1_fold.append(np.mean(met, axis=0)[3])

print(
    'folds accuracy: %.3f ± %.3f, folds recall: %.3f ± %.3f, folds precision: %.3f ± %.3f, folds F1-Score: %.3f ± %.3f' %
    (
    np.mean(acc_fold), np.std(acc_fold), np.mean(re_fold), np.std(re_fold), np.mean(pre_fold), np.std(pre_fold),
    np.mean(f1_fold), np.std(f1_fold)))
