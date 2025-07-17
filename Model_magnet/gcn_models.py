import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GINConv, APPNP
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
import numpy as np
import scipy.sparse as sp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GCNNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(GCNNet, self).__init__()
        args = kwargs['args']
        self.conv1 = GCNConv(in_c, args.num_filter)
        self.conv2 = GCNConv(args.num_filter, args.num_filter)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index)
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index)
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)


class SAGENet(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(SAGENet, self).__init__()
        args = kwargs['args']
        self.conv1 = SAGEConv(in_c, args.num_filter)
        self.conv2 = SAGEConv(args.num_filter, args.num_filter)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index)
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index)
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)


class GATNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(GATNet, self).__init__()
        args = kwargs['args']
        heads = 1
        self.conv1 = GATConv(in_c, args.num_filter, heads=heads)
        self.conv2 = GATConv(args.num_filter * heads, args.num_filter, heads=heads)
        self.lin = nn.Linear(args.num_filter * heads, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index)
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index)
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)


class ChebNetPyg(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(ChebNetPyg, self).__init__()
        args = kwargs['args']
        K = args.get('K', 2)
        self.conv1 = ChebConv(in_c, args.num_filter, K)
        self.conv2 = ChebConv(args.num_filter, args.num_filter, K)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index)
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index)
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)


class APPNPNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(APPNPNet, self).__init__()
        args = kwargs['args']
        alpha = 0.1
        self.line1 = nn.Linear(in_c, args.num_filter)
        self.line2 = nn.Linear(args.num_filter, args.num_filter)
        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.line1(batch.x)
        xi = self.relu(xi)
        xi = self.dropout(xi)
        xi = self.conv1(xi, batch.edge_index)

        if layer > 1:
            xi = self.line2(xi)
            xi = self.relu(xi)
            xi = self.dropout(xi)
            xi = self.conv2(xi, batch.edge_index)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)


class GINNet(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(GINNet, self).__init__()
        args = kwargs['args']
        self.line1 = nn.Linear(in_c, args.num_filter)
        self.conv1 = GINConv(self.line1)
        self.line2 = nn.Linear(args.num_filter, args.num_filter)
        self.conv2 = GINConv(self.line2)
        self.lin = nn.Linear(args.num_filter, 1)
        self.conv = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.shape[0]):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, _ = from_scipy_sparse_matrix(sp_adj)
            data = Data(x=x[i], edge_index=edge_index.to(device))
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index)
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index)
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.shape[0], num_nodes, -1)

        x = self.lin(x_out).squeeze(-1)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1)
