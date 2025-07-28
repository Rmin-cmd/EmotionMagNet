import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from typing import List
from torch import Tensor
from torch.jit import Future

# import all signed-directed convolution layers and models
from torch_geometric_signed_directed.nn import (
    DGCNConv,
    MagNetConv,
    DiGCNConv,
    DiGCN_node_classification,
    DiGCN_Inception_Block_node_classification,
    MSConv
)
from torch.jit import ScriptModule, script_method
from torch_geometric_signed_directed.nn import DGCN_node_classification
from torch_geometric_signed_directed.nn import DiGCN_node_classification as SimpleDiGCN
from torch_geometric_signed_directed.nn import MSGNN_node_classification as PygMSGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Base template for directed GCNNs with two message-passing layers
class BaseDirectedNet(nn.Module):
    def __init__(self, conv1, conv2, in_c, args):
        super().__init__()
        self.conv1 = conv1(in_c, args.num_filter)
        self.conv2 = conv2(args.num_filter, args.num_filter)
        self.lin = nn.Linear(args.num_filter, 1)
        self.head = nn.Conv1d(30, args.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, real, imag, graph, layer=2):
        x = real.to(device)
        adj = torch.mean(graph, dim=1).real

        data_list = []
        for i in range(x.size(0)):
            sp_adj = sp.coo_matrix(adj[i].cpu().numpy())
            edge_index, edge_weight = from_scipy_sparse_matrix(sp_adj)
            data = Data(
                x=x[i],
                edge_index=edge_index.to(device),
                edge_weight=edge_weight.to(device)
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        xi = self.conv1(batch.x, batch.edge_index, getattr(batch, 'edge_weight', None))
        xi = self.relu(xi)
        xi = self.dropout(xi)

        if layer > 1:
            xi = self.conv2(xi, batch.edge_index, getattr(batch, 'edge_weight', None))
            xi = self.relu(xi)
            xi = self.dropout(xi)

        num_nodes = data_list[0].num_nodes
        x_out = xi.view(x.size(0), num_nodes, -1)

        x_lin = self.lin(x_out).squeeze(-1)
        x_relu = self.relu(x_lin).unsqueeze(-1)
        out = self.head(x_relu).squeeze(-1)
        return out


class DGCNNet(ScriptModule):
    def __init__(self, in_c, **kwargs):
        super().__init__()
        args = kwargs['args']
        self.num_nodes   = 30
        self.num_classes = args.num_classes

        # 1) single‐graph DGCN node‐classification
        self.model = DGCN_node_classification(
            num_features=in_c,
            hidden      = args.num_filter,
            label_dim   = args.num_classes,
            dropout     = args.dropout,
            improved    = getattr(args, 'improved', False),
            cached      = False
        )

    @script_method
    def forward(self,
                real: torch.Tensor,
                imag: torch.Tensor,
                graph_sigs: torch.Tensor
                ) -> torch.Tensor:
        B, N, _ = real.size()

        # Annotate the list as holding Future[Tensor]s:
        handles: List[Future[Tensor]] = []

        for i in range(B):
            # fork returns a Future[Tensor]:
            h: Future[Tensor] = torch.jit.fork(self._per_graph,
                                               real[i],  # [N,F]
                                               graph_sigs[i]  # [T,N,N]
                                               )
            handles.append(h)

        # Now wait on them, producing actual Tensors:
        results: List[Tensor] = []
        for h in handles:
            results.append(torch.jit.wait(h))

        # Stack into [B, C]
        return torch.stack(results, dim=0)

    @script_method
    def _per_graph(self,
                   x: torch.Tensor,      # [N, F]
                   graph_sig: torch.Tensor  # [T, N, N]
                  ) -> torch.Tensor:
        # average & real part → [N,N]
        adj = torch.mean(graph_sig, dim=0).real

        # build edge sets
        sym = adj + adj.t()
        # src, dst = (sym != 0).nonzero(as_tuple=True)
        # edge_index = torch.stack([src, dst], dim=0).to(device)
        sym = adj + adj.t()
        # nonzero() returns a [E,2] tensor of (row, col) pairs
        idx = (sym != 0).nonzero()
        src = idx[:, 0]
        dst = idx[:, 1]

        edge_index = torch.stack([src, dst], dim=0).to(x.device)

        # src_in, dst_in = (adj != 0).nonzero(as_tuple=True)
        idx_in = (adj != 0).nonzero()
        src_in = idx_in[:, 0]
        dst_in = idx_in[:, 1]
        edge_in = torch.stack([src_in, dst_in], dim=0).to(x.device)
        # edge_out = torch.stack([dst_in, src_in], dim=0).to(device)
        edge_out = torch.stack([dst_in, src_in], dim=0).to(x.device)

        # per‐node log‐probs [N, C]
        node_logp = self.model(
            x.to(x.device),
            edge_index,
            edge_in,
            edge_out,
            None,  # in_w
            None   # out_w
        )
        # here you averaged across nodes; you could do pooling instead
        return node_logp.mean(dim=0)


class DiGCNNet(nn.Module):
    """
    Wrapper around the plain two‐layer DiGCN_node_classification (no inception),
    producing one graph‐level prediction per input graph.
    """
    def __init__(self, in_c, **kwargs):
        super().__init__()
        args = kwargs['args']
        self.num_nodes   = 30
        # self.num_classes = args.num_classes
        #
        # # 1) two‐layer DiGCN (node‐level)
        # self.model = SimpleDiGCN(
        #     num_features=in_c,
        #     hidden      = args.num_filter,
        #     label_dim   = args.num_classes,
        #     dropout     = args.dropout
        # )

        self.conv1 = DiGCNConv(in_channels=in_c, out_channels=args.num_filter, cached=False)
        # self.conv2 = DiGCNConv(in_channels=args.num_filter, out_channels=args.num_classes)

        # 2) pool per‐node C→1
        self.pool_lin = nn.Linear(args.num_filter, 1, bias=True)
        nn.init.zeros_(self.pool_lin.bias)

        # 3) head: treat N node‐scores as channels → C graph‐scores
        self.head = nn.Conv1d(self.num_nodes, args.num_classes, kernel_size=1)

    def forward(self, real, imag, graph_sigs, **kwargs):
        """
        real:      [B, N, F]
        imag:      [B, N, F]      (ignored by the simple DiGCN)
        graph_sigs:[B, T, N, N]
        returns:   [B, C] graph‐level log‐probs
        """
        B, N, _ = real.shape
        outs = []

        for i in range(B):
            # 1) average & real part → [N, N]
            adj = torch.mean(graph_sigs[i], dim=0).real.cpu().numpy()
            sp_adj = sp.coo_matrix(adj)

            # build edge_index + edge_weight
            edge_index, edge_weight = from_scipy_sparse_matrix(sp_adj)
            edge_index  = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            # 2) node‐level log‐probs [N, C]
            x = real[i].to(device)
            # node_logp = self.model(x, edge_index, edge_weight)
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            # x = F.dropout(x, p=0.2, training=self.training)
            # x = F.relu(self.conv2(x, edge_index, edge_weight))

            # 3) pool C→1
            node_score = self.pool_lin(x).squeeze(-1)  # [N]

            # 4) head conv1d over N node‐scores
            t = node_score.unsqueeze(0).unsqueeze(-1)         # [1, N, 1]
            graph_logp = F.softmax(self.head(t).squeeze(-1).squeeze(0))   # [C]

            outs.append(graph_logp)

        return torch.stack(outs, dim=0)  # [B, C]


