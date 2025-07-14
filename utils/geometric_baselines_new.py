import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedGCNLayer(nn.Module):
    """
    Batched GCN layer for input x: [B, N, F_in], adj: [B, N, N]
    Computes Â = A + I, normalizes with D̂^{-1/2} Â D̂^{-1/2},
    then H_out = σ(A_norm @ (X W) + b).
    """
    def __init__(self, in_features, out_features, bias=True):
        super(BatchedGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features)).to(torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [B, N, F_in], adj: [B, N, N]
        B, N, _ = x.size()
        # add self-loops
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj + I
        # degree
        deg = A_hat.sum(dim=2)               # [B, N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = deg_inv_sqrt.unsqueeze(2) * I  # broadcast to [B,N,N]
        # normalized adjacency
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt)
        # linear transform
        support = torch.matmul(x, self.weight)      # [B, N, out_features]
        out = torch.bmm(A_norm, support)            # [B, N, out]
        if self.bias is not None:
            out = out + self.bias.view(1,1,-1)
        return out

class GCNBatch(nn.Module):
    """
    Two-layer GCN for batched graph classification.
    Input shapes:
      x: [B, N, F_in]
      adj: [B, N, N]
    Output: log probabilities [B, num_classes]
    """
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super(GCNBatch, self).__init__()
        self.conv1 = BatchedGCNLayer(in_features, hidden_dim)
        self.conv2 = BatchedGCNLayer(hidden_dim, hidden_dim)
        self.fc = nn.Conv1d(30, 9, kernel_size=1)
        self.dropout = dropout

    def forward(self, x, adj):
        # First GCN layer + ReLU + dropout
        x = self.conv1(x, adj)               # [B, N, hidden]
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Second GCN layer + ReLU
        x = self.conv2(x, adj)
        x = F.relu(x)
        # Global mean pooling over nodes
        graph_repr = x.mean(dim=2)           # [B, hidden]
        out = self.fc(graph_repr.unsqueeze(2))            # [B, num_classes]
        return out.squeeze()

# Example usage with your data:
# Suppose you have:
#   adj_tensor: torch.Tensor of shape [batch_size, 30, 30]
#   feat_tensor: torch.Tensor of shape [batch_size, 150]
# If each graph has 30 nodes and 5 features per node (30*5 = 150), then:
#   x = feat_tensor.view(batch_size, 30, 5)
#   adj = adj_tensor.float()
# model = GCNBatch(in_features=5, hidden_dim=64, num_classes=3)
# logits = model(x, adj)
