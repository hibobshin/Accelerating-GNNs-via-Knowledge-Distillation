import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_hidden=False):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = F.dropout(h, p=0.2, training=self.training)
        out = self.conv2(h, edge_index)

        if return_hidden:
            return out, h
        else:
            return out


class GATStudent(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        torch.manual_seed(1234567)
        self.heads = heads
        assert (
            hidden_channels % heads == 0
        ), "Hidden channels must be divisible by heads"
        head_channels = hidden_channels // heads

        self.conv1 = GATConv(in_channels, head_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, dropout=0.2
        )

    def forward(self, x, edge_index, return_hidden=False):
        x = F.dropout(x, p=0.6, training=self.training)
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h_intermediate = F.dropout(h, p=0.6, training=self.training)
        out = self.conv2(h_intermediate, edge_index)

        if return_hidden:
            return out, h
        else:
            return out
