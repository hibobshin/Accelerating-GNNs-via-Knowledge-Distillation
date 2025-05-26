import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5
    ):
        super().__init__()
        torch.manual_seed(1234567)
        self.heads = heads
        assert (
            hidden_channels % heads == 0
        ), "Hidden channels must be divisible by heads"
        head_channels = hidden_channels // heads
        self.dropout_p = dropout

        self.conv1 = TransformerConv(
            in_channels, head_channels, heads=heads, dropout=self.dropout_p
        )
        self.conv2 = TransformerConv(
            hidden_channels, out_channels, heads=1, concat=False, dropout=self.dropout_p
        )

    def forward(self, x, edge_index, return_hidden=False, return_attention=False):

        h, (edge_index_att1, att_weights1) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        hidden_features = h
        out = self.conv2(h, edge_index)

        results = [out]
        if return_hidden:
            results.append(hidden_features)
        if return_attention:
            results.append((edge_index_att1, att_weights1))

        return tuple(results) if len(results) > 1 else results[0]
