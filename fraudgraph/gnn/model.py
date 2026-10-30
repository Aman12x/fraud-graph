"""Tier 2: heterogeneous GraphSAGE over the transaction-entity graph."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero


class _SageStack(nn.Module):
    def __init__(self, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(SAGEConv((-1, -1), hidden) for _ in range(num_layers))
        self.drop = nn.Dropout(dropout)  # a module, so eval() reaches it after FX tracing

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.drop(x)
        return x


class HeteroSAGE(nn.Module):
    def __init__(self, metadata, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gnn = to_hetero(_SageStack(hidden, num_layers, dropout), metadata, aggr="sum")
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch) -> torch.Tensor:
        """Logits for the seed transactions (first batch['txn'].batch_size rows)."""
        out = self.gnn(batch.x_dict, batch.edge_index_dict)
        return self.head(out["txn"][: batch["txn"].batch_size]).squeeze(-1)
