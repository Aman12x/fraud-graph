"""Tier 4: card-history Transformer. The current txn is an extra, never-masked final token."""
import torch
from torch import nn

from fraudgraph.config import SEQ_LEN


class SeqTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_len: int = SEQ_LEN):
        super().__init__()
        self.hist_proj = nn.Linear(n_features + 1, d_model)  # +1 = log-gap channel
        self.cur_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(max_len + 1, d_model))  # last slot = current
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, hist: torch.Tensor, mask: torch.Tensor, current: torch.Tensor):
        """hist [N,k,F+1], mask [N,k] True=padded, current [N,F] -> logits [N]."""
        k = hist.shape[1]
        h = self.hist_proj(hist) + self.pos[:k]
        c = (self.cur_proj(current) + self.pos[-1]).unsqueeze(1)
        x = self.drop(torch.cat([h, c], dim=1))
        full = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        return self.head(self.encoder(x, src_key_padding_mask=full)[:, -1]).squeeze(-1)
