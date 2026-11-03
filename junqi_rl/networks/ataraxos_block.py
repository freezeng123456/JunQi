"""Official Ataraxos pre-norm block, with portable PyTorch SDPA dispatch.

Separate Q/K/V linear layers preserve the upstream initialization as well as
the computation. Source: AtaraxosAI/stratego @ 92db29e8, transformer_basics.py.
"""
from torch import nn
from torch.nn import functional as F


class AtaraxosBlock(nn.Module):
    def __init__(self, width, heads, ff_factor=4, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (
            nn.Linear(width, width) for _ in range(4))
        self.linear1 = nn.Linear(width, width * ff_factor)
        self.linear2 = nn.Linear(width * ff_factor, width)

    def forward(self, x):
        b, t, d = x.shape
        z = self.norm1(x)
        q, k, v = (layer(z).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for layer in (self.q_proj, self.k_proj, self.v_proj))
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        z = self.out_proj(z.transpose(1, 2).reshape(b, t, d))
        x = x + F.dropout(z, self.dropout, self.training)
        z = self.linear2(F.relu(self.linear1(self.norm2(x))))
        return x + F.dropout(z, self.dropout, self.training)
