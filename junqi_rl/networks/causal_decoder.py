"""Shared causal decoder with cross-attention and incremental KV caching."""
from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class _DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.heads = cfg.n_head
        self.width = d // cfg.n_head
        self.norm_self, self.norm_cross, self.norm_ff = (nn.LayerNorm(d) for _ in range(3))
        self.qkv = nn.Linear(d, 3*d)
        self.self_out = nn.Linear(d, d)
        self.cross_q = nn.Linear(d, d)
        self.cross_kv = nn.Linear(d, 2*d)
        self.cross_out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d*cfg.ff_factor), nn.GELU(),
                                nn.Linear(d*cfg.ff_factor, d))
        self.dropout = cfg.dropout

    def split(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.width).transpose(1, 2)

    def merge(self, x: Tensor) -> Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], -1, self.heads*self.width)

    def memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.cross_kv(memory).chunk(2, -1)
        return self.split(k), self.split(v)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        q, k, v = map(self.split, self.qkv(self.norm_self(x)).chunk(3, -1))
        drop = self.dropout if self.training else 0.0
        x = x + self.self_out(self.merge(F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=drop)))
        k, v = self.memory(memory)
        q = self.split(self.cross_q(self.norm_cross(x)))
        x = x + self.cross_out(self.merge(F.scaled_dot_product_attention(q, k, v,
                                                                     dropout_p=drop)))
        return x + self.ff(self.norm_ff(x))

    def step(self, x: Tensor, memory_kv, cache, index: int) -> Tensor:
        q, k, v = map(self.split, self.qkv(self.norm_self(x)).chunk(3, -1))
        cache[0][:, :, index:index+1] = k
        cache[1][:, :, index:index+1] = v
        x = x + self.self_out(self.merge(F.scaled_dot_product_attention(
            q, cache[0][:, :, :index+1], cache[1][:, :, :index+1])))
        q = self.split(self.cross_q(self.norm_cross(x)))
        x = x + self.cross_out(self.merge(F.scaled_dot_product_attention(q, *memory_kv)))
        return x + self.ff(self.norm_ff(x))

