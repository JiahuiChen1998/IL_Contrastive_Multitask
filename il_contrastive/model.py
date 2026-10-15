"""Compact EGT adaptation; architecture provenance is recorded in docs/design.md.

EGT attention equations follow Bamboo Mixer's Apache-2.0 implementation,
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates. Reimplemented in PyTorch
without PyG; this is not checkpoint-compatible with the upstream model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def pool(x, ids, n):
    sums = x.new_zeros(n, x.shape[-1]).index_add_(0, ids, x)
    counts = torch.bincount(ids, minlength=n).clamp_min(1).unsqueeze(-1)
    return sums / counts


class EGTLayer(nn.Module):
    """Local multihead attention with edge bias, gate and edge updates."""
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by heads")
        self.heads, self.width = heads, dim // heads
        self.norm = nn.LayerNorm(dim); self.enorm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.bias = nn.Linear(dim, heads); self.gate = nn.Linear(dim, heads)
        self.proj = nn.Linear(dim, dim); self.eproj = nn.Linear(heads, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), mlp(dim, dim * 2, dim))
        self.eff = nn.Sequential(nn.LayerNorm(dim), mlp(dim, dim * 2, dim))

    def forward(self, x, e, edge_index):
        src, dst = edge_index
        q, k, v = self.qkv(self.norm(x)).reshape(-1, 3, self.heads, self.width).unbind(1)
        en = self.enorm(e)
        logits = (q[dst] * k[src]).sum(-1) / math.sqrt(self.width) + self.bias(en)
        maxima = x.new_full((len(x), self.heads), -torch.inf)
        maxima.scatter_reduce_(0, dst[:, None].expand_as(logits), logits, reduce='amax', include_self=True)
        weights = (logits - maxima[dst]).exp()
        denom = x.new_zeros(len(x), self.heads).index_add_(0, dst, weights)
        weights = weights / denom[dst].clamp_min(1e-12) * self.gate(en).sigmoid()
        messages = (v[src] * weights.unsqueeze(-1)).reshape(-1, self.heads * self.width)
        agg = torch.zeros_like(x).index_add_(0, dst, messages)
        x = x + self.proj(agg); x = x + self.ff(x)
        e = e + self.eproj(logits); e = e + self.eff(e)
        # Both directions represent the same physical bond.
        e = e.reshape(-1, 2, e.shape[-1]).mean(1).repeat_interleave(2, 0)
        return x, e


class ILMixer(nn.Module):
    def __init__(self, tasks, hidden=64, layers=3, heads=4, cross_ion=False, head_mode='physics', t_floor=150.0, target_scalers=None, cp_degree=3):
        super().__init__()
        self.tasks = tasks
        self.head_mode, self.t_floor = head_mode, t_floor
        self.embeddings = nn.ModuleList([nn.Embedding(n, hidden) for n in [119, 17, 9, 9, 2, 10, 301]])
        self.edge_embedding = nn.Linear(5, hidden)
        self.layers = nn.ModuleList([EGTLayer(hidden, heads) for _ in range(layers)])
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True) if cross_ion else None
        self.cross_norm = nn.LayerNorm(hidden) if cross_ion else None
        self.readouts = nn.ModuleDict()
        for name, spec in tasks.items():
            if head_mode == 'equations' and spec['temperature']:
                from .equations import EquationHead
                scale = (target_scalers or {}).get(name, {'mean':0., 'std':1.})
                self.readouts[name] = EquationHead(name, hidden*2, hidden*2, **scale, cp_degree=cp_degree, t_floor=t_floor)
                continue
            physics = head_mode == 'physics' and spec['head'] in ('vft', 'linear')
            inputs = hidden * 2 + (0 if physics else int(bool(spec['temperature'])) + int(bool(spec['pressure'])))
            outputs = 3 if physics and spec['head'] == 'vft' else 2 if physics else 1
            self.readouts[name] = mlp(inputs, hidden * 2, outputs)

    def encode(self, g):
        x = sum(emb(g['x'][:, i]) for i, emb in enumerate(self.embeddings))
        e = self.edge_embedding(g['edge_attr'])
        for layer in self.layers:
            x, e = layer(x, e, g['edge_index'])
        n = int(g['batch'].max()) + 1
        if self.cross is not None:
            parts = []
            for i in range(n):
                mask = g['batch'] == i
                xi = x[mask].unsqueeze(0)
                fi = g['fragment'][mask]
                # Every query has at least one other fragment; no all-masked rows.
                attended, _ = self.cross(xi, xi, xi, attn_mask=fi[:, None] == fi[None, :], need_weights=False)
                parts.append(self.cross_norm(xi + attended).squeeze(0))
            x = torch.cat(parts)
        return torch.cat([pool(x, g['batch'], n), pool(e, g['batch'][g['edge_index'][0]], n)], -1)

    def forward(self, g, temperature, pressure, task_names=None):
        h = self.encode(g)
        out = {}
        for name, spec in self.tasks.items():
            active = None if task_names is None else torch.tensor([n == name for n in task_names], device=h.device)
            t = temperature if active is None else torch.where(active, temperature, 298.15)
            pressure_task = pressure if active is None else torch.where(active, pressure, 0.)
            if self.head_mode == 'equations' and spec['temperature']:
                out[name] = self.readouts[name](h, t, pressure_task)
                continue
            physics = self.head_mode == 'physics' and spec['head'] in ('vft', 'linear')
            if physics:
                p = self.readouts[name](h)
                if spec['head'] == 'vft':
                    # Predict standardized ln(viscosity). T0 stays below the supported domain.
                    t0 = self.t_floor - 1.0 - F.softplus(p[:, 2]) * 50
                    if not torch.isfinite(t).all() or (t < self.t_floor).any():
                        raise ValueError(f"VFT requires finite T >= {self.t_floor} K")
                    out[name] = p[:, 0] + F.softplus(p[:, 1]) * 300 * (
                        1 / (t - t0) - 1 / (298.15 - t0))
                else:
                    out[name] = p[:, 0] - F.softplus(p[:, 1]) * (t - 298.15) / 100
            else:
                inputs = [h]
                if spec['temperature']:
                    inputs.append(((t - 298.15) / 100).unsqueeze(-1))
                if spec['pressure']:
                    inputs.append(torch.log1p(pressure_task).unsqueeze(-1))  # task-specific pressure units; see tasks.py
                out[name] = self.readouts[name](torch.cat(inputs, -1)).squeeze(-1)
        return out
