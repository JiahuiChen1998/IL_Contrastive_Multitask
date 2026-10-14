"""Task-specific channel attention with an identity-initialized capacity control."""
import torch
from torch import nn
from il_mixer.model import ILMixer


class HeatTransferMixer(ILMixer):
    def __init__(self, tasks, hidden=64, layers=3, heads=4, cross_ion=True,
                 head_mode='mlp', feature_mode='plain', adapter_width=32):
        if head_mode != 'mlp' or feature_mode not in {'plain', 'attention', 'adapter'}:
            raise ValueError('This experiment uses MLP heads and plain/attention/adapter features')
        super().__init__(tasks, hidden, layers, heads, cross_ion, head_mode)
        self.feature_mode = feature_mode
        self.task_features = nn.ModuleDict()
        if feature_mode != 'plain':
            for task in tasks:
                net = nn.Sequential(nn.LayerNorm(2*hidden), nn.Linear(2*hidden, adapter_width),
                                    nn.GELU(), nn.Linear(adapter_width, 2*hidden))
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                self.task_features[task] = net

    def task_representation(self, h, task):
        if self.feature_mode == 'plain':
            return h
        delta = self.task_features[task](h)
        if self.feature_mode == 'attention':
            # Each property chooses channels of the shared atom/bond pooled vector.
            # Gates start at one and remain in (0, 2); this is not teacher distillation.
            return h * (2 * torch.sigmoid(delta))
        return h + delta

    def forward(self, g, temperature, pressure, task_names=None):
        if self.feature_mode == 'plain':
            return super().forward(g, temperature, pressure, task_names)
        h = self.encode(g)
        outputs = {}
        for task, spec in self.tasks.items():
            active = None if task_names is None else torch.tensor([x == task for x in task_names], device=h.device)
            t = temperature if active is None else torch.where(active, temperature, 298.15)
            p = pressure if active is None else torch.where(active, pressure, 0.)
            inputs = [self.task_representation(h, task)]
            if spec['temperature']:
                inputs.append(((t-298.15)/100).unsqueeze(-1))
            if spec['pressure']:
                inputs.append(torch.log1p(p).unsqueeze(-1))
            outputs[task] = self.readouts[task](torch.cat(inputs, -1)).squeeze(-1)
        return outputs
