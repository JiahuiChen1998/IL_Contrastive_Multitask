"""User-specified property equations, evaluated in a common standardized loss space."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .tasks import TASKS

STATIC_TASKS = ['melting_point', 'glass_transition', 'decomposition_temperature', 'toxicity']
CONDITION_TASKS = [n for n in TASKS if n not in STATIC_TASKS]


def study_tasks():
    tasks = copy.deepcopy(TASKS)
    # Both equation and concatenation variants must use identical target spaces.
    tasks['heat_capacity']['transform'] = 'identity'  # polynomial is Cp, not ln(Cp)
    tasks['co2_solubility']['transform'] = 'log'  # the supplied equation is ln(x)
    return tasks


def choose_cp_degree(training):
    sub = training[training.task == 'heat_capacity']
    counts = sub.groupby('smiles').temperature_K.nunique()
    # Conservative coverage rule, independent of target values/validation/test labels.
    degree = 1
    for candidate in (2, 3):
        if len(sub) >= 50 * (candidate + 1) and len(counts) >= 20 and (counts >= 2*(candidate+1)).mean() >= .5:
            degree = candidate
    return degree, {'training_rows':len(sub), 'training_ils':len(counts),
                    'median_distinct_T':None if counts.empty else float(counts.median()),
                    'ils_with_eight_temperatures':int((counts >= 8).sum()), 'selected_degree':degree,
                    'rule':'At least 50*(degree+1) rows, 20 ILs, and half of ILs with >=2*(degree+1) distinct temperatures; maximum cubic.'}


class EquationHead(nn.Module):
    def __init__(self, name, input_dim, hidden, mean=0., std=1., cp_degree=3, t_floor=150.):
        super().__init__()
        self.name, self.mean, self.std = name, float(mean), float(std)
        self.cp_degree, self.t_floor = cp_degree, t_floor
        sizes = {'viscosity':3, 'surface_tension':2, 'density':3, 'heat_capacity':cp_degree+1,
                 'co2_solubility':2, 'refractive_index':2, 'thermal_conductivity':2}
        self.net = nn.Sequential(nn.Linear(input_dim + int(name == 'co2_solubility'), hidden),
                                 nn.GELU(), nn.Linear(hidden, sizes[name]))

    def parameters_for(self, h, pressure):
        if self.name == 'co2_solubility':
            h = torch.cat([h, torch.log1p(pressure).unsqueeze(-1)], -1)
        return self.net(h)

    def forward(self, h, temperature, pressure):
        q = self.parameters_for(h, pressure)
        u = (temperature - 298.15) / 100
        if self.name == 'viscosity':
            if not torch.isfinite(temperature).all() or (temperature < self.t_floor).any():
                raise ValueError('VFT requires T >=150 K')
            t0 = self.t_floor - 1 - 50 * F.softplus(q[:, 2])
            return q[:, 0] + 300 * F.softplus(q[:, 1]) * (1/(temperature-t0) - 1/(298.15-t0))
        if self.name == 'density':
            return q[:, 0] + q[:, 1]*u + q[:, 2]*pressure/100
        if self.name == 'heat_capacity':
            # Centering is a change of polynomial basis, NOT a polynomial of log Cp.
            x = temperature / 100 - 3
            out = q[:, -1]
            for k in range(self.cp_degree-1, -1, -1): out = out*x + q[:, k]
            return out
        if self.name == 'co2_solubility':
            return q[:, 0] + q[:, 1]*(300/temperature - 300/298.15)
        slope = -F.softplus(q[:, 1]) if self.name == 'refractive_index' else q[:, 1]
        return q[:, 0] + slope*u

    def physical_coefficients(self, h, pressure):
        """Coefficients in requested bases after undoing target standardization.

        Cp retains workbook-native target units until its molar units are verified.
        CO2 pressure in MPa; apparent delta_h in J/mol under the supplied relation.
        """
        q = self.parameters_for(h, pressure)
        s, m = self.std, self.mean
        if self.name == 'viscosity':
            t0 = self.t_floor - 1 - 50*F.softplus(q[:, 2])
            b = s*300*F.softplus(q[:, 1])
            return {'a':m+s*q[:, 0], 'B':b, 'T0':t0}
        if self.name == 'heat_capacity':
            raw = [s*q[:, k] + (m if k == 0 else 0) for k in range(self.cp_degree+1)]
            return {f'A{i}':sum(raw[j]*math.comb(j,i)*(-3)**(j-i)
                                for j in range(i,self.cp_degree+1)) for i in range(self.cp_degree+1)}
        if self.name == 'co2_solubility':
            return {'delta_h_J_per_mol':-8.314*s*q[:, 1]*300,
                    'C':m+s*(q[:, 0]-q[:, 1]*300/298.15)}
        slope = -F.softplus(q[:, 1]) if self.name == 'refractive_index' else q[:, 1]
        intercept = m+s*(q[:, 0]-slope*298.15/100)
        if self.name == 'density': return {'A':intercept,'B':s*slope/100,'C':s*q[:, 2]/100}
        if self.name == 'surface_tension': return {'a':s*slope/100,'b':intercept}
        if self.name == 'refractive_index': return {'a':intercept,'b':s*slope/100}
        return {'a1':intercept,'a2':s*slope/100}
