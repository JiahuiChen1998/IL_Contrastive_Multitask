"""Prespecified condition bases and a train-only empirical trend for tabular models."""
import numpy as np
from sklearn.linear_model import Ridge


def condition_basis(frame, spec, task):
    n = len(frame); cols = []; names = []
    if spec['temperature']:
        t = frame.temperature_K.to_numpy(float)
        if not np.isfinite(t).all() or (t <= 0).any(): raise ValueError('Finite positive Kelvin required')
        u = (t - 298.15)/100
        if task in ['viscosity', 'co2_solubility']:
            cols += [300/t - 300/298.15]; names += ['centered_inverse_T']
            if task == 'viscosity':
                # Fixed-T0 VFT basis: this is not a fitted per-IL VFT coefficient model.
                if (t < 150).any(): raise ValueError('Viscosity basis requires T>=150 K')
                cols += [300/(t-100)-300/(298.15-100)]; names += ['fixed_T0_100K_vft']
        else:
            cols += [u]; names += ['centered_T']
        if task == 'heat_capacity':
            cols += [u**2, u**3]; names += ['centered_T_squared','centered_T_cubed']
    if spec['pressure']:
        pressure = frame.pressure_input.to_numpy(float)
        if not np.isfinite(pressure).all() or (pressure < 0).any(): raise ValueError('Finite nonnegative pressure required')
        cols += [pressure]; names += ['pressure_native']
        if task == 'co2_solubility':
            logp = np.log1p(pressure)  # MPa; does not impose Henry's-law limit.
            cols += [logp, logp*(300/t-300/298.15)]
            names += ['log1p_pressure_MPa', 'pressure_inverse_T_interaction']
    return (np.column_stack(cols) if cols else np.empty((n,0))), names


class EmpiricalTrend:
    """Global regularized condition trend in transformed-target units.

    All fitting uses training rows. Residuals are ordinary in-sample stagewise
    residuals, not out-of-fold predictions or per-IL fitted thermodynamic labels.
    """
    def fit(self, basis, y):
        if not basis.shape[1]: raise ValueError('Empirical trend requires condition features')
        self.mean = basis.mean(0); self.scale = basis.std(0)
        self.scale[self.scale < 1e-12] = 1.
        self.model = Ridge(alpha=1.).fit((basis-self.mean)/self.scale, y)
        return self
    def predict(self, basis):
        return self.model.predict((basis-self.mean)/self.scale)
    def state(self):
        return dict(mean=self.mean.tolist(),scale=self.scale.tolist(),coefficients=self.model.coef_.tolist(),
                    intercept=float(self.model.intercept_),alpha=1.,fit='training rows only')


def condition_domains(train, query, spec):
    flags = {}
    for col, required in [('temperature_K',spec['temperature']),('pressure_input',spec['pressure'])]:
        if required:
            low, high = float(train[col].min()), float(train[col].max())
            flags[col+'_outside_train_range'] = (query[col].to_numpy()<low)|(query[col].to_numpy()>high)
    return flags
