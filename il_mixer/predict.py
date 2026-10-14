"""Predict long-form CSV rows using saved model and train-only scalers."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import ILMixer
from .train import batch


def predict(checkpoint, frame):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ILMixer(**saved['config']); model.load_state_dict(saved['model_state']); model.eval()
    tasks = saved['config']['tasks']
    frame = frame.copy()
    if not {'smiles', 'task'} <= set(frame): raise ValueError('CSV needs smiles and task columns')
    for col in ['temperature_K', 'pressure_input']:
        if col not in frame: frame[col] = np.nan
    for name, sub in frame.groupby('task'):
        if name not in tasks: raise ValueError(f'Untrained task: {name}')
        spec = tasks[name]
        for col, required in [('temperature_K', spec['temperature']), ('pressure_input', spec['pressure'])]:
            if required and not np.isfinite(sub[col]).all(): raise ValueError(f'{name} requires finite {col}')
        if spec['temperature'] and (sub.temperature_K <= 0).any(): raise ValueError('T must be positive Kelvin')
        if name == 'viscosity' and model.head_mode in ('physics', 'equations') and (sub.temperature_K < 150).any():
            raise ValueError('Viscosity requires T >=150 K')
        if spec['pressure'] and (sub.pressure_input < 0).any(): raise ValueError('Pressure must be nonnegative')
    predictions = []
    with torch.no_grad():
        for start in range(0, len(frame), 128):
            sub = frame.iloc[start:start + 128]
            outputs = model(*batch(sub, 'cpu'))
            for i, name in enumerate(sub.task):
                scale = saved['scalers'][name]
                p = float(outputs[name][i]) * scale['std'] + scale['mean']
                predictions.append(np.exp(p) if tasks[name]['transform'] == 'log' else p)
    frame['prediction'] = predictions
    frame['unit'] = [tasks[n]['unit'] for n in frame.task]
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    predict(args.checkpoint, pd.read_csv(args.input)).to_csv(output, index=False)
