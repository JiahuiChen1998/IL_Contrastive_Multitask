"""Task-balanced training, train-only target scaling, and held-out evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .graph import pack
from .model import ILMixer
from .tasks import TASKS


def transformed(values, spec):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (spec['transform'] == 'log' and (values <= 0).any()):
        raise ValueError('Target must be finite and strictly positive when log-transformed')
    return np.log(values) if spec['transform'] == 'log' else values


def fit_scalers(frame, tasks):
    scalers = {}
    for name, spec in tasks.items():
        values = transformed(frame.loc[frame.task == name, 'value'], spec)
        if not len(values): raise ValueError(f'No training labels for {name}')
        scalers[name] = dict(mean=float(values.mean()), std=max(float(values.std()), 1e-6))
    return scalers


def batch(frame, device):
    # Placeholders only for tasks without these conditions; they are ignored by those heads.
    return (pack(frame.smiles.tolist(), device),
            torch.tensor(frame.temperature_K.fillna(298.15).to_numpy(), dtype=torch.float32, device=device),
            torch.tensor(frame.pressure_input.fillna(0).to_numpy(), dtype=torch.float32, device=device), frame.task.tolist())


def loss_for(predictions, frame, scalers, tasks, device):
    losses = []
    for name in frame.task.unique():
        mask = frame.task.to_numpy() == name
        y = transformed(frame.loc[mask, 'value'], tasks[name])
        s = scalers[name]
        target = torch.tensor((y - s['mean']) / s['std'], dtype=torch.float32, device=device)
        losses.append((predictions[name][torch.as_tensor(mask, device=device)] - target).square().mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, frame, scalers, tasks, device='cpu', batch_size=128):
    model.eval()
    result = frame.copy()
    predicted = []
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start:start + batch_size]
        outputs = model(*batch(sub, device))
        for i, name in enumerate(sub.task):
            s = scalers[name]
            predicted.append(float(outputs[name][i]) * s['std'] + s['mean'])
    result['prediction_transformed'] = predicted
    result['prediction'] = [np.exp(p) if tasks[n]['transform'] == 'log' else p
                            for p, n in zip(predicted, frame.task)]
    metrics = {}
    for name, sub in result.groupby('task'):
        y = sub.value.to_numpy(); p = sub.prediction.to_numpy()
        yt = transformed(y, tasks[name]); pt = sub.prediction_transformed.to_numpy()
        residual = y - p
        total = float(((y - y.mean())**2).sum())
        metrics[name] = dict(n=len(sub), mae=float(np.abs(residual).mean()),
                             rmse=float(np.sqrt((residual**2).mean())),
                             r2=None if total == 0 else float(1 - (residual**2).sum() / total),
                             transformed_mae=float(np.abs(yt - pt).mean()),
                             standardized_mse=float(((yt - pt)**2).mean() / scalers[name]['std']**2),
                             train_mean_baseline_transformed_mae=float(np.abs(yt - scalers[name]['mean']).mean()),
                             unit=tasks[name]['unit'])
    return metrics, result


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    frame = pd.read_csv(args.data, low_memory=False)
    selected = args.tasks.split(',') if args.tasks else list(TASKS)
    from .equations import study_tasks, choose_cp_degree
    registry = study_tasks() if args.task_schema == 'ablation_v1' else TASKS
    if args.head_mode == 'equations' and args.task_schema != 'ablation_v1':
        raise ValueError('Equation heads require --task-schema ablation_v1 for matched target spaces')
    tasks = {n: registry[n] for n in selected}
    frame = frame[frame.task.isin(tasks)].copy()
    if frame.groupby('smiles').split.nunique().max() > 1:
        raise ValueError('Ion-pair leakage across splits')
    for name, spec in tasks.items():
        sub = frame[frame.task == name]
        for col, needed in [('temperature_K', spec['temperature']), ('pressure_input', spec['pressure'])]:
            if needed and not np.isfinite(sub[col]).all(): raise ValueError(f'Missing {col} for {name}')
        if set(sub.split) != {'train', 'validation', 'test'}:
            raise ValueError(f'{name} needs train, validation and test observations')
    splits = {s: frame[frame.split == s].reset_index(drop=True) for s in ['train', 'validation', 'test']}
    if args.limit_per_task:
        splits = {s: pd.concat([x.sample(min(args.limit_per_task, len(x)), random_state=args.seed)
                                for _, x in f.groupby('task')]).reset_index(drop=True)
                  for s, f in splits.items()}
    scalers = fit_scalers(splits['train'], tasks)
    cp_degree, cp_audit = choose_cp_degree(splits['train'])
    if args.cp_degree != 'auto': cp_degree = int(args.cp_degree)
    config = dict(tasks=tasks, hidden=args.hidden, layers=args.layers, cross_ion=args.cross_ion,
                  head_mode=args.head_mode)
    if args.head_mode == 'equations':
        config.update(target_scalers=scalers, cp_degree=cp_degree)
    model = ILMixer(**config).to(device)
    pretraining = None
    if getattr(args, 'pretrained_encoder', None):
        from .pretraining import load_encoder
        pretraining = load_encoder(model, args.pretrained_encoder,
                                   include_cross=getattr(args, 'pretrained_cross', False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'task_config.json').write_text(json.dumps(dict(tasks=tasks, cp_degree=cp_degree, cp_audit=cp_audit), indent=2))
    groups = {n: f.index.to_numpy() for n, f in splits['train'].groupby('task')}
    best, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for _ in range(args.steps_per_epoch):
            # Equal labels per task per optimizer step prevents density dominating small tasks.
            ids = np.concatenate([rng.choice(v, args.per_task_batch, replace=len(v) < args.per_task_batch)
                                  for v in groups.values()])
            sub = splits['train'].iloc[ids]
            optimizer.zero_grad()
            loss = loss_for(model(*batch(sub, device)), sub, scalers, tasks, device)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        valid, _ = evaluate(model, splits['validation'], scalers, tasks, device)
        score = float(np.mean([m['standardized_mse'] for m in valid.values()]))
        if not np.isfinite(score): raise FloatingPointError('Non-finite validation score')
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=score))
        print(json.dumps(history[-1]), flush=True)
        if score < best:
            best, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= args.patience: break
    model.load_state_dict(best_state)
    checkpoint = dict(model_state=best_state, config=config, scalers=scalers, arguments=vars(args), pretraining=pretraining,
                      data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    torch.save(checkpoint, output / 'best.pt')
    summary = dict(validation_best=best, history=history, subset_run=bool(args.limit_per_task), pretraining=pretraining,
                   arguments=vars(args), data_sha256=checkpoint['data_sha256'], task_schema=args.task_schema,
                   cp_degree=cp_degree, trainable_parameters=sum(p.numel() for p in model.parameters()))
    for split in ['validation', 'test']:
        metrics, predictions = evaluate(model, splits[split], scalers, tasks, device)
        summary[split] = metrics
        predictions.to_csv(output / f'{split}_predictions.csv', index=False)
    metrics_temp = output / 'metrics.json.tmp'
    metrics_temp.write_text(json.dumps(summary, indent=2, allow_nan=False))
    metrics_temp.replace(output / 'metrics.json')
    (output / 'split_counts.json').write_text(json.dumps({s: f.task.value_counts().to_dict() for s, f in splits.items()}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/observations.csv.gz')
    parser.add_argument('--output', default='runs/baseline')
    parser.add_argument('--tasks')
    parser.add_argument('--cross-ion', action='store_true')
    parser.add_argument('--pretrained-encoder', help='ILMixer SSL checkpoint; transfers trunk, reinitializes property heads')
    parser.add_argument('--pretrained-cross', action='store_true', help='Also transfer pretrained cross-ion attention (separate ablation)')
    parser.add_argument('--head-mode', choices=['physics', 'mlp', 'equations'], default='physics')
    parser.add_argument('--task-schema', choices=['legacy', 'ablation_v1'], default='legacy')
    parser.add_argument('--cp-degree', choices=['auto','1','2','3'], default='auto')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
    parser.add_argument('--per-task-batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--limit-per-task', type=int, default=0, help='Smoke test only; subsample each split/task')
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.per_task_batch, args.hidden, args.layers) < 1:
        parser.error('Training dimensions and counts must be positive')
    train(args)
