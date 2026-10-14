"""One fit with fixed task sampling and application-specific checkpoint selection."""
import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import torch

from il_mixer.pretraining import load_encoder
from il_mixer.rdkit_logging import suppress_isolated_hydrogen_warning
from il_mixer.train import batch, evaluate, fit_scalers, transformed
from scripts.pretrain_full_coverage import atomic_json
from il_mixer.model import ILMixer
from .protocol import DATA, PRIMARY, verify_inputs


def weighted_loss(outputs, frame, scalers, tasks, weights, device):
    terms, mass = [], 0.
    for task in frame.task.unique():
        mask = frame.task.to_numpy() == task
        scale = scalers[task]
        y = (transformed(frame.loc[mask, 'value'], tasks[task])-scale['mean'])/scale['std']
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        err = (outputs[task][torch.as_tensor(mask, device=device)]-y).square().mean()
        terms.append(weights[task]*err)
        mass += weights[task]
    if mass <= 0:
        raise ValueError('No observed weighted labels')
    return torch.stack(terms).sum()/mass


def selection_score(metrics):
    return float(np.mean([metrics[t]['standardized_mse'] for t in PRIMARY]))


def fit(plan, job):
    verify_inputs(plan)
    s, tasks = plan['settings'], plan['tasks']
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(s['threads'])
    torch.manual_seed(job['seed'])
    rng = np.random.default_rng(job['seed'])
    device = s['device']
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use CPU only for isolated smoke verification')
        torch.cuda.set_per_process_memory_fraction(.7)
    frame = pd.read_csv(DATA, low_memory=False)
    if frame.groupby('smiles').split.nunique().max() != 1:
        raise ValueError('Global complete-IL split changed')
    if s['limit_per_task']:
        frame = pd.concat([g.sample(min(s['limit_per_task'], len(g)), random_state=job['seed'])
                           for _, g in frame.groupby(['split','task'], sort=False)])
    splits = {split:frame[frame.split.eq(split)].reset_index(drop=True) for split in ['train','validation','test']}
    for task, spec in tasks.items():
        for col, needed in [('temperature_K', spec['temperature']), ('pressure_input', spec['pressure'])]:
            if needed and not np.isfinite(frame.loc[frame.task.eq(task), col]).all():
                raise ValueError(f'Missing condition: {task}/{col}')
    scalers = fit_scalers(splits['train'], tasks)
    config = {k:s[k] for k in ['hidden','layers','heads','cross_ion','head_mode']}
    config.update(tasks=tasks)
    model = ILMixer(**config).to(device)
    pretraining = load_encoder(model, plan['checkpoint'], include_cross=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
    groups = {t:g.index.to_numpy() for t,g in splits['train'].groupby('task')}
    atomic_json(dict(tasks=tasks, weights=job['weights'], selection_tasks=PRIMARY, config=config), output/'task_config.json')
    best, state, stale, history, best_epoch = float('inf'), None, 0, [], None
    started = time.time()
    for epoch in range(s['epochs']):
        model.train()
        losses = []
        for _ in range(s['steps_per_epoch']):
            ids = np.concatenate([rng.choice(v, s['per_task_batch'], replace=len(v)<s['per_task_batch']) for v in groups.values()])
            sub = splits['train'].iloc[ids]
            optimizer.zero_grad()
            loss = weighted_loss(model(*batch(sub, device)), sub, scalers, tasks, job['weights'], device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s['gradient_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
        valid, _ = evaluate(model, splits['validation'], scalers, tasks, device)
        score = selection_score(valid)
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite application validation score')
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_loss=score,
                            all_task_validation_loss=float(np.mean([v['standardized_mse'] for v in valid.values()])),
                            per_task_validation_mse={t:v['standardized_mse'] for t,v in valid.items()}))
        print(json.dumps(history[-1]), flush=True)
        if score < best:
            best, state, stale, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch+1
        else:
            stale += 1
        if stale >= s['patience']:
            break
    model.load_state_dict(state)
    arguments = dict(settings=s, seed=job['seed'], arm=job['arm'], weights=job['weights'])
    torch.save(dict(model_state=state, config=config, scalers=scalers, arguments=arguments,
                    pretraining=pretraining, data_sha256=plan['data_sha256'], source_sha256=plan['source_sha256']), output/'best.pt')
    metrics = dict(arguments=arguments, data_sha256=plan['data_sha256'], source_sha256=plan['source_sha256'],
                   subset_run=plan['smoke'], pretraining=pretraining, history=history, validation_best=best,
                   best_epoch=best_epoch, epochs=len(history), reached_epoch_cap=len(history)==s['epochs'],
                   trainable_parameters=sum(p.numel() for p in model.parameters()), fitting_wall_seconds=time.time()-started)
    for split in ['validation','test']:
        m, predictions = evaluate(model, splits[split], scalers, tasks, device)
        predictions.to_csv(output/f'{split}_predictions.csv', index=False)
        metrics[split] = m
    if not np.isclose(selection_score(metrics['validation']), best, rtol=1e-7):
        raise ValueError('Reloaded best checkpoint does not reproduce selection score')
    atomic_json(metrics, output/'metrics.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--job', required=True)
    args = p.parse_args()
    plan = json.loads((Path(args.root)/'plan.json').read_text())
    job = next(j for j in plan['jobs'] if j['id']==args.job)
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output/'execution.json').exists() and json.loads((output/'execution.json').read_text()).get('state')=='complete':
        raise ValueError('Refusing to overwrite a completed fit')
    suppress_isolated_hydrogen_warning()
    started = time.time()
    execution = dict(state='running', started_unix=started, source_sha256=plan['source_sha256'],
                     versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__))
    atomic_json(execution, output/'execution.json')
    try:
        fit(plan, job)
        execution['state'] = 'complete'
    except BaseException as error:
        execution.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        execution.update(finished_unix=time.time(), wall_seconds=time.time()-started,
                         peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                         gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else 0)
        atomic_json(execution, output/'execution.json')


if __name__ == '__main__':
    main()
