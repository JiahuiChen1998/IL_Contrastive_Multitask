"""One parameterized full-data fit with TabPFN V2's size safeguard disabled.

No fitting without --run. Keep legacy scientific source and output roots unchanged.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from il_contrastive.equations import study_tasks
from il_contrastive.pretraining import digest
from il_contrastive.tabular import feature_matrix, TrainPreprocessor
from il_contrastive.train import transformed
from scripts.run_tabular_benchmark import scores, write_json
from scripts.tabular_benchmark_archive import freeze

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('runs/tabpfn_override_feasibility_v1')
CHECKPOINT = Path('models/tabular/tabpfn/tabpfn-v2-regressor.ckpt')
CHECKPOINT_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--task', choices=['density','viscosity'], required=True)
    p.add_argument('--representation', choices=['descriptors','descriptors_fp'], required=True)
    p.add_argument('--seed', type=int, choices=range(42,47), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    OUT = a.output
    os.chdir(ROOT)
    torch.set_num_threads(4)
    frozen = freeze(Path('data/processed/observations.csv.gz'), Path('data/benchmarks/il_v1'))
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from the completed V2 baseline')
    frame = pd.read_csv('data/benchmarks/il_v1/observations.csv.gz', low_memory=False)
    frame = frame[frame.task == a.task].copy()
    job = dict(task=a.task, model='tabpfn', representation=a.representation, mode='raw', seed=a.seed)
    versions = {n: importlib.metadata.version(n) for n in ['tabpfn','torch','numpy','pandas','rdkit','scikit-learn']}
    source_files = [__file__, 'il_contrastive/tabular.py', 'il_contrastive/train.py', 'il_contrastive/equations.py',
                    'il_contrastive/tasks.py', 'scripts/run_tabular_benchmark.py']
    plan = dict(schema='tabpfn_override_completion_v1', jobs=[job], data_sha256=frozen['data_sha256'],
        subset_run=False, checkpoint_sha256=CHECKPOINT_SHA,
        settings=dict(ignore_pretraining_limits=True, version='V2', estimators=4, device='cuda',
                      threads=4, n_preprocessing_jobs=1, full_training_rows=True),
        counts=frame.groupby('split').size().to_dict(), package_versions=versions,
        source_sha256={str(Path(f).relative_to(ROOT) if Path(f).is_absolute() else Path(f)):digest(f) for f in source_files})
    OUT.mkdir(parents=True, exist_ok=True)
    lockfile = OUT / 'plan.json'
    if lockfile.exists() and json.loads(lockfile.read_text()) != plan:
        raise ValueError('Feasibility settings changed; use a new version')
    write_json(lockfile, plan)
    print(f'One {a.task}/{a.representation}/seed-{a.seed} job: {plan["counts"]}; override=True; checkpoint verified.', flush=True)
    if not a.run:
        print('Preparation only; no fit launched.', flush=True)
        return
    path = OUT / a.task / 'tabpfn' / a.representation / 'raw' / f'seed_{a.seed}'
    if (path / 'metrics.json').exists():
        print(f'Already complete: {path}/metrics.json', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Use a GPU-enabled terminal')
    path.mkdir(parents=True, exist_ok=True)
    spec = study_tasks()[a.task]
    masks = {s:(frame.split == s).to_numpy() for s in ['train','validation','test']}
    print('Preparing the same structural and T/P inputs as the raw baseline.', flush=True)
    x, names = feature_matrix(frame, spec, a.representation)
    prep = TrainPreprocessor().fit(x[masks['train']])
    x = prep.transform(x)
    y = transformed(frame.loc[masks['train'], 'value'], spec)
    mean = float(y.mean()); std = max(float(y.std()), 1e-6)
    os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(ROOT / 'models/tabular/tabpfn'))
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cuda', n_estimators=4,
        random_state=a.seed, n_preprocessing_jobs=1, ignore_pretraining_limits=True,
        model_path=str(CHECKPOINT.resolve()))
    started = time.time()
    print('Fitting full training context; one model process, no subsampling.', flush=True)
    model.fit(x[masks['train']], ((y-mean)/std).astype(np.float32))
    if digest(Path(model.model_path)) != CHECKPOINT_SHA:
        raise ValueError('Loaded checkpoint differs from the baseline')
    result = dict(**job, data_sha256=frozen['data_sha256'], subset_run=False, task_schema='ablation_v1',
        model_info=dict(version='V2', estimators=4, checkpoint=str(CHECKPOINT.resolve()),
                        checkpoint_sha256=CHECKPOINT_SHA, ignore_pretraining_limits=True),
        package_versions=versions, preprocessing=prep.state(), residual_scaler=dict(mean=mean,std=std),
        target_std=std, empirical_trend=None,
        feature_names=[n for n,k in zip(names,prep.keep) if k])
    for split in ['validation','test']:
        print(f'Predicting {split}.', flush=True)
        pred = np.asarray(model.predict(x[masks[split]])).reshape(-1)*std+mean
        f = frame.loc[masks[split]].copy()
        result[split], raw = scores(f, pred, spec, std)
        f['prediction_transformed'] = pred; f['prediction'] = raw
        f.to_csv(path / f'{split}_predictions.csv', index=False)
    result['elapsed_seconds'] = time.time()-started
    write_json(path / 'metrics.json', result)
    print(f'COMPLETE: {path}/metrics.json', flush=True)


if __name__ == '__main__':
    main()
