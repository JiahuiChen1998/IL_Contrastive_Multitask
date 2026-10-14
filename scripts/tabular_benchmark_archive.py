"""Freeze the benchmark data/splits and index all planned raw tabular results.

This command performs no fitting. Existing snapshots are verified, never replaced.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from il_mixer.equations import study_tasks
from il_mixer.pretraining import digest

KEY = ['smiles', 'task', 'source_file', 'source_sheet', 'source_row']
ROOTS = [
    'runs/tabular_classical_v1', 'runs/tabular_under500rows_v2',
    'runs/tabular_remaining_gpu_v2', 'runs/tabular_expansion_trees_v1_raw',
    'runs/tabular_foundation_completion_v1_raw',
]


def freeze(data, out):
    lock = out / 'manifest.json'
    if lock.exists():
        manifest = json.loads(lock.read_text())
        if digest(data) != manifest['data_sha256']:
            raise ValueError('Current data differ from frozen benchmark; use a new benchmark version')
        for name, sha in manifest['files'].items():
            if digest(out / name) != sha:
                raise ValueError(f'Frozen artifact changed: {name}')
        if json.loads((out / 'task_schema.json').read_text()) != study_tasks():
            raise ValueError('Task units/transforms changed from frozen benchmark')
        return manifest
    if out.exists() and any(out.iterdir()):
        raise ValueError('Incomplete snapshot exists; inspect it before creating a new snapshot')
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(data, low_memory=False)
    if f[KEY + ['split']].isna().any().any() or f.duplicated(KEY).any():
        raise ValueError('Missing or duplicate row identity')
    if set(f.split) != {'train', 'validation', 'test'} or f.groupby('smiles').split.nunique().max() != 1:
        raise ValueError('Invalid complete-structure split')
    shutil.copyfile(data, out / 'observations.csv.gz')
    rows = f[KEY + ['split']].copy()
    rows.insert(0, 'row_id', [hashlib.sha256(json.dumps(list(row), ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest() for row in f[KEY].itertuples(index=False, name=None)])
    if rows.row_id.duplicated().any():
        raise ValueError('Row identity collision')
    rows.to_csv(out / 'row_splits.csv.gz', index=False)
    f[['smiles', 'split']].drop_duplicates().sort_values('smiles').to_csv(out / 'structure_splits.csv', index=False)
    f.groupby(['task', 'split']).agg(rows=('smiles', 'size'), ils=('smiles', 'nunique')).to_csv(out / 'counts.csv')
    (out / 'task_schema.json').write_text(json.dumps(study_tasks(), indent=2))
    source_manifest = data.parent / 'manifest.json'
    if source_manifest.exists():
        shutil.copyfile(source_manifest, out / 'curation_manifest.json')
    files = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    manifest = dict(schema='il_benchmark_v1', data_sha256=digest(data), files=files,
        split_policy='One fixed complete-SMILES group split across all tasks; shared component ions allowed',
        fit_seeds=[42, 43, 44], selection='Mean validation standardized MSE; never test selection',
        row_identity=KEY, primary_metric='ln-viscosity MAE for viscosity; native target MAE otherwise',
        uncertainty='Report seed SD and paired whole-IL bootstrap; also equal-IL weighting')
    lock.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def index_results(roots, snapshot, out, manifest, seeds=(42, 43, 44)):
    out.mkdir(parents=True, exist_ok=True)
    registry = out / 'artifact_registry.json'
    previous = {r['path']: r for r in json.loads(registry.read_text())} if registry.exists() else {}
    jobs = {}; provenance = []; scores = []
    expected_rows = pd.read_csv(snapshot / 'observations.csv.gz', low_memory=False)
    for root_name in roots:
        root = Path(root_name)
        if not (root / 'plan.json').exists():
            continue
        plan = json.loads((root / 'plan.json').read_text())
        if plan.get('subset_run') or plan.get('settings', {}).get('smoke') or plan['data_sha256'] != manifest['data_sha256']:
            raise ValueError(f'Incompatible plan: {root}')
        failures = json.loads((root / 'failures.json').read_text()) if (root / 'failures.json').exists() else []
        for job in plan['jobs']:
            if job.get('mode', 'raw') != 'raw':
                continue
            key = (job['task'], job['model'], job['representation'], job['seed'])
            if key in jobs:
                raise ValueError(f'Duplicate experiment: {key}')
            path = root / job['task'] / job['model'] / job['representation']
            if 'mode' in job:
                path /= job['mode']
            path /= f"seed_{job['seed']}"
            entry = dict(task=key[0], model=key[1], representation=key[2], seed=key[3],
                         mode='raw', path=str(path.resolve()), status='pending')
            match = [x for x in failures if all(x.get(k) == job.get(k) for k in ['task', 'model', 'representation', 'seed'])]
            if match:
                entry['status'] = match[-1]['status']
            if (path / 'metrics.json').exists():
                metrics = json.loads((path / 'metrics.json').read_text())
                if metrics.get('subset_run') or metrics['data_sha256'] != manifest['data_sha256']:
                    raise ValueError(f'Incompatible metrics: {path}')
                hashes = {'metrics.json': digest(path / 'metrics.json')}
                for split in ['validation', 'test']:
                    pred = path / f'{split}_predictions.csv'
                    f = pd.read_csv(pred, low_memory=False)
                    expected = expected_rows[(expected_rows.task == key[0]) & (expected_rows.split == split)]
                    # Match identities, targets and conditions, independent of CSV row order.
                    cols = KEY + [c for c in ['value', 'temperature_K', 'pressure_input', 'split'] if c in expected]
                    pd.testing.assert_frame_equal(f[cols].sort_values(KEY).reset_index(drop=True),
                        expected[cols].sort_values(KEY).reset_index(drop=True), check_dtype=False,
                        check_exact=False, rtol=1e-12, atol=1e-12)
                    hashes[pred.name] = digest(pred)
                    scores.append(dict(**entry, split=split, **metrics[split]))
                entry['status'] = 'completed'
                prior = previous.get(entry['path'])
                if prior and (prior['artifact_sha256'] != hashes or prior['plan_sha256'] != digest(root / 'plan.json')):
                    raise ValueError(f'Previously registered result changed: {path}; use a new experiment directory')
                provenance.append(dict(**entry, data_sha256=manifest['data_sha256'],
                    plan_sha256=digest(root / 'plan.json'), artifact_sha256=hashes,
                    model_info=metrics.get('model_info'), package_versions=metrics.get('package_versions'),
                    elapsed_seconds=metrics.get('elapsed_seconds')))
            jobs[key] = entry
    coverage = []
    for task in study_tasks():
        for model in ['extratrees', 'tabpfn', 'limix']:
            for rep in ['descriptors', 'descriptors_fp']:
                for seed in seeds:
                    coverage.append(jobs.get((task, model, rep, seed), dict(task=task, model=model,
                        representation=rep, seed=seed, mode='raw', status='unplanned', path='')))
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    # Status belongs to coverage, not metric values.
    pd.DataFrame(scores).drop(columns=['status'], errors='ignore').to_csv(out / 'per_seed_statistics.csv', index=False)
    (out / 'artifact_registry.json').write_text(json.dumps(provenance, indent=2) + '\n')
    counts = pd.Series([x['status'] for x in coverage]).value_counts().to_dict()
    (out / 'STATUS.md').write_text('# Three-model raw benchmark\n\n'
        + f'{66 * len(seeds)} expected fits (11 properties × 3 models × 2 representations × {len(seeds)} seeds). Status: {counts}.\n\n'
        + f'Frozen data SHA-256: `{manifest["data_sha256"]}`.\n\n'
        + 'Completed prediction identities, targets and conditions were checked against the frozen split. '
        + 'Failed or pending fits are not scores. Per-seed statistics include both validation and test; select on validation only.\n')
    print(f'Frozen split verified. Raw coverage: {counts}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/processed/observations.csv.gz'))
    p.add_argument('--snapshot', type=Path, default=Path('data/benchmarks/il_v1'))
    p.add_argument('--output', type=Path, default=Path('reports/tabular_three_models_v1'))
    p.add_argument('--roots', nargs='+', default=ROOTS)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--seeds', default='42,43,44')
    a = p.parse_args()
    manifest = freeze(a.data, a.snapshot)
    if not a.verify_only:
        seeds = list(map(int, a.seeds.split(',')))
        if len(seeds) != len(set(seeds)) or not seeds:
            p.error('Seeds must be unique and nonempty')
        index_results(a.roots, a.snapshot, a.output, manifest, seeds)
    else:
        print('Frozen data, split artifacts and task schema verified.')


if __name__ == '__main__':
    main()
