"""Immutable 90/1 specialist experiment and pinned historical comparators."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from il_contrastive.equations import study_tasks
from il_contrastive.pretraining import digest
from scripts import run_multitask_heads_study as legacy

ROOT = legacy.ROOT
OUT = Path('runs/property_specialists_30m_v1')
DATA, CHECKPOINT = legacy.DATA, legacy.CHECKPOINT
TAB = Path('reports/tabular_final_v2')
SEEDS = list(range(42, 47))
ARMS = {'priority90': .9, 'uniform': 1/11, 'target_only': 1.}
FILES = ['best.pt', 'metrics.json', 'task_config.json', 'validation_predictions.csv',
         'test_predictions.csv', 'execution.json']


def loss_weights(target, arm):
    tasks = study_tasks()
    if target not in tasks or arm not in ARMS:
        raise ValueError('Unknown target or arm')
    weight = ARMS[arm]
    return {t: weight if t == target else (1-weight)/(len(tasks)-1) for t in tasks}


def fingerprint():
    files = [Path(__file__), Path(__file__).with_name('train.py')]
    return hashlib.sha256(legacy.training_fingerprint().encode() + b''.join(p.read_bytes() for p in files)).hexdigest()


def choose_tabular(frame):
    """Freeze validation choice first; test minimum is a descriptive reference only."""
    keys = ['task', 'model', 'representation']
    if frame.duplicated(keys + ['seed', 'split']).any():
        raise ValueError('Duplicate tabular cells')
    groups = frame.groupby(keys + ['split'])
    if len(frame) != 660 or not groups.seed.apply(lambda s: sorted(s) == SEEDS).all():
        raise ValueError('Expected all 330 tabular fits with five seeds and both splits')
    if set(frame.model) != {'extratrees', 'tabpfn', 'limix'} or set(frame.task) != set(study_tasks()):
        raise ValueError('Tabular family/task coverage changed')
    if set(frame.representation) != {'descriptors', 'descriptors_fp'} or set(frame.split) != {'validation', 'test'}:
        raise ValueError('Tabular representation/split coverage changed')
    means = groups[['standardized_mse', 'primary_mae']].mean().reset_index()
    validation = means[means.split.eq('validation')].sort_values(
        ['task', 'standardized_mse', 'model', 'representation']).groupby('task').head(1)
    test_floor = means[means.split.eq('test')].sort_values(
        ['task', 'primary_mae', 'model', 'representation']).groupby('task').head(1)
    return {str(t): {'validation_selected': validation.set_index('task').loc[t, ['model', 'representation']].to_dict(),
                    'observed_test_minimum': test_floor.set_index('task').loc[t, ['model', 'representation']].to_dict()}
            for t in study_tasks()}


def make_plan(output=OUT, smoke=False):
    output = Path(output)
    parent = json.loads((legacy.OUT/'plan.json').read_text())
    if parent['source_sha256'] != legacy.training_fingerprint():
        raise ValueError('Original scientific source changed')
    if digest(DATA) != parent['data_sha256'] or digest(CHECKPOINT) != parent['checkpoint_sha256']:
        raise ValueError('Frozen data or pretraining checkpoint changed')
    tasks = study_tasks()
    if tasks != json.loads((DATA.parent/'task_schema.json').read_text()):
        raise ValueError('Frozen task schema differs')
    targets = ['co2_solubility'] if smoke else list(tasks)
    seeds = [42] if smoke else SEEDS
    settings = dict(hidden=64, layers=3, heads=4, cross_ion=True, head_mode='mlp',
                    epochs=1 if smoke else 200, steps_per_epoch=2 if smoke else 100,
                    per_task_batch=2 if smoke else 8, lr=.0005, weight_decay=.0001,
                    patience=15, threads=4, gradient_clip=5., device='cpu' if smoke else 'cuda',
                    limit_per_task=8 if smoke else 0, task_schema='ablation_v1')
    jobs = []
    for target in targets:
        for seed in seeds:
            for arm in ARMS:
                ident = f'{target}/{seed}/{arm}'
                jobs.append(dict(id=ident, target=target, seed=seed, arm=arm, weights=loss_weights(target, arm),
                                 output=str(output/'fits'/target/f'seed_{seed}'/arm),
                                 command=[sys.executable, '-u', '-m', 'scripts.property_specialists.train',
                                          '--root', str(output), '--job', ident]))
    references, tabular, choices = [], [], {}
    if not smoke:
        for j in parent['jobs']:
            if j['initialization'] != 'pretrained_30m':
                continue
            multi = j['scope'] == 'multi' and j['head_family'] == 'mlp'
            single = j['scope'] == 'single' and 'mlp' in j['head_aliases']
            if not (multi or single):
                continue
            if not legacy.verify_fit(parent, j):
                raise ValueError(f'Missing historical reference: {j["id"]}')
            path = Path(j['output'])
            files = legacy.ARTIFACTS + (['execution.json'] if (path/'execution.json').exists() else [])
            references.append(dict(id=j['id'], arm='historical_multi' if multi else 'historical_single',
                                   seed=j['seed'], tasks=j['tasks'], output=str(path),
                                   hashes={name: digest(path/name) for name in files}))
        if len(references) != 60:
            raise ValueError('Expected five historical MTL and 55 single-task references')
        frame = pd.read_csv(TAB/'per_seed_statistics.csv')
        choices = choose_tabular(frame)
        registry = {str(Path(x['path']).resolve()): x for x in json.loads((TAB/'artifact_registry.json').read_text())}
        for row in frame[frame.split.eq('validation')].to_dict('records'):
            path = Path(row['path'])
            reg = registry[str(path.resolve())]
            hashes = reg['artifact_sha256']
            if any(digest(path/name) != value for name, value in hashes.items()):
                raise ValueError(f'Tabular artifact changed: {path}')
            tabular.append(dict(id=f'tabular/{row["task"]}/{row["model"]}/{row["representation"]}/{row["seed"]}',
                                task=row['task'], seed=int(row['seed']), model=row['model'],
                                representation=row['representation'], output=str(path), hashes=hashes))
    inputs = {str(p): digest(p) for p in [DATA, DATA.parent/'manifest.json', DATA.parent/'structure_splits.csv',
              DATA.parent/'task_schema.json', CHECKPOINT, legacy.OUT/'plan.json']}
    if not smoke:
        inputs.update({str(TAB/p): digest(TAB/p) for p in ['per_seed_statistics.csv', 'artifact_registry.json']})
    return dict(schema='property_specialists_30m_v1', smoke=smoke, tasks=tasks, targets=targets, seeds=seeds,
                settings=settings, arms=ARMS, jobs=jobs, references=references, tabular=tabular,
                tabular_choices=choices, inputs=inputs, data_sha256=digest(DATA), checkpoint=str(CHECKPOINT),
                checkpoint_sha256=digest(CHECKPOINT), source_sha256=fingerprint(),
                selection='Each new fit selects and early-stops on its target validation standardized MSE only.',
                protocol='Eleven target-focused plain MLPs, each with 90% target / 1% per auxiliary loss. '
                'Uniform and 100%-target controls share all eleven heads, initialization, balanced batches and settings. '
                'One shared best checkpoint per fit; full encoder fine-tuning; no new task attention. '
                'Five seeds on the frozen complete-IL split. Prespecified 200-epoch cap for all new arms, patience 15. '
                'Historical 100-epoch single-task and all-task-selected MTL are contextual references. '
                'Tabular primary comparator: lowest five-seed validation standardized MSE across all six configurations. '
                'Lowest historical five-seed test MAE is a separate descriptive benchmark, never a deployment nomination. '
                'No best-seed selection. Native CO2 MAE, ln-viscosity MAE; preserve both target spaces. '
                'Previously inspected test split: exploratory comparison, not independent confirmation.')


def verify_inputs(plan):
    for path, expected in plan['inputs'].items():
        if digest(path) != expected:
            raise ValueError(f'Changed immutable input: {path}')
    if fingerprint() != plan['source_sha256']:
        raise ValueError('Scientific source changed; use a separately versioned plan')


def verify_fit(plan, job):
    path = Path(job['output'])
    if not (path/'execution.json').exists():
        return False
    execution = json.loads((path/'execution.json').read_text())
    if execution['state'] != 'complete':
        return False
    if not all((path/name).exists() for name in FILES):
        raise ValueError(f'Incomplete completed artifacts: {path}')
    m = json.loads((path/'metrics.json').read_text())
    expected = dict(settings=plan['settings'], seed=job['seed'], arm=job['arm'], target=job['target'], weights=job['weights'])
    if m['arguments'] != expected or m['source_sha256'] != plan['source_sha256']:
        raise ValueError(f'Configuration/source mismatch: {path}')
    if execution['source_sha256'] != plan['source_sha256'] or m['data_sha256'] != plan['data_sha256']:
        raise ValueError(f'Execution/data mismatch: {path}')
    if bool(m['subset_run']) != plan['smoke']:
        raise ValueError(f'Smoke contamination: {path}')
    if m['pretraining']['sha256'] != plan['checkpoint_sha256'] or m['pretraining']['include_cross']:
        raise ValueError(f'Initialization mismatch: {path}')
    config = {k: plan['settings'][k] for k in ['hidden', 'layers', 'heads', 'cross_ion', 'head_mode']}
    config['tasks'] = plan['tasks']
    expected_cfg = dict(tasks=plan['tasks'], weights=job['weights'], selection_tasks=[job['target']], config=config)
    if json.loads((path/'task_config.json').read_text()) != expected_cfg or m['trainable_parameters'] != 384803:
        raise ValueError(f'Model/selection mismatch: {path}')
    return True
