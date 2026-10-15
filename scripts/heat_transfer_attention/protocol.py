"""Prespecified arms, immutable scientific inputs and completed-fit verification."""
import hashlib
import json
from pathlib import Path
import sys

from il_contrastive.equations import study_tasks
from il_contrastive.pretraining import digest
from scripts import run_multitask_heads_study as legacy

ROOT = legacy.ROOT
OUT = Path('runs/heat_transfer_attention_30m_v1')
DATA = legacy.DATA
CHECKPOINT = legacy.CHECKPOINT
PRIMARY = ['melting_point', 'glass_transition', 'decomposition_temperature',
           'heat_capacity', 'thermal_conductivity', 'viscosity']
SEEDS = list(range(42, 47))
ARMS = {
    'uniform_plain': dict(feature_mode='plain', priority='uniform'),
    'weighted_plain': dict(feature_mode='plain', priority='heat90'),
    'uniform_attention': dict(feature_mode='attention', priority='uniform'),
    'weighted_attention': dict(feature_mode='attention', priority='heat90'),
    'weighted_adapter': dict(feature_mode='adapter', priority='heat90'),
}
FILES = ['best.pt', 'metrics.json', 'task_config.json', 'validation_predictions.csv',
         'test_predictions.csv', 'execution.json', 'feature_diagnostics.csv']


def loss_weights(priority, tasks=None):
    tasks = list(study_tasks()) if tasks is None else list(tasks)
    if set(PRIMARY) - set(tasks):
        raise ValueError('All six primary properties are required')
    if priority == 'uniform':
        return {task: 1/len(tasks) for task in tasks}
    if priority != 'heat90' or len(tasks) <= len(PRIMARY):
        raise ValueError('Unknown priority or missing auxiliary tasks')
    return {task: .9/len(PRIMARY) if task in PRIMARY else .1/(len(tasks)-len(PRIMARY)) for task in tasks}


def fingerprint():
    files = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('train.py')]
    return hashlib.sha256(legacy.training_fingerprint().encode() + b''.join(p.read_bytes() for p in files)).hexdigest()


def make_plan(output=OUT, smoke=False):
    output = Path(output)
    parent = json.loads((legacy.OUT/'plan.json').read_text())
    if parent['source_sha256'] != legacy.training_fingerprint():
        raise ValueError('Original scientific source changed')
    if digest(DATA) != parent['data_sha256'] or digest(CHECKPOINT) != parent['checkpoint_sha256']:
        raise ValueError('Frozen data or 30.2M checkpoint changed')
    settings = dict(hidden=64, layers=3, heads=4, cross_ion=True, head_mode='mlp', adapter_width=32,
                    epochs=1 if smoke else 100, steps_per_epoch=2 if smoke else 100,
                    per_task_batch=2 if smoke else 8, lr=.0005, weight_decay=.0001, patience=15,
                    threads=4, gradient_clip=5., device='cpu' if smoke else 'cuda',
                    limit_per_task=8 if smoke else 0, task_schema='ablation_v1')
    seeds = [42] if smoke else SEEDS
    jobs = []
    for seed in seeds:
        for arm, design in ARMS.items():
            path = output/'fits'/f'seed_{seed}'/arm
            jobs.append(dict(id=f'{seed}/{arm}', seed=seed, arm=arm, output=str(path),
                             **design, weights=loss_weights(design['priority']),
                             command=[sys.executable, '-u', '-m', 'scripts.heat_transfer_attention.train',
                                      '--root', str(output), '--job', f'{seed}/{arm}']))
    refs = []
    if not smoke:
        for job in parent['jobs']:
            if job['initialization'] != 'pretrained_30m':
                continue
            historical = job['scope'] == 'multi' and job['head_family'] == 'mlp'
            single = job['scope'] == 'single' and job['tasks'][0] in PRIMARY and 'mlp' in job['head_aliases']
            if not (historical or single):
                continue
            if not legacy.verify_fit(parent, job):
                raise ValueError(f'Missing historical control: {job["id"]}')
            path = Path(job['output'])
            refs.append(dict(id=job['id'], seed=job['seed'], output=str(path), tasks=job['tasks'],
                             arm='historical_multi' if historical else 'single_task',
                             hashes={name:digest(path/name) for name in legacy.ARTIFACTS}))
    return dict(schema='heat_transfer_attention_30m_v1', smoke=smoke, primary=PRIMARY,
                tasks=study_tasks(), seeds=seeds, settings=settings, arms=ARMS, jobs=jobs, references=refs,
                data_sha256=digest(DATA), manifest_sha256=digest(DATA.parent/'manifest.json'),
                split_sha256=digest(DATA.parent/'structure_splits.csv'),
                checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), source_sha256=fingerprint(),
                legacy_plan_sha256=digest(legacy.OUT/'plan.json'),
                selection='One checkpoint: equal mean validation standardized MSE of six heat-transfer properties.',
                targets='All eleven continuous regression targets; no screening thresholds.',
                protocol='User-chosen 30.2M initialization; all eleven MLP tasks; cross-ion attention fixed on. '
                '2x2 uniform/heat90 loss x plain/channel-attention plus a parameter-matched weighted residual adapter. '
                'Five fitting seeds, frozen complete-IL split. Historical all-task-selected baseline is separate '
                'from the new application-selected control. No test selection; prior test exposure disclosed.')


def verify_inputs(plan):
    checks = {DATA:plan['data_sha256'], DATA.parent/'manifest.json':plan['manifest_sha256'],
              DATA.parent/'structure_splits.csv':plan['split_sha256'], CHECKPOINT:plan['checkpoint_sha256'],
              legacy.OUT/'plan.json':plan['legacy_plan_sha256']}
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f'Changed immutable input: {path}')
    if fingerprint() != plan['source_sha256']:
        raise ValueError('New scientific source changed; choose a separately versioned plan')


def verify_fit(plan, job):
    path = Path(job['output'])
    if not (path/'execution.json').exists():
        return False
    execution = json.loads((path/'execution.json').read_text())
    if execution['state'] != 'complete':
        return False
    if not all((path/f).exists() for f in FILES):
        raise ValueError(f'Completed fit missing artifacts: {path}')
    m = json.loads((path/'metrics.json').read_text())
    expected = dict(settings=plan['settings'], seed=job['seed'], arm=job['arm'],
                    feature_mode=job['feature_mode'], weights=job['weights'])
    if m['arguments'] != expected or m['source_sha256'] != plan['source_sha256']:
        raise ValueError(f'Configuration or source mismatch: {path}')
    if m['data_sha256'] != plan['data_sha256'] or bool(m['subset_run']) != plan['smoke']:
        raise ValueError(f'Data/subset mismatch: {path}')
    if m['pretraining']['sha256'] != plan['checkpoint_sha256'] or m['pretraining']['include_cross']:
        raise ValueError(f'Initialization mismatch: {path}')
    if execution['source_sha256'] != plan['source_sha256']:
        raise ValueError(f'Execution source mismatch: {path}')
    cfg = json.loads((path/'task_config.json').read_text())
    if cfg['tasks'] != plan['tasks'] or cfg['selection_tasks'] != plan['primary']:
        raise ValueError(f'Task/selection mismatch: {path}')
    return True
