"""Immutable, user-launched two-application protocol; no weight search."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from il_mixer.equations import study_tasks
from il_mixer.pretraining import digest
from scripts import run_multitask_heads_study as legacy
from scripts import run_multitask_heads_1m as parent_study

ROOT = legacy.ROOT
OUT = Path('runs/application_weighted_1m_v1')
DATA = legacy.DATA
CHECKPOINT = parent_study.CHECKPOINT
APPLICATIONS = {
    'heat_transfer': dict(label='Heat-transfer fluids', primary=[
        'heat_capacity', 'thermal_conductivity', 'viscosity', 'melting_point', 'decomposition_temperature']),
    'co2_absorbent': dict(label='CO2 absorbents', primary=['co2_solubility', 'viscosity', 'melting_point']),
}
ARMS = ['uniform_plain', 'weighted_plain']
SEEDS = list(range(42, 47))
FILES = ['best.pt', 'metrics.json', 'task_config.json', 'validation_predictions.csv',
         'test_predictions.csv', 'execution.json', 'best_validation_predictions.csv', 'reload_audit.json']


def loss_weights(application, arm):
    primary = APPLICATIONS[application]['primary']
    tasks = study_tasks()
    if arm == 'uniform_plain':
        return {t: 1/len(tasks) for t in tasks}
    if arm != 'weighted_plain':
        raise ValueError('Only prespecified uniform and weighted arms are supported')
    return {t: .9/len(primary) if t in primary else .1/(len(tasks)-len(primary)) for t in tasks}


def fingerprint():
    files = [Path(__file__), Path(__file__).with_name('train.py')]
    return hashlib.sha256(legacy.training_fingerprint().encode() + b''.join(p.read_bytes() for p in files)).hexdigest()


def make_plan(output=OUT, smoke=False):
    output = Path(output)
    parent = json.loads((parent_study.OUT/'plan.json').read_text())
    parent_study.verify_inputs(parent)
    status = json.loads(Path('reports/multitask_heads_1m_v1/status.json').read_text())
    if not status['complete'] or status['plan_sha256'] != digest(parent_study.OUT/'plan.json'):
        raise ValueError('Expected completed and audited 1M parent study')
    ssl = json.loads((CHECKPOINT.parent/'status.json').read_text())
    if ssl['state'] != 'complete' or ssl['completed_passes'] != 3 or ssl['presentations'] != 3_000_000:
        raise ValueError('Expected the full three-pass 1M checkpoint')
    if digest(CHECKPOINT) != (CHECKPOINT.parent/'final_sha256.txt').read_text().strip():
        raise ValueError('1M checkpoint hash changed')
    settings = dict(hidden=64, layers=3, heads=4, cross_ion=True, head_mode='mlp',
        epochs=1 if smoke else 200, steps_per_epoch=2 if smoke else 100,
        per_task_batch=2 if smoke else 8, lr=.0005, weight_decay=.0001,
        patience=15, gradient_clip=5., threads=4, device='cpu' if smoke else 'cuda',
        limit_per_task=8 if smoke else 0, task_schema='ablation_v1')
    seeds = [42] if smoke else SEEDS
    jobs = []
    for application, design in APPLICATIONS.items():
        for seed in seeds:
            for arm in ARMS:
                job_id = f'{application}/{seed}/{arm}'
                jobs.append(dict(id=job_id, application=application, seed=seed, arm=arm,
                    selection_tasks=design['primary'], weights=loss_weights(application, arm),
                    output=str(output/'fits'/application/f'seed_{seed}'/arm),
                    command=[sys.executable, '-u', '-m', 'scripts.application_weighted_1m.train',
                             '--root', str(output), '--job', job_id]))
    references = []
    if not smoke:
        primary_union = {t for a in APPLICATIONS.values() for t in a['primary']}
        for job in parent['jobs']:
            if job['initialization'] != 'pretrained_1m':
                continue
            multi = job['scope'] == 'multi' and job['head_family'] == 'mlp'
            single = job['scope'] == 'single' and 'mlp' in job['head_aliases'] and job['tasks'][0] in primary_union
            if not (multi or single):
                continue
            if not parent_study.verify_fit(parent, job):
                raise ValueError(f'Missing 1M reference: {job["id"]}')
            path = Path(job['output'])
            files = legacy.ARTIFACTS + (['execution.json'] if (path/'execution.json').exists() else [])
            references.append(dict(id=job['id'], output=str(path), seed=job['seed'], tasks=job['tasks'],
                arm='historical_multi' if multi else 'single_task',
                applications=[app for app, design in APPLICATIONS.items() if multi or job['tasks'][0] in design['primary']],
                hashes={n: digest(path/n) for n in files}, epoch_cap=100))
        if len(references) != 35:
            raise ValueError('Expected five original uniform MLPs and thirty single-task 1M references')
    inputs = [DATA, DATA.parent/'manifest.json', DATA.parent/'structure_splits.csv',
              DATA.parent/'task_schema.json', parent_study.OUT/'plan.json', CHECKPOINT,
              CHECKPOINT.parent/'status.json', CHECKPOINT.parent/'final_sha256.txt']
    return dict(schema='application_weighted_1m_v1', smoke=smoke, applications=APPLICATIONS,
        tasks=study_tasks(), seeds=seeds, arms=ARMS, jobs=jobs, references=references, settings=settings,
        inputs={str(p): digest(p) for p in inputs}, data_sha256=digest(DATA), checkpoint=str(CHECKPOINT),
        checkpoint_sha256=digest(CHECKPOINT), source_sha256=fingerprint(),
        reload_tolerance_training_sd=5e-6,
        selection='One checkpoint per application/arm/seed, selected by equal-priority-property mean validation standardized MSE. '
                  'Heat transfer: Cp, conductivity, ln(viscosity), MP, Td. CO2: ln(CO2), ln(viscosity), MP. '
                  'Choose between fixed arms by five-seed mean of the same validation objective; no test selection.',
        protocol='User-selected 1M three-pass general-molecular pretraining; full encoder fine-tuning and unchanged eleven-head MLP. '
                 'Fixed 90% total priority loss and 10% auxiliary loss, divided equally within groups. No weight search or new attention. '
                 'Twenty fits: two applications x uniform/weighted x five seeds, matched initialization and balanced per-task sampling. '
                 'Frozen complete-IL split and train-only standardization. Prespecified 200-epoch cap and patience 15. '
                 'Historical 1M references use 100 epochs and other checkpoint criteria; they are contextual controls. '
                 'The previous 30.2M heat-transfer study included Tg and used a 100-epoch cap, so it is not an initialization-only contrast. '
                 'CO2/viscosity are trained in ln space; native CO2 and log-viscosity MAEs are displayed separately. '
                 'Previously inspected test split: exploratory results, not independent chemical generalization.',
        inference='Weighted minus matched uniform: paired whole-IL bootstrap intervals and sign randomization for each application '
                  'and all eleven properties, under row/equal-IL MAE weighting. BH across 44 tests only after all twenty fits finish. '
                  'Intervals condition on the fitted models; no inferential selection of weights.')


def verify_inputs(plan):
    for filename, sha in plan['inputs'].items():
        if digest(filename) != sha:
            raise ValueError(f'Changed immutable input: {filename}')
    if fingerprint() != plan['source_sha256']:
        raise ValueError('Scientific source changed; use a separately versioned plan')
    if plan['applications'] != APPLICATIONS or plan['tasks'] != study_tasks() or plan['arms'] != ARMS:
        raise ValueError('Application/task design changed')
    if plan['checkpoint'] != str(CHECKPOINT) or plan['checkpoint_sha256'] != plan['inputs'][str(CHECKPOINT)]:
        raise ValueError('Unexpected checkpoint provenance')
    for job in plan['jobs']:
        if (job['weights'] != loss_weights(job['application'], job['arm'])
                or job['selection_tasks'] != APPLICATIONS[job['application']]['primary']):
            raise ValueError('Weights or application selection tasks changed')


def arguments(plan, job):
    return dict(settings=plan['settings'], seed=job['seed'], application=job['application'],
                arm=job['arm'], weights=job['weights'], selection_tasks=job['selection_tasks'])


def verify_fit(plan, job):
    path = Path(job['output'])
    if not (path/'execution.json').exists():
        return False
    execution = json.loads((path/'execution.json').read_text())
    if execution['state'] != 'complete':
        return False
    if not all((path/n).exists() for n in FILES):
        raise ValueError(f'Incomplete artifacts: {path}')
    m = json.loads((path/'metrics.json').read_text())
    if m['arguments'] != arguments(plan, job) or m['source_sha256'] != plan['source_sha256']:
        raise ValueError(f'Configuration or scientific source mismatch: {path}')
    if m['data_sha256'] != plan['data_sha256'] or bool(m['subset_run']) != plan['smoke']:
        raise ValueError(f'Data/subset mismatch: {path}')
    if m['pretraining']['sha256'] != plan['checkpoint_sha256'] or m['pretraining']['include_cross']:
        raise ValueError(f'Initialization mismatch: {path}')
    if execution['source_sha256'] != plan['source_sha256'] or m['trainable_parameters'] != 384803:
        raise ValueError(f'Execution/capacity mismatch: {path}')
    config = {k: plan['settings'][k] for k in ['hidden','layers','heads','cross_ion','head_mode']}
    config['tasks'] = plan['tasks']
    expected = dict(tasks=plan['tasks'], config=config, weights=job['weights'], selection_tasks=job['selection_tasks'])
    if json.loads((path/'task_config.json').read_text()) != expected:
        raise ValueError(f'Task/model/selection mismatch: {path}')
    audit = json.loads((path/'reload_audit.json').read_text())
    if (audit['device'] != plan['settings']['device'] or not audit['passed']
            or not np.isfinite(audit['max_abs_standardized_prediction_delta'])
            or audit['max_abs_standardized_prediction_delta'] > plan['reload_tolerance_training_sd']):
        raise ValueError(f'Checkpoint replay failed: {path}')
    for name, sha in audit['hashes'].items():
        if digest(path/name) != sha:
            raise ValueError(f'Replay-audited artifact changed: {path/name}')
    if not m['initial_state_sha256'] or len(m['history']) != m['epochs']:
        raise ValueError(f'Missing initial state/history: {path}')
    return True
