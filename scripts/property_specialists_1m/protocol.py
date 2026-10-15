"""Immutable all-property 1M weight experiment, matched to the 30.2M design."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from il_contrastive.equations import study_tasks
from il_contrastive.pretraining import digest
from scripts import run_multitask_heads_study as legacy
from scripts import run_multitask_heads_1m as parent_study
from scripts.property_specialists.protocol import choose_tabular, loss_weights, ARMS, SEEDS, TAB

ROOT = legacy.ROOT
OUT = Path('runs/property_specialists_1m_v1')
DATA = legacy.DATA
CHECKPOINT = parent_study.CHECKPOINT
SCALE_ROOT = Path('runs/property_specialists_30m_v1')
SCALE_REPORT = Path('reports/property_specialists_30m_v1')
FILES = ['best.pt', 'metrics.json', 'task_config.json', 'validation_predictions.csv',
         'test_predictions.csv', 'execution.json', 'best_validation_predictions.csv', 'reload_audit.json']


def fingerprint():
    files = [Path(__file__), Path(__file__).with_name('train.py'),
             Path('scripts/property_specialists/protocol.py'),
             Path('scripts/application_weighted_1m/train.py')]
    return hashlib.sha256(legacy.training_fingerprint().encode() + b''.join(p.read_bytes() for p in files)).hexdigest()


def settings(smoke):
    return dict(hidden=64, layers=3, heads=4, cross_ion=True, head_mode='mlp',
        epochs=1 if smoke else 200, steps_per_epoch=2 if smoke else 100,
        per_task_batch=2 if smoke else 8, lr=.0005, weight_decay=.0001,
        patience=15, threads=4, gradient_clip=5., device='cpu' if smoke else 'cuda',
        limit_per_task=8 if smoke else 0, task_schema='ablation_v1')


def make_plan(output=OUT, smoke=False):
    output = Path(output)
    parent = json.loads((parent_study.OUT/'plan.json').read_text())
    parent_study.verify_inputs(parent)
    status = json.loads(Path('reports/multitask_heads_1m_v1/status.json').read_text())
    if not status['complete'] or status['plan_sha256'] != digest(parent_study.OUT/'plan.json'):
        raise ValueError('Expected completed and audited 1M parent study')
    ssl = json.loads((CHECKPOINT.parent/'status.json').read_text())
    if ssl['state'] != 'complete' or ssl['completed_passes'] != 3 or ssl['presentations'] != 3_000_000:
        raise ValueError('Expected three complete passes of the 1M encoder')
    if digest(CHECKPOINT) != (CHECKPOINT.parent/'final_sha256.txt').read_text().strip():
        raise ValueError('1M checkpoint hash changed')
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if ck['completed_passes'] != 3 or any(ck['manifest']['settings'][k] != v for k, v in
            dict(hidden=64, layers=3, structures=1_000_000, seed=42).items()):
        raise ValueError('Unexpected 1M checkpoint architecture or exposure')
    tasks = study_tasks()
    if tasks != json.loads((DATA.parent/'task_schema.json').read_text()):
        raise ValueError('Frozen task schema differs')
    old = json.loads((SCALE_ROOT/'plan.json').read_text())
    old_status = json.loads((SCALE_REPORT/'status.json').read_text())
    if (not old_status['complete'] or old_status['completed_new'] != 165
            or old_status['audits'] != 4510 or old_status['plan_sha256'] != digest(SCALE_ROOT/'plan.json')):
        raise ValueError('Expected the complete audited 30.2M specialist study')
    if old['settings'] != settings(False) or old['tasks'] != tasks or old['seeds'] != SEEDS or old['arms'] != ARMS:
        raise ValueError('1M design is not matched to the 30.2M specialist protocol')
    if old['data_sha256'] != digest(DATA):
        raise ValueError('Scale comparison requires the identical frozen data')
    targets = ['co2_solubility'] if smoke else list(tasks)
    seeds = [42] if smoke else SEEDS
    jobs = []
    for target in targets:
        for seed in seeds:
            for arm in ARMS:
                ident = f'{target}/{seed}/{arm}'
                jobs.append(dict(id=ident, target=target, seed=seed, arm=arm,
                    weights=loss_weights(target, arm), output=str(output/'fits'/target/f'seed_{seed}'/arm),
                    command=[sys.executable, '-u', '-m', 'scripts.property_specialists_1m.train',
                             '--root', str(output), '--job', ident]))
    references, tabular, scale_references, choices = [], [], [], {}
    if not smoke:
        for j in parent['jobs']:
            if j['initialization'] != 'pretrained_1m':
                continue
            multi = j['scope'] == 'multi' and j['head_family'] == 'mlp'
            single = j['scope'] == 'single' and 'mlp' in j['head_aliases']
            if not (multi or single):
                continue
            if not parent_study.verify_fit(parent, j):
                raise ValueError(f'Missing 1M reference: {j["id"]}')
            path = Path(j['output'])
            names = legacy.ARTIFACTS + (['execution.json'] if (path/'execution.json').exists() else [])
            references.append(dict(id=j['id'], arm='historical_multi' if multi else 'historical_single',
                seed=j['seed'], tasks=j['tasks'], output=str(path), hashes={n: digest(path/n) for n in names}))
        if len(references) != 60:
            raise ValueError('Expected five 1M historical MTL and 55 single-task references')
        frame = pd.read_csv(TAB/'per_seed_statistics.csv')
        choices = choose_tabular(frame)
        registry = {str(Path(x['path']).resolve()): x for x in json.loads((TAB/'artifact_registry.json').read_text())}
        for row in frame[frame.split.eq('validation')].to_dict('records'):
            path = Path(row['path'])
            hashes = registry[str(path.resolve())]['artifact_sha256']
            if any(digest(path/n) != sha for n, sha in hashes.items()):
                raise ValueError(f'Tabular artifact changed: {path}')
            tabular.append(dict(id=f'tabular/{row["task"]}/{row["model"]}/{row["representation"]}/{row["seed"]}',
                task=row['task'], seed=int(row['seed']), model=row['model'], representation=row['representation'],
                output=str(path), hashes=hashes))
        registry = {r['id']: r for r in json.loads((SCALE_REPORT/'artifact_registry.json').read_text())}
        for job in old['jobs']:
            ref = registry[job['id']]
            if ref['output'] != job['output'] or any(digest(Path(ref['output'])/n) != sha for n, sha in ref['hashes'].items()):
                raise ValueError(f'Changed 30.2M specialist artifact: {job["id"]}')
            scale_references.append(ref)
    inputs = [DATA, DATA.parent/'manifest.json', DATA.parent/'structure_splits.csv',
              DATA.parent/'task_schema.json', CHECKPOINT, CHECKPOINT.parent/'status.json',
              CHECKPOINT.parent/'final_sha256.txt', parent_study.OUT/'plan.json', SCALE_ROOT/'plan.json']
    if not smoke:
        inputs += [TAB/'per_seed_statistics.csv', TAB/'artifact_registry.json']
        inputs += [SCALE_REPORT/n for n in ['status.json', 'artifact_registry.json', 'per_seed.csv',
                    'aggregates.csv', 'validation_selection.csv', 'resources.csv']]
    return dict(schema='property_specialists_1m_v1', smoke=smoke, output=str(output), tasks=tasks,
        targets=targets, seeds=seeds, settings=settings(smoke), arms=ARMS, jobs=jobs,
        references=references, tabular=tabular, tabular_choices=choices, scale_references=scale_references,
        inputs={str(p): digest(p) for p in inputs}, data_sha256=digest(DATA), checkpoint=str(CHECKPOINT),
        checkpoint_sha256=digest(CHECKPOINT), source_sha256=fingerprint(), reload_tolerance_training_sd=5e-6,
        selection='Every new fit selects and early-stops on its target validation standardized MSE only. '
                  'Choose among the three weights using five-seed mean target validation MSE; no test selection.',
        protocol='1M companion to the completed 30.2M property-specialist study: eleven targets x '
                 'uniform/90%-target/100%-target x five fitting seeds. This is a three-point weight comparison, '
                 'not a dense search. Uniform loss is 1/11 per task; priority90 uses .90 target and .01 each auxiliary; '
                 'target_only uses 1.0 target and zero auxiliary losses. All arms retain eleven plain MLP heads, '
                 'identical initial weights per seed, balanced eight-per-task batches, 100 steps/epoch, '
                 'full encoder fine-tuning, fixed cross-ion attention, 200-epoch cap and patience 15. '
                 'Same frozen complete-IL split and training-only standardization. The only intended scientific '
                 'change relative to the 30.2M specialists is the three-pass 1M encoder initialization. '
                 'A numerical safeguard replays the saved best checkpoint on its fitting device against saved '
                 'selected-epoch predictions, with the unchanged 5e-6 training-SD tolerance. '
                 'Historical 1M single-task/all-task-selected MTL are contextual controls with different stopping criteria. '
                 'Tabular selection uses validation across six configurations; its test minimum is descriptive only. '
                 'Native CO2 and ln-viscosity MAEs remain distinct. Repeatedly inspected test split is exploratory.',
        inference='As in the 30.2M study: priority90 versus matched uniform, target_only and validation-selected tabular; '
                  'whole-IL intervals and sign randomization under row/equal-IL MAE weightings. '
                  'BH over 66 tests after all 165 fits complete. Across-scale differences are descriptive; '
                  'one pretrained checkpoint per scale does not estimate pretraining-seed uncertainty.')


def verify_inputs(plan):
    for path, sha in plan['inputs'].items():
        if digest(path) != sha:
            raise ValueError(f'Changed immutable input: {path}')
    if fingerprint() != plan['source_sha256']:
        raise ValueError('Scientific source changed; use a separately versioned plan')
    if (plan['schema'] != 'property_specialists_1m_v1' or plan['settings'] != settings(plan['smoke'])
            or plan['tasks'] != study_tasks() or plan['arms'] != ARMS):
        raise ValueError('Scientific design changed')
    if plan['checkpoint'] != str(CHECKPOINT) or plan['checkpoint_sha256'] != plan['inputs'][str(CHECKPOINT)]:
        raise ValueError('Unexpected checkpoint provenance')
    expected = {(t,s,a) for t in (['co2_solubility'] if plan['smoke'] else study_tasks())
                for s in ([42] if plan['smoke'] else SEEDS) for a in ARMS}
    actual = {(j['target'],j['seed'],j['arm']) for j in plan['jobs']}
    if actual != expected or len(plan['jobs']) != len(expected):
        raise ValueError('Missing or duplicate target/seed/arm jobs')
    for job in plan['jobs']:
        if job['weights'] != loss_weights(job['target'], job['arm']):
            raise ValueError('Loss weights changed')
        path = Path(plan['output'])/'fits'/job['target']/f'seed_{job["seed"]}'/job['arm']
        if Path(job['output']) != path:
            raise ValueError('Job output escapes its fixed experiment directory')


def verify_fit(plan, job):
    path = Path(job['output'])
    if not (path/'execution.json').exists():
        return False
    execution = json.loads((path/'execution.json').read_text())
    if execution['state'] != 'complete':
        return False
    if not all((path/n).exists() for n in FILES):
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
    config = {k: plan['settings'][k] for k in ['hidden','layers','heads','cross_ion','head_mode']}
    config['tasks'] = plan['tasks']
    cfg = dict(tasks=plan['tasks'], weights=job['weights'], selection_tasks=[job['target']], config=config)
    if json.loads((path/'task_config.json').read_text()) != cfg or m['trainable_parameters'] != 384803:
        raise ValueError(f'Model/selection mismatch: {path}')
    audit = json.loads((path/'reload_audit.json').read_text())
    if (audit['device'] != plan['settings']['device'] or not audit['passed']
            or not np.isfinite(audit['max_abs_standardized_prediction_delta'])
            or audit['max_abs_standardized_prediction_delta'] > plan['reload_tolerance_training_sd']):
        raise ValueError(f'Checkpoint replay failed: {path}')
    if set(audit['hashes']) != {'best.pt','best_validation_predictions.csv','validation_predictions.csv','test_predictions.csv'}:
        raise ValueError('Incomplete checkpoint replay hashes')
    for name, sha in audit['hashes'].items():
        if digest(path/name) != sha:
            raise ValueError(f'Replay-audited artifact changed: {path/name}')
    return True
