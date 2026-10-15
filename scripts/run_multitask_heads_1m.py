"""Matched 1M extension: 85 new fits, with 215 immutable reference models."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from il_contrastive.pretraining import digest
from scripts.pretrain_full_coverage import atomic_json
from scripts import run_multitask_heads_study as original

ROOT = original.ROOT
OUT = Path('runs/multitask_heads_1m_v1')
CHECKPOINT = original.SCALE / 'ssl/1m/final.pt'
PARENT = original.OUT
REGISTRIES = [Path('reports/pretraining_full_coverage_v1/artifact_registry.json'),
              Path('reports/multitask_heads_30m_v1/artifact_registry.json')]


def orchestration_fingerprint():
    paths = [Path(__file__), Path('scripts/report_multitask_heads_study.py'),
             Path('scripts/report_multitask_heads_1m.py')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def design_jobs(parent, output, smoke=False):
    """Retain old model identities; append only the third initialization."""
    jobs = []
    if not smoke:
        for old in parent['jobs']:
            job = copy.deepcopy(old)
            job.update(reference_study='30m', source_reused=old['reused'], reused=True)
            jobs.append(job)
    for old in parent['jobs']:
        if old['initialization'] != 'pretrained_30m' or (smoke and old['seed'] != 42):
            continue
        job = copy.deepcopy(old)
        job['id'] = old['id'].replace('pretrained_30m', 'pretrained_1m')
        job['initialization'] = 'pretrained_1m'
        job['reused'] = old['reused'] and not smoke
        if job['reused']:
            path = original.SCALE / 'downstream' / job['tasks'][0] / f'seed_{job["seed"]}' / '1m'
        else:
            path = output / 'fits' / f'seed_{job["seed"]}' / 'pretrained_1m' / job['scope'] / job['head_family']
            if job['scope'] == 'single':
                path = path / job['tasks'][0]
        job['output'] = str(path)
        cmd = job['command']
        cmd[0] = sys.executable
        cmd[cmd.index('--output') + 1] = str(path)
        cmd[cmd.index('--pretrained-encoder') + 1] = str(CHECKPOINT)
        if smoke:
            for flag, value in {'--epochs': '1', '--steps-per-epoch': '2', '--per-task-batch': '2',
                                '--limit-per-task': '8', '--device': 'cpu'}.items():
                cmd[cmd.index(flag) + 1] = value
        jobs.append(job)
    return jobs


def make_plan(output=OUT, smoke=False):
    parent = json.loads((PARENT / 'plan.json').read_text())
    # Reconstructing the old plan verifies its frozen data, source and checkpoint.
    if original.make_plan() != parent:
        raise ValueError('The original scientific plan/source has changed')
    status = json.loads(Path('reports/multitask_heads_30m_v1/status.json').read_text())
    if not status['complete'] or status['completed_fits'] != 200 or status['plan_sha256'] != digest(PARENT / 'plan.json'):
        raise ValueError('Finish and audit the 200 original models first')
    ssl = json.loads((CHECKPOINT.parent / 'status.json').read_text())
    if ssl['state'] != 'complete' or ssl['completed_passes'] != 3 or ssl['presentations'] != 3_000_000:
        raise ValueError('The 1M encoder has not completed three full passes')
    ck_hash = digest(CHECKPOINT)
    if ck_hash != (CHECKPOINT.parent / 'final_sha256.txt').read_text().strip():
        raise ValueError('1M checkpoint checksum changed')
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    settings = ck['manifest']['settings']
    if ck['format'] != 'ilmixer_ssl_v1' or ck['completed_passes'] != 3 or any(
            settings[k] != v for k, v in dict(hidden=64, layers=3, structures=1_000_000, seed=42).items()):
        raise ValueError('Unexpected 1M checkpoint architecture/exposure')
    plan = {k: copy.deepcopy(v) for k, v in parent.items() if k != 'jobs'}
    plan.update(schema='multitask_heads_1m_v1', smoke=smoke, checkpoint=str(CHECKPOINT),
                checkpoint_sha256=ck_hash, reference_30m_checkpoint_sha256=parent['checkpoint_sha256'],
                orchestration_sha256=orchestration_fingerprint(),
                reference_hashes={str(p): digest(p) for p in [PARENT / 'plan.json', *REGISTRIES]},
                checkpoint_choice='1M chosen for this extension using the previous three-property validation ranking; '
                                  'not asserted to be best across eleven properties.')
    if smoke:
        plan['settings'].update(epochs=1, steps_per_epoch=2, per_task_batch=2, limit_per_task=8, device='cpu')
        plan['seeds'] = [42]
    plan['jobs'] = design_jobs(parent, output, smoke)
    plan['protocol'] = (
        'Matched extension with scratch, 1M and full-corpus 30.2M initializations; all eleven properties, '
        'single/multitask scopes, MLP/equation head families and fitting seeds 42–46. '
        'Reuse 200 original models and 15 compatible 1M scale controls; train only 85 additional 1M models. '
        'The unchanged trainer, frozen split, target schema, cross-ion attention and maximum per-property '
        'sampled-label budget are retained. Static single-task head cells are aliases. '
        'Choose individual-task models by validation; choose any shared multitask encoder using the '
        'equal-eleven-property validation criterion. Heat-capacity heads also differ in pressure input. '
        'The previously inspected test set is descriptive; no universal or monotonic scaling claim.')
    return plan


def verify_fit(plan, job):
    """Use the existing checks with an explicit initialization-name adapter.

    The old verifier calls its pretrained arm 'pretrained_30m'. Supplying the 1M
    checksum is essential: only the label is normalized, never the loaded weights.
    """
    if not job['reused']:
        execution = Path(job['output']) / 'execution.json'
        if execution.exists() and json.loads(execution.read_text())['state'] != 'complete':
            return False
    adapted_plan = plan
    adapted_job = job
    if job.get('reference_study') == '30m':
        adapted_plan = plan | {'checkpoint_sha256': plan['reference_30m_checkpoint_sha256']}
        adapted_job = job | {'reused': job['source_reused']}
    elif job['initialization'] == 'pretrained_1m':
        adapted_job = job | {'initialization': 'pretrained_30m'}
    else:
        raise ValueError(f'Unrecognized extension job: {job["id"]}')
    return original.verify_fit(adapted_plan, adapted_job)


def verify_inputs(plan):
    if original.training_fingerprint() != plan['source_sha256']:
        raise ValueError('Training source changed')
    if orchestration_fingerprint() != plan['orchestration_sha256']:
        raise ValueError('Extension runner/report source changed; use a new plan version')
    for filename, expected in plan['reference_hashes'].items():
        if digest(filename) != expected:
            raise ValueError(f'Reference plan/registry changed: {filename}')
    for filename, expected in [(original.DATA, plan['data_sha256']),
                               (original.DATA.parent / 'manifest.json', plan['frozen_manifest_sha256']),
                               (CHECKPOINT, plan['checkpoint_sha256']),
                               (original.CHECKPOINT, plan['reference_30m_checkpoint_sha256'])]:
        if digest(filename) != expected:
            raise ValueError(f'Frozen data/checkpoint changed: {filename}')


def report(output):
    subprocess.run([sys.executable, '-u', '-m', 'scripts.report_multitask_heads_1m', '--root', str(output)], check=True)


def progress(plan, output, state, **extra):
    done = [j for j in plan['jobs'] if verify_fit(plan, j)]
    atomic_json(dict(state=state, updated_unix=time.time(), completed_fits=len(done),
                     total_fits=len(plan['jobs']), new_completed=sum(not j['reused'] for j in done),
                     new_total=sum(not j['reused'] for j in plan['jobs']), **extra), output / 'queue_status.json')


def show_status(output):
    if not (output / 'plan.json').exists():
        print('1M extension has not been prepared.'); return
    plan = json.loads((output / 'plan.json').read_text())
    state = json.loads((output / 'queue_status.json').read_text()) if (output / 'queue_status.json').exists() else {}
    with (output / 'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
    saved = state.get('state', 'prepared')
    if saved == 'running' and not active:
        saved = 'interrupted (no queue lock held)'
    elif active:
        saved = 'active / ' + saved
    done = [j for j in plan['jobs'] if verify_fit(plan, j)]
    print(f'1M extension: {saved}; one GPU worker')
    print(f'New fits: {sum(not j["reused"] for j in done)}/{sum(not j["reused"] for j in plan["jobs"])}; '
          f'combined models: {len(done)}/{len(plan["jobs"])}')
    if (output / 'current_job.json').exists():
        current = json.loads((output / 'current_job.json').read_text())
        log = Path(current['log'])
        print('Latest job:', current['job_id'])
        print('Log:', log)
        if log.exists():
            print(f'Last log update: {max(0, time.time()-log.stat().st_mtime):.0f} seconds ago')
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 2048))
                print('\n'.join(stream.read().decode(errors='replace').splitlines()[-2:]))
    if state.get('error'):
        print('Error:', state['error'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args(); os.chdir(ROOT)
    output = Path(args.output)
    if args.status:
        show_status(output); return
    if output.resolve() in {PARENT.resolve(), original.SCALE.resolve()}:
        parser.error('The extension needs a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [output / 'queue.lock', PARENT / 'queue.lock', original.SCALE / 'queue.lock',
                 Path('runs/pretraining_downstream_program_v1/program.lock')]:
        handle = path.open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another study is active: {path}') from None
        locks.append(handle)
    plan = make_plan(output)
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Immutable plan differs; preserve it and choose a new version')
    if not path.exists():
        atomic_json(plan, path)
    (output / 'commands.txt').write_text('\n'.join(shlex.join(j['command']) for j in plan['jobs'] if not j['reused']) + '\n')
    verify_inputs(plan)
    print('300 combined models; 215 reference models; 85 new 1M fits; one GPU worker.', flush=True)
    report(output)  # Hash-verify and audit every reference before any training.
    for job in plan['jobs']:
        if job['reused'] and not verify_fit(plan, job):
            raise ValueError(f'Missing reference fit: {job["id"]}')
    if not args.run:
        progress(plan, output, 'complete' if all(verify_fit(plan, j) for j in plan['jobs']) else 'prepared')
        print('Preparation and reference audits complete. No production fits started.', flush=True); return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; launch from the GPU-enabled WSL terminal')
    progress(plan, output, 'running')
    try:
        for job in plan['jobs']:
            if verify_fit(plan, job):
                continue
            if job['reused']:
                raise ValueError(f'Missing reference fit: {job["id"]}')
            original.execute(job, output)
            if not verify_fit(plan, job):
                raise ValueError(f'Fit did not complete: {job["id"]}')
            progress(plan, output, 'running', latest_completed=job['id'])
            # Each seed ends with the MLP multitask fit; refresh the complete audit then.
            if job['scope'] == 'multi' and job['head_family'] == 'mlp':
                report(output)
        verify_inputs(plan)
        report(output)
        progress(plan, output, 'complete')
    except BaseException as error:
        progress(plan, output, 'stopped', error=f'{type(error).__name__}: {error}')
        try:
            report(output)
        except Exception as reporting_error:
            print(f'Additional reporting error: {reporting_error}', file=sys.stderr, flush=True)
        raise
    print('All 300 combined models completed and audited; 1M extension is ready for analysis.', flush=True)


if __name__ == '__main__':
    main()
