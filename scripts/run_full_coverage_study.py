"""Seven-scale full-coverage study; explicit --run required to train."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from il_contrastive.full_coverage import corpus_metadata
from il_contrastive.pretraining import digest
from scripts.pretrain_full_coverage import atomic_json, source_hash
from scripts.tabular_benchmark_archive import freeze

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('runs/pretraining_full_coverage_v1')
TASKS = ['melting_point', 'viscosity', 'co2_solubility']
SCALES = {'100k':100000, '500k':500000, '1m':1000000, '5m':5000000,
          '10m':10000000, '20m':20000000, '30.2m':30214506}
SEEDS = list(range(42, 47))


def make_plan():
    data = Path('data/benchmarks/il_v1/observations.csv.gz')
    frozen = freeze(data, Path('data/benchmarks/il_v1'))
    corpus = Path('data/pretraining/corpus_v1.sqlite')
    meta = corpus_metadata(corpus)
    n = sum(r['n'] for r in meta['counts'] if r['split']=='train')
    if n != SCALES['30.2m'] or meta['scope'] != 'full' or meta['downstream_sha256'] != frozen['data_sha256']:
        raise ValueError('Corpus size, scope or downstream-exclusion provenance mismatch')
    ssl = []
    for label, size in SCALES.items():
        dest = OUT/'ssl'/label
        cmd = [sys.executable, '-u', '-m', 'scripts.pretrain_full_coverage', '--corpus', str(corpus),
               '--output', str(dest), '--structures', str(size), '--passes', '3', '--hidden', '64', '--layers', '3',
               '--batch-size', '128', '--seed', '42', '--device', 'cuda', '--threads', '4', '--checkpoint-every', '500', '--log-every', '100']
        ssl.append(dict(scale=label, structures=size, output=str(dest), command=cmd))
    jobs = []
    for scale in ['scratch', *SCALES]:
        for seed in SEEDS:
            for task in TASKS:
                dest = OUT/'downstream'/task/f'seed_{seed}'/scale
                cmd = [sys.executable, '-u', '-m', 'scripts.train_coverage_benchmark', '--data', str(data),
                       '--output', str(dest), '--tasks', task, '--seed', str(seed), '--device', 'cuda', '--threads', '4',
                       '--hidden', '64', '--layers', '3', '--cross-ion', '--head-mode', 'equations', '--task-schema', 'ablation_v1',
                       '--cp-degree', '1', '--epochs', '100', '--steps-per-epoch', '100', '--per-task-batch', '8',
                       '--lr', '.0005', '--patience', '15']
                checkpoint = None if scale == 'scratch' else str(OUT/'ssl'/scale/'final.pt')
                if checkpoint:
                    cmd += ['--pretrained-encoder', checkpoint]
                jobs.append(dict(scale=scale, seed=seed, task=task, output=str(dest), command=cmd, checkpoint=checkpoint))
    sources = sorted(Path('il_contrastive').glob('*.py'))+[Path(__file__), Path('scripts/train_coverage_benchmark.py')]
    return dict(schema='pretraining_full_coverage_v1', scales=SCALES, tasks=TASKS, seeds=SEEDS,
                passes=3, batch_size=128, hidden=64, layers=3, pretraining_seed=42,
                data_sha256=frozen['data_sha256'], frozen_manifest_sha256=digest('data/benchmarks/il_v1/manifest.json'),
                corpus=meta, corpus_file=dict(path=str(corpus), bytes=corpus.stat().st_size, mtime_ns=corpus.stat().st_mtime_ns),
                ssl_source_sha256=source_hash(), downstream_source_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
                ssl_jobs=ssl, jobs=jobs,
                protocol='Fresh independent initialization at each nested data scale. Three complete passes; final third-pass encoder is the prespecified scale-comparison checkpoint. '
                'No old pretrained results or scratch fits are reused. Same frozen split and five downstream fitting seeds. '
                'Increasing both unique data and optimizer updates; not an isolated data-only effect. '
                'Main transfer measure: validation standardized MSE per property; shared candidate ranked by mean across three properties and all five seeds. '
                'Keep per-property regressions visible; the full corpus is trained regardless of interim ranking.')


def verify_fit(plan, job):
    p = Path(job['output'])/'metrics.json'
    if not p.exists():
        return False
    m = json.loads(p.read_text()); a = m['arguments']
    expected = dict(tasks=job['task'], seed=job['seed'], hidden=64, layers=3, cross_ion=True,
                    pretrained_cross=False, head_mode='equations', task_schema='ablation_v1', limit_per_task=0,
                    epochs=100, steps_per_epoch=100, per_task_batch=8, lr=.0005, patience=15)
    if m['data_sha256'] != plan['data_sha256'] or m.get('subset_run') or any(a.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Incompatible fit {p}')
    pre = m.get('pretraining')
    if job['checkpoint']:
        if not pre or pre['sha256'] != digest(job['checkpoint']) or pre['include_cross']:
            raise ValueError(f'Wrong SSL checkpoint {p}')
    elif pre is not None:
        raise ValueError(f'Scratch fit has a pretrained encoder: {p}')
    for name in ['validation_predictions.csv','test_predictions.csv','task_config.json','best.pt']:
        if not (p.parent/name).exists():
            raise ValueError(f'Missing fit artifact {p.parent/name}')
    return True


def report():
    subprocess.run([sys.executable, '-m', 'scripts.report_full_coverage_study'], check=True)


def execute(cmd, directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    number = len(list(directory.glob('attempt_*.log')))+1
    log = directory/f'attempt_{number:03d}.log'
    print(f'START {label}; log: {log}', flush=True)
    atomic_json(dict(label=label, log=str(log), started_unix=time.time()), OUT/'current_job.json')
    with log.open('x') as stream:
        child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        prior = {}
        def forward(sig, _):
            child.send_signal(sig)
        for sig in [signal.SIGINT,signal.SIGTERM]:
            prior[sig] = signal.signal(sig, forward)
        try:
            result = child.wait()
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
    if result:
        with (OUT/'failures.jsonl').open('a') as f:
            f.write(json.dumps(dict(label=label,returncode=result,log=str(log),time_unix=time.time()))+'\n')
        raise RuntimeError(f'Stopped ({result}): {label}; see {log}. Completed work is preserved.')
    print(f'COMPLETE {label}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--through',choices=[*SCALES,'all'],default='100k')
    p.add_argument('--ssl-only',action='store_true')
    a = p.parse_args();os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'queue.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('The full-coverage queue is already running')
    plan = make_plan();file = OUT/'plan.json'
    if file.exists() and json.loads(file.read_text()) != plan:
        raise ValueError('Immutable plan/source changed; do not mix experiment versions')
    if not file.exists():
        atomic_json(plan,file)
    labels = list(SCALES)
    chosen = labels if a.through=='all' else labels[:labels.index(a.through)+1]
    print(f'Full design: 7 SSL runs + 120 fresh downstream fits. Selected through {chosen[-1]}; one GPU worker.',flush=True)
    report()
    if not a.run:
        print('Preparation only; no training started.',flush=True);return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; launch in a GPU-enabled environment')
    try:
        for scale in chosen:
            j = next(j for j in plan['ssl_jobs'] if j['scale']==scale)
            folder = Path(j['output'])
            status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            if status.get('state')!='complete' or not (folder/'final.pt').exists():
                cmd = j['command']+(['--resume'] if (folder/'plan.json').exists() else [])
                execute(cmd,folder, f'{scale}: three full SSL passes')
            status = json.loads((folder/'status.json').read_text())
            if status['unique_structures_seen']!=j['structures'] or status['presentations']!=3*j['structures'] or status['completed_passes']!=3:
                raise ValueError('SSL completion does not match the full-coverage protocol')
            checkpoint = digest(folder/'final.pt');checksum_file=folder/'final_sha256.txt'
            if checksum_file.exists() and checksum_file.read_text().strip()!=checkpoint:
                raise ValueError('The finalized SSL checkpoint changed')
            checksum_file.write_text(checkpoint+'\n')
            report()
            if a.ssl_only:
                continue
            # Interleave scratch/pretrained fits so progress contains paired controls.
            for seed in SEEDS:
                for task in TASKS:
                    for condition in ['scratch',scale]:
                        fit = next(j for j in plan['jobs'] if (j['seed'],j['task'],j['scale'])==(seed,task,condition))
                        if verify_fit(plan,fit):
                            continue
                        execute(fit['command'],Path(fit['output']),f'{task} / {condition} / seed {seed}')
                        verify_fit(plan,fit);report()
        atomic_json(dict(state='selected_stages_complete',through=chosen[-1],ssl_only=a.ssl_only),OUT/'queue_status.json')
    finally:
        report()
    print('Selected stages finished; larger stages require their explicit launch command.',flush=True)


if __name__=='__main__':
    main()
