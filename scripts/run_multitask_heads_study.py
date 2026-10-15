"""Matched all-property initialization x single/multitask x head-family study."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pandas as pd
import torch
from il_contrastive.equations import STATIC_TASKS, choose_cp_degree, study_tasks
from il_contrastive.pretraining import digest
from scripts.pretrain_full_coverage import atomic_json
from scripts.tabular_benchmark_archive import freeze

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('runs/multitask_heads_30m_v1')
SCALE=Path('runs/pretraining_full_coverage_v1')
DATA=Path('data/benchmarks/il_v1/observations.csv.gz')
CHECKPOINT=SCALE/'ssl/30.2m/final.pt'
BENCHMARK_TASKS=['melting_point','viscosity','co2_solubility']
SEEDS=list(range(42,47))
ARTIFACTS=['metrics.json','best.pt','task_config.json','validation_predictions.csv','test_predictions.csv']


def training_fingerprint():
    files=sorted(Path('il_contrastive').glob('*.py'))+[Path(__file__),Path('scripts/train_multitask_heads.py')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def design_jobs(tasks, seeds, output, cp_degree, smoke=False):
    jobs=[]
    for seed in seeds:
        for init in ['scratch','pretrained_30m']:
            for task in tasks:
                for head in (['equations'] if task in STATIC_TASKS else ['equations','mlp']):
                    reused=not smoke and task in BENCHMARK_TASKS and head=='equations'
                    path=(SCALE/'downstream'/task/f'seed_{seed}'/('scratch' if init=='scratch' else '30.2m')) if reused else (
                        output/'fits'/f'seed_{seed}'/init/'single'/head/task)
                    jobs.append(dict(id=f'{seed}/{init}/single/{head}/{task}',seed=seed,initialization=init,
                        scope='single',head_family=head,tasks=[task],output=str(path),reused=reused,
                        cp_degree=cp_degree if task=='heat_capacity' else 1,
                        head_aliases=['equations','mlp'] if task in STATIC_TASKS else [head]))
            for head in ['equations','mlp']:
                jobs.append(dict(id=f'{seed}/{init}/multi/{head}',seed=seed,initialization=init,scope='multi',
                    head_family=head,tasks=list(tasks),output=str(output/'fits'/f'seed_{seed}'/init/'multi'/head),
                    reused=False,cp_degree=cp_degree,head_aliases=[head]))
    return jobs


def make_plan(output=OUT,smoke=False,device='cuda'):
    frozen=freeze(DATA,DATA.parent)
    frame=pd.read_csv(DATA,low_memory=False)
    if frame.groupby('smiles').split.nunique().max()!=1:raise ValueError('Global structure split leakage')
    degree,cp_audit=choose_cp_degree(frame[frame.split=='train'])
    registry=study_tasks()
    status=json.loads((CHECKPOINT.parent/'status.json').read_text())
    if status['state']!='complete' or status['presentations']!=3*30214506:raise ValueError('Full-corpus SSL incomplete')
    ck_hash=digest(CHECKPOINT)
    if ck_hash!=(CHECKPOINT.parent/'final_sha256.txt').read_text().strip():raise ValueError('Checkpoint checksum changed')
    settings=dict(hidden=64,layers=3,epochs=1 if smoke else 100,steps_per_epoch=2 if smoke else 100,
        per_task_batch=2 if smoke else 8,lr=.0005,patience=15,threads=4,device=device,
        cross_ion=True,pretrained_cross=False,task_schema='ablation_v1',limit_per_task=8 if smoke else 0)
    seeds=[42] if smoke else SEEDS
    jobs=design_jobs(registry,seeds,output,degree,smoke)
    for job in jobs:
        cmd=[sys.executable,'-u','-m','scripts.train_multitask_heads','--data',str(DATA),
             '--output',job['output'],'--tasks',','.join(job['tasks']),'--seed',str(job['seed']),
             '--head-mode',job['head_family'],'--cp-degree',str(job['cp_degree']),'--cross-ion']
        for k,v in settings.items():
            if k not in ['cross_ion','pretrained_cross']:cmd += ['--'+k.replace('_','-'),str(v)]
        if job['initialization']=='pretrained_30m':cmd+=['--pretrained-encoder',str(CHECKPOINT)]
        job['command']=cmd
    return dict(schema='multitask_heads_30m_v1',smoke=smoke,tasks=list(registry),task_schema=registry,
        seeds=seeds,settings=settings,cp_degree=degree,cp_audit=cp_audit,
        data_sha256=frozen['data_sha256'],frozen_manifest_sha256=digest(DATA.parent/'manifest.json'),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=ck_hash,source_sha256=training_fingerprint(),
        scale_plan_sha256=digest(SCALE/'plan.json'),jobs=jobs,
        protocol='User-chosen full-corpus third-pass encoder; fixed cross-ion attention; all 11 properties. '
        'Five seeds on one frozen split. Equal-task loss and per-task label budget. '
        'Structure-only single-task heads are aliased, not independently refitted. '
        'Single-task validation selection is per property; multitask selection uses one equal-property validation mean. '
        'Heat-capacity head comparison also changes pressure input. No fixed-compute or fresh-test claim.')


def verify_fit(plan,job):
    out=Path(job['output']);file=out/'metrics.json'
    if not file.exists():return False
    m=json.loads(file.read_text());a=m['arguments']
    expected=plan['settings']|dict(tasks=','.join(job['tasks']),seed=job['seed'],head_mode=job['head_family'])
    if m['data_sha256']!=plan['data_sha256'] or bool(m['subset_run'])!=plan['smoke']:
        raise ValueError(f'Data/subset mismatch: {file}')
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError(f'Configuration mismatch: {file}')
    if int(m['cp_degree'])!=job['cp_degree']:raise ValueError(f'Cp degree mismatch: {file}')
    pre=m.get('pretraining')
    if job['initialization']=='pretrained_30m':
        if not pre or pre['sha256']!=plan['checkpoint_sha256'] or pre['include_cross']:
            raise ValueError(f'Wrong initialization: {file}')
    elif pre is not None:raise ValueError(f'Scratch fit loaded weights: {file}')
    cfg=json.loads((out/'task_config.json').read_text())
    if cfg['tasks']!={t:plan['task_schema'][t] for t in job['tasks']}:raise ValueError(f'Task schema mismatch: {file}')
    if not all((out/n).exists() for n in ARTIFACTS):raise ValueError(f'Incomplete artifacts: {out}')
    if not job['reused']:
        execution=json.loads((out/'execution.json').read_text())
        if execution['state']!='complete':return False
        if execution['source_sha256']!=plan['source_sha256']:raise ValueError(f'Training source changed: {out}')
    return True


def report(output):
    subprocess.run([sys.executable,'-m','scripts.report_multitask_heads_study','--root',str(output)],check=True)


def execute(job,output):
    folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
    number=len(list(folder.glob('attempt_*.log')))+1
    log=folder/f'attempt_{number:03d}.log'
    # Retain interrupted/failed artifacts before a deterministic restart.
    leftovers=[folder/n for n in ARTIFACTS+['execution.json','split_counts.json'] if (folder/n).exists()]
    if leftovers:
        archive=folder/f'incomplete_before_attempt_{number:03d}';archive.mkdir()
        for p in leftovers:p.rename(archive/p.name)
    print(f'START {job["id"]}; {log}',flush=True)
    current=dict(job_id=job['id'],log=str(log),started_unix=time.time())
    atomic_json(current,output/'current_job.json')
    with log.open('x') as stream:
        child=subprocess.Popen(job['command'],stdout=stream,stderr=subprocess.STDOUT)
        previous={}
        for sig in [signal.SIGTERM,signal.SIGINT]:
            previous[sig]=signal.signal(sig,lambda s,_:child.send_signal(s))
        try:code=child.wait()
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    if code:
        with (output/'failures.jsonl').open('a') as f:f.write(json.dumps(current|dict(returncode=code,finished_unix=time.time()))+'\n')
        raise RuntimeError(f'Fit failed ({code}): {job["id"]}; inspect {log}')
    print(f'COMPLETE {job["id"]}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();os.chdir(ROOT)
    if a.device!='cuda' and not a.smoke:p.error('Production uses CUDA; CPU is only for separate smoke checks')
    output=Path(a.output) if a.output else (Path('runs/multitask_heads_smoke_v1') if a.smoke else OUT)
    if a.smoke and output==OUT:p.error('Smoke output must be separate from production')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=make_plan(output,a.smoke,a.device);path=output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Immutable plan differs; choose a new version')
    if not path.exists():atomic_json(plan,path)
    (output/'commands.txt').write_text('\n'.join(shlex.join(j['command']) for j in plan['jobs'] if not j['reused'])+'\n')
    print(f'{len(plan["jobs"])} distinct fits; {sum(j["reused"] for j in plan["jobs"])} reference controls; smoke={a.smoke}',flush=True)
    if not a.run:
        report(output);print('Preparation only; no fitting started.',flush=True);return
    scale_lock=(SCALE/'queue.lock').open('a')
    fcntl.flock(scale_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.smoke:
        from scripts.run_full_coverage_study import make_plan as scale_plan,verify_fit as verify_scale_fit
        legacy=scale_plan()
        if legacy!=json.loads((SCALE/'plan.json').read_text()):raise ValueError('Legacy scientific source/plan changed')
        if not all(verify_scale_fit(legacy,j) for j in legacy['jobs']):raise ValueError('Finish all 120 scale-study fits first')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    atomic_json(dict(state='running',started_unix=time.time()),output/'queue_status.json')
    try:
        report(output)
        for job in plan['jobs']:
            if verify_fit(plan,job):continue
            if job['reused']:raise ValueError(f'Missing reference control: {job["id"]}')
            execute(job,output)
            if not verify_fit(plan,job):raise ValueError('Fit did not complete')
            report(output)
        atomic_json(dict(state='complete',finished_unix=time.time()),output/'queue_status.json')
    except BaseException as error:
        atomic_json(dict(state='stopped',error=f'{type(error).__name__}: {error}',updated_unix=time.time()),output/'queue_status.json')
        raise
    finally:report(output)
    print('All planned fits completed and audited.',flush=True)


if __name__=='__main__':main()
