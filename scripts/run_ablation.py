"""Plan/resume a per-property, paired-seed 2x2 ablation without changing data splits."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
import pandas as pd
from il_mixer.equations import STATIC_TASKS, CONDITION_TASKS, choose_cp_degree

ROOT = Path(__file__).resolve().parents[1]


def variants(task):
    return [('gnn', False, 'mlp'), ('ion', True, 'mlp')] if task in STATIC_TASKS else [
        ('gnn_concat', False, 'mlp'), ('gnn_equations', False, 'equations'),
        ('ion_concat', True, 'mlp'), ('ion_equations', True, 'equations')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/ablation_v1')
    p.add_argument('--data', default='data/processed/observations.csv.gz')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--tasks', default=','.join(STATIC_TASKS+CONDITION_TASKS))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps-per-epoch', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--run', action='store_true', help='Execute jobs; otherwise only write the plan')
    p.add_argument('--smoke', action='store_true', help='All selected variants, 1 epoch/2 steps, 8 rows per split/task')
    a = p.parse_args()
    data = (ROOT/a.data).resolve(); output = (ROOT/a.output).resolve()
    seeds = [int(s) for s in a.seeds.split(',')]; tasks = a.tasks.split(',')
    if len(set(seeds)) != len(seeds) or len(set(tasks)) != len(tasks): p.error('Duplicate seed or task')
    if set(tasks)-set(STATIC_TASKS+CONDITION_TASKS): p.error('Unknown task')
    frame = pd.read_csv(data, low_memory=False)
    if frame.groupby('smiles').split.nunique().max() != 1: raise ValueError('Structure leakage')
    degree, coverage = choose_cp_degree(frame[frame.split == 'train'])
    source_hash = hashlib.sha256(b''.join(f.read_bytes() for f in sorted((ROOT/'il_mixer').glob('*.py')))).hexdigest()
    manifest = {'version':'ablation_v1','design':'single_task','seeds':seeds,'tasks':tasks,
                'data':str(data),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                'source_sha256':source_hash,'cp_degree':degree,'cp_coverage':coverage,'smoke':a.smoke,
                'epochs':1 if a.smoke else a.epochs,'steps_per_epoch':2 if a.smoke else a.steps_per_epoch,
                'batch_size':2 if a.smoke else a.batch_size,'hidden':a.hidden,'layers':a.layers,
                'patience':a.patience,'device':a.device,'threads':a.threads,'jobs':[]}
    for task in tasks:
        for seed in seeds:
            for variant, attention, head in variants(task):
                destination = output/task/f'seed_{seed}'/variant
                cmd = [sys.executable,'-m','il_mixer.train','--data',str(data),'--output',str(destination),
                       '--tasks',task,'--seed',str(seed),'--task-schema','ablation_v1','--head-mode',head,
                       '--cp-degree',str(degree),'--epochs',str(manifest['epochs']),
                       '--steps-per-epoch',str(manifest['steps_per_epoch']),
                       '--per-task-batch',str(manifest['batch_size']),'--hidden',str(a.hidden),'--layers',str(a.layers),
                       '--patience',str(a.patience),'--device',a.device,'--threads',str(a.threads)]
                if attention: cmd += ['--cross-ion']
                if a.smoke: cmd += ['--limit-per-task','8']
                manifest['jobs'].append(dict(task=task,seed=seed,variant=variant,output=str(destination),command=cmd))
    output.mkdir(parents=True,exist_ok=True)
    mp = output/'plan.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Existing plan differs (data/code/settings). Choose a new --output directory.')
    mp.write_text(json.dumps(manifest,indent=2))
    (output/'commands.txt').write_text('\n'.join(shlex.join(j['command']) for j in manifest['jobs'])+'\n')
    print(f"{len(manifest['jobs'])} jobs; Cp degree {degree}; plan: {mp}",flush=True)
    if not a.run: return
    for i, job in enumerate(manifest['jobs'],1):
        dest = Path(job['output']); metrics = dest/'metrics.json'
        if metrics.exists():
            completed = json.loads(metrics.read_text())
            if completed['data_sha256'] != manifest['data_sha256'] or completed['task_schema'] != 'ablation_v1':
                raise ValueError(f'Incompatible completed job {dest}')
            print(f'[{i}] Completed; skip {dest.relative_to(output)}',flush=True); continue
        dest.mkdir(parents=True,exist_ok=True)
        print(f'[{i}/{len(manifest["jobs"])}] {dest.relative_to(output)}',flush=True)
        with (dest/'train.log').open('w') as log:
            subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-m','scripts.summarize_ablation','--root',str(output)],cwd=ROOT,check=True)


if __name__ == '__main__': main()
