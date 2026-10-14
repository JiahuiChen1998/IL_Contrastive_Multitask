"""Portable, explicit execution of one manuscript experiment job.

Commands are printed by default. --run starts one job, writes a new local plan,
and never overwrites the released results. Weighted fits use the unchanged 1M
trainer's checkpoint-replay safeguard with the original per-scale settings.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.verify_release import ROOT, digest


def job_id(job):
    return job.get('id',f"{job.get('task','ssl')}/{job.get('seed',42)}/{job.get('scale','')}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True,help='Study name from configs/experiments/index.json')
    p.add_argument('--job',help='Exact job ID; SSL jobs use the scale label')
    p.add_argument('--list',action='store_true',help='List job IDs without running')
    p.add_argument('--ssl',action='store_true',help='Select a pretraining job instead of a downstream fit')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--run',action='store_true',help='Actually run the selected job')
    a=p.parse_args();os.chdir(ROOT)
    path=ROOT/'configs/experiments'/(a.study+'.json')
    if not path.exists():p.error('Unknown study')
    config=json.loads(path.read_text())
    jobs=config.get('ssl_jobs',[]) if a.ssl else config['jobs']
    lookup={j['scale'] if a.ssl else job_id(j):j for j in jobs}
    if a.list:
        print('\n'.join(lookup));return
    if not a.job or a.job not in lookup:p.error('Use --list, then provide an exact --job')
    job=copy.deepcopy(lookup[a.job])
    dest=Path('reproduced')/a.study/('ssl' if a.ssl else 'fits')/a.job
    if any(x in ['..',''] for x in dest.parts):p.error('Invalid output')
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Choose a fresh output; this job already has local artifacts: '+str(dest))
    weighted='weights' in job
    if not weighted:
        cmd=job['command'];cmd[0]=sys.executable
        cmd[cmd.index('--output')+1]=str(dest)
        if '--device' in cmd:cmd[cmd.index('--device')+1]=a.device
        print(shlex.join(cmd),flush=True)
        if not a.run:return
        if digest('data/benchmarks/il_v1/observations.csv.gz')!=config['data_sha256']:raise ValueError('Frozen data changed')
        if a.ssl and not Path('data/pretraining/corpus_v1.sqlite').exists():raise FileNotFoundError('Restore the pretraining inputs and prepare corpus_v1.sqlite; see docs/REPRODUCING.md')
        checkpoint=job.get('checkpoint')
        if not checkpoint and '--pretrained-encoder' in cmd:checkpoint=cmd[cmd.index('--pretrained-encoder')+1]
        if checkpoint:
            expected=(Path(checkpoint).parent/'final_sha256.txt').read_text().strip()
            if digest(checkpoint)!=expected:raise ValueError('Encoder checksum mismatch')
        for r in json.loads((ROOT/'provenance/source_files.json').read_text()):
            if r['path'].endswith('.py') and digest(r['path'])!=r['sha256']:raise ValueError('Scientific source changed: '+r['path'])
        subprocess.run(cmd,check=True);return

    module_name='scripts.property_specialists_1m.train' if 'target' in job else 'scripts.application_weighted_1m.train'
    job['output']=str(dest)
    if 'target' not in job:
        job.setdefault('application','heat_transfer' if a.study.startswith('heat_transfer') else 'co2_absorbent')
        job.setdefault('selection_tasks',config.get('primary'))
    settings=copy.deepcopy(config['settings']);settings['device']=a.device
    settings.setdefault('heads',4);settings.setdefault('weight_decay',.0001);settings.setdefault('gradient_clip',5.)
    reference=json.loads((ROOT/'configs/reference_plans'/(a.study+'.json')).read_text())
    sources=json.loads((ROOT/'provenance/source_files.json').read_text())
    public_source_hash=hashlib.sha256((''.join(r['sha256'] for r in sorted(sources,key=lambda r:r['path']) if r['path'].endswith('.py'))+digest(__file__)).encode()).hexdigest()
    plan=dict(settings=settings,tasks=config['tasks'],smoke=False,checkpoint=config['checkpoint'],
        checkpoint_sha256=config['checkpoint_sha256'],data_sha256=config['data_sha256'],
        source_sha256=public_source_hash,reference_source_sha256=reference.get('source_sha256'),reload_tolerance_training_sd=config.get('reload_tolerance_training_sd',5e-6),
        publication_adapter=True,reference_config=str(path.relative_to(ROOT)),reference_config_sha256=digest(path),jobs=[job])
    print(json.dumps(dict(trainer=module_name,output=str(dest),seed=job['seed'],weights=job['weights'],
        selection_tasks=[job['target']] if 'target' in job else job['selection_tasks'],settings=settings,
        checkpoint=plan['checkpoint'],run=a.run),indent=2),flush=True)
    if not a.run:return
    def verify_public_plan(current):
        if current!=plan or digest(path)!=plan['reference_config_sha256']:raise ValueError('Public configuration changed')
        if digest('data/benchmarks/il_v1/observations.csv.gz')!=plan['data_sha256']:raise ValueError('Frozen data changed')
        if digest(plan['checkpoint'])!=plan['checkpoint_sha256']:raise ValueError('Encoder changed')
        if set(job['weights'])!=set(plan['tasks']) or abs(sum(job['weights'].values())-1)>1e-12:raise ValueError('Invalid loss allocation')
        for r in json.loads((ROOT/'provenance/source_files.json').read_text()):
            if r['path'].endswith('.py') and digest(r['path'])!=r['sha256']:raise ValueError('Scientific source changed: '+r['path'])
    verify_public_plan(plan)
    dest.mkdir(parents=True)
    (dest/'public_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    module=importlib.import_module(module_name)
    # Replace only the historical queue/input guard. The public guard above
    # checks frozen inputs, original sources and the released scientific config.
    # The original optimization/loss/selection/replay implementation is intact.
    module.verify_inputs=verify_public_plan
    module.fit(plan,job)


if __name__=='__main__':main()
