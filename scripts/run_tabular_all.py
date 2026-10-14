"""All-property tabular experiments, with raw/basis/empirical-residual conditions."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from il_mixer.equations import study_tasks
from il_mixer.tabular import feature_matrix, TrainPreprocessor
from il_mixer.conditions import condition_basis, EmpiricalTrend, condition_domains
from il_mixer.train import transformed
from il_mixer.pretraining import digest
from scripts.run_tabular_benchmark import predict_model, scores, write_json, KEY

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/processed/observations.csv.gz'); p.add_argument('--output',default='runs/tabular_all_v1')
    p.add_argument('--tasks',default=','.join(study_tasks())); p.add_argument('--models',default='extratrees,tabpfn,limix')
    p.add_argument('--representations',default='descriptors,descriptors_fp'); p.add_argument('--modes',default='raw,basis,residual')
    p.add_argument('--seeds',default='42,43,44'); p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--estimators',type=int,default=4); p.add_argument('--tabpfn-version',default='V2',choices=['V2','V2_5','V2_6','V3'])
    p.add_argument('--limix-repo',default='external/LimiX'); p.add_argument('--limix-checkpoint',default='models/tabular/limix/LimiX-16M.ckpt')
    p.add_argument('--job-timeout',type=int,default=7200); p.add_argument('--run',action='store_true')
    p.add_argument('--smoke',action='store_true',help='At most 32 rows/split; results excluded from science')
    p.add_argument('--job-index',type=int,help=argparse.SUPPRESS)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    tasks=a.tasks.split(','); models=a.models.split(','); reps=a.representations.split(','); modes=a.modes.split(','); seeds=list(map(int,a.seeds.split(',')))
    for items,allowed in [(tasks,study_tasks()),(models,['extratrees','tabpfn','limix']),(reps,['descriptors','descriptors_fp']),(modes,['raw','basis','residual'])]:
        if not items or set(items)-set(allowed) or len(items)!=len(set(items)): p.error('Unknown or duplicate selection')
    if len(seeds)!=len(set(seeds)): p.error('Duplicate seeds')
    f=pd.read_csv(a.data,low_memory=False)
    if f.groupby('smiles').split.nunique().max()!=1 or f.duplicated(KEY).any(): raise ValueError('Invalid structure split/source identities')
    jobs=[dict(task=t,model=m,representation=r,mode=c,seed=s) for t in tasks for m in models for r in reps
          for c in (modes if study_tasks()[t]['temperature'] else ['raw']) for s in seeds]
    code=b''.join(Path(x).read_bytes() for x in [__file__,'il_mixer/conditions.py','il_mixer/tabular.py','scripts/run_tabular_benchmark.py','il_mixer/train.py','il_mixer/equations.py','il_mixer/tasks.py'])
    manifest=dict(schema='tabular_all_v1',settings={k:v for k,v in vars(a).items() if k not in ['run','job_index','job_timeout']},
                  data_sha256=digest(a.data),code_sha256=hashlib.sha256(code).hexdigest(),jobs=jobs,subset_run=a.smoke)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);plan=out/'plan.json'
    if plan.exists() and json.loads(plan.read_text())!=manifest: raise ValueError('Settings/code/data changed; use fresh output')
    write_json(plan,manifest)
    if a.job_index is None:
        f.groupby(['task','split']).agg(rows=('smiles','size'),ils=('smiles','nunique')).to_csv(out/'counts.csv')
        print(f'{len(jobs)} jobs; plan {plan}',flush=True)
        if not a.run:return
        failures=[];base=[x for x in sys.argv[1:] if x!='--run']
        for i,j in enumerate(jobs):
            d=out/j['task']/j['model']/j['representation']/j['mode']/f"seed_{j['seed']}"
            d.mkdir(parents=True,exist_ok=True)
            if (d/'metrics.json').exists():continue
            print(f'[{i+1}/{len(jobs)}] {j}',flush=True)
            with (d/'run.log').open('w') as log:
                try:
                    r=subprocess.run([sys.executable,'-m','scripts.run_tabular_all',*base,'--job-index',str(i)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.job_timeout)
                    status='completed' if r.returncode==0 else 'failed'
                except subprocess.TimeoutExpired:status='timeout'
            if status!='completed':failures.append(dict(**j,status=status,log=str(d/'run.log')))
        write_json(out/'failures.json',failures)
        print(f'Queue finished: {len(failures)} incomplete jobs; failures are not scores',flush=True)
        if failures:sys.exit(1)
        return
    j=jobs[a.job_index];spec=study_tasks()[j['task']];sub=f[f.task==j['task']].copy()
    if set(sub.split)!={'train','validation','test'}:raise ValueError('Missing split')
    if a.smoke:sub=pd.concat([x.sample(min(32,len(x)),random_state=42) for _,x in sub.groupby('split')]).reset_index(drop=True)
    masks={s:(sub.split==s).to_numpy() for s in ['train','validation','test']}
    x,names=feature_matrix(sub,spec,j['representation']);basis,bnames=condition_basis(sub,spec,j['task'])
    if j['mode']=='basis':x=np.column_stack([x,basis]);names+=bnames
    prep=TrainPreprocessor().fit(x[masks['train']]);x=prep.transform(x)
    y=transformed(sub.loc[masks['train'],'value'],spec);trend=None;basepred=np.zeros(len(sub))
    if j['mode']=='residual':
        trend=EmpiricalTrend().fit(basis[masks['train']],y);basepred=trend.predict(basis)
    residual=y-basepred[masks['train']];mean=float(residual.mean());std=max(float(residual.std()),1e-6)
    target_std=max(float(y.std()),1e-6);start=time.time()
    predictions,info=predict_model(a,j['model'],j['seed'],x[masks['train']],((residual-mean)/std).astype(np.float32),[x[masks[s]] for s in ['validation','test']])
    d=out/j['task']/j['model']/j['representation']/j['mode']/f"seed_{j['seed']}";d.mkdir(parents=True,exist_ok=True)
    result=dict(**j,data_sha256=manifest['data_sha256'],subset_run=a.smoke,task_schema='ablation_v1',model_info=info,elapsed_seconds=time.time()-start,
        feature_names=[n for n,k in zip(names,prep.keep) if k],preprocessing=prep.state(),residual_scaler=dict(mean=mean,std=std),
        target_std=target_std,empirical_trend=trend.state() if trend else None,basis_names=bnames,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','rdkit','scikit-learn']})
    for s,pred in zip(['validation','test'],predictions):
        frame=sub.loc[masks[s]].copy();pt=np.asarray(pred).reshape(-1)*std+mean+basepred[masks[s]]
        if len(pt)!=len(frame):raise ValueError('Prediction count mismatch')
        metric,raw=scores(frame,pt,spec,target_std);result[s]=metric;frame['prediction_transformed']=pt;frame['prediction']=raw
        if trend:
            frame['empirical_prediction_transformed']=basepred[masks[s]]
            result[s+'_empirical_only']=scores(frame,basepred[masks[s]],spec,target_std)[0]
        flags=condition_domains(sub.loc[masks['train']],frame,spec)
        for name,flag in flags.items():frame[name]=flag
        outside=np.logical_or.reduce(list(flags.values())) if flags else np.zeros(len(frame),dtype=bool)
        result[s+'_condition_support']={}
        for name,flag in [('within_train_ranges',~outside),('outside_train_ranges',outside)]:
            result[s+'_condition_support'][name]=scores(frame.loc[flag],pt[flag],spec,target_std)[0] if flag.any() else {'n':0}
        frame.to_csv(d/f'{s}_predictions.csv',index=False)
    write_json(d/'metrics.json',result);print(json.dumps({s:result[s] for s in ['validation','test']},indent=2))

if __name__=='__main__':main()
