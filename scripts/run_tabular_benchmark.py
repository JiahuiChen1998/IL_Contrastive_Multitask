"""Paired, resumable tabular benchmark on the saved global IL-group split."""
import argparse,hashlib,importlib.metadata,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from il_contrastive.equations import study_tasks
from il_contrastive.train import transformed
from il_contrastive.tabular import feature_matrix,TrainPreprocessor,train_only_limix_preprocessing
ROOT=Path(__file__).resolve().parents[1]
KEY=['smiles','task','source_file','source_sheet','source_row']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def scores(frame,pred_t,spec,scale):
    y=frame.value.to_numpy();yt=transformed(y,spec);p=np.exp(pred_t) if spec['transform']=='log' else pred_t
    if not np.isfinite(p).all():raise ValueError('Nonfinite predictions; no clipping applied')
    e=np.abs(y-p);primary=np.abs(yt-pred_t) if frame.task.iloc[0]=='viscosity' else e
    den=float(np.square(y-y.mean()).sum())
    return dict(n=len(y),n_ils=frame.smiles.nunique(),mae=float(e.mean()),rmse=float(np.sqrt(np.square(y-p).mean())),
      r2=float(1-np.square(y-p).sum()/den) if den else None,transformed_mae=float(np.abs(yt-pred_t).mean()),
      standardized_mse=float(np.square(yt-pred_t).mean()/scale**2),primary_mae=float(primary.mean()),
      il_macro_mae=float(pd.DataFrame({'smiles':frame.smiles.to_numpy(),'error':primary}).groupby('smiles').error.mean().mean()),unit=spec['unit']),p

def predict_model(a,name,seed,x,y,queries):
    if name=='extratrees':
        from sklearn.ensemble import ExtraTreesRegressor
        model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=1.,random_state=seed,n_jobs=a.threads)
        model.fit(x,y);return [model.predict(q) for q in queries],{'trees':500,'min_samples_leaf':2,'pretraining':'none'}
    if name=='tabpfn':
        os.environ.setdefault('TABPFN_MODEL_CACHE_DIR',str(ROOT/'models/tabular/tabpfn'))
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        model=TabPFNRegressor.create_default_for_version(getattr(ModelVersion,a.tabpfn_version),device=a.device,n_estimators=a.estimators,random_state=seed,n_preprocessing_jobs=1)
        model.fit(x,y)
        path=Path(model.model_path)
        info={'version':a.tabpfn_version,'estimators':a.estimators,'checkpoint':str(path),'checkpoint_sha256':sha(path) if path.is_file() else None}
        return [np.asarray(model.predict(q)) for q in queries],info
    if name=='limix':
        repo=Path(a.limix_repo).resolve();checkpoint=Path(a.limix_checkpoint).resolve()
        if not checkpoint.is_file():raise FileNotFoundError(f'Obtain the official LimiX checkpoint first: {checkpoint}')
        sys.path.insert(0,str(repo))
        from inference.predictor import LimiXPredictor
        config=json.loads((repo/'config/reg_default_noretrieval.json').read_text())
        # Preserve equal representation of the upstream quantile and power pipelines.
        if a.estimators%2 or a.estimators>len(config):raise ValueError('LimiX estimators must be even and <= 8')
        config=config[:a.estimators//2]+config[len(config)//2:len(config)//2+a.estimators//2]
        model=LimiXPredictor(device=torch.device(a.device),model_path=str(checkpoint),inference_config=config,seed=seed,mix_precision=a.device!='cpu')
        results=[]
        for q in queries:
            with train_only_limix_preprocessing(model,len(y)):
                p=model.predict(x.copy(),y.copy(),q.copy(),task_type='Regression')
            results.append(p.detach().cpu().numpy().reshape(-1) if torch.is_tensor(p) else np.asarray(p).reshape(-1))
        info={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'config':config,'preprocessing':'training context only; adapted from upstream',
              'git_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()}
        return results,info
    raise ValueError(name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/tabular_small_v1');p.add_argument('--data',default='data/processed/observations.csv.gz')
    p.add_argument('--models',default='extratrees,tabpfn,limix');p.add_argument('--representations',default='descriptors,descriptors_fp')
    p.add_argument('--seeds',default='42,43,44');p.add_argument('--tasks',default='');p.add_argument('--threshold',type=int,default=500)
    p.add_argument('--count-by',choices=['unique_ils','rows'],default='unique_ils');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--estimators',type=int,default=4);p.add_argument('--tabpfn-version',choices=['V2','V2_5','V2_6','V3'],default='V2')
    p.add_argument('--limix-repo',default='external/LimiX');p.add_argument('--limix-checkpoint',default='models/tabular/limix/LimiX-16M.ckpt')
    p.add_argument('--run',action='store_true');p.add_argument('--job-timeout',type=int,default=1800);p.add_argument('--job-index',type=int,default=None,help=argparse.SUPPRESS)
    a=p.parse_args();torch.set_num_threads(a.threads);out=Path(a.output).resolve();data=Path(a.data).resolve();f=pd.read_csv(data,low_memory=False)
    if f.groupby('smiles').split.nunique().max()!=1:raise ValueError('Structure leakage in supplied data')
    if f.duplicated(KEY).any():raise ValueError('Duplicate source row identity')
    specs=study_tasks();counts=f.groupby('task').agg(rows=('value','size'),unique_ils=('smiles','nunique'))
    tasks=a.tasks.split(',') if a.tasks else sorted(counts.index[counts[a.count_by]<a.threshold])
    models=a.models.split(',');reps=a.representations.split(',');seeds=list(map(int,a.seeds.split(',')))
    if set(tasks)-set(specs) or set(models)-{'extratrees','tabpfn','limix'} or set(reps)-{'descriptors','descriptors_fp'}:p.error('Unknown task/model/representation')
    if any(len(x)!=len(set(x)) for x in [tasks,models,reps,seeds]) or not tasks:p.error('Empty tasks or duplicate entries')
    jobs=[dict(task=t,model=m,representation=r,seed=s) for t in tasks for m in models for r in reps for s in seeds]
    settings={k:v for k,v in vars(a).items() if k not in ['run','job_index','job_timeout']}
    manifest=dict(schema='tabular_small_v1',data_sha256=sha(data),settings=settings,jobs=jobs,
        code_sha256=hashlib.sha256((ROOT/'il_contrastive/tabular.py').read_bytes()+Path(__file__).read_bytes()).hexdigest(),
        criterion='Strictly below threshold, all-split coverage; train/validation/test counts also reported',
        task_schema='ablation_v1',preprocessing='train only',refit_train_plus_validation=False)
    out.mkdir(parents=True,exist_ok=True);plan=out/'plan.json'
    if plan.exists() and json.loads(plan.read_text())!=manifest:raise ValueError('Plan/code changed; choose a new output directory')
    write_json(plan,manifest);counts.to_csv(out/'property_counts.csv');f.groupby(['task','split']).agg(rows=('value','size'),ils=('smiles','nunique')).to_csv(out/'split_counts.csv')
    if a.job_index is None:
        print(f'{len(jobs)} jobs, tasks={tasks}; fixed original IL split. Plan: {plan}',flush=True)
        if not a.run:return
        base=[s for s in sys.argv[1:] if s!='--run']
        failures=[]
        for i,job in enumerate(jobs):
            d=out/job['task']/job['model']/job['representation']/f"seed_{job['seed']}";d.mkdir(parents=True,exist_ok=True)
            if (d/'metrics.json').exists():continue
            print(f'[{i+1}/{len(jobs)}] {job}',flush=True)
            with (d/'run.log').open('w') as log:
                try:r=subprocess.run([sys.executable,'-m','scripts.run_tabular_benchmark',*base,'--job-index',str(i)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.job_timeout);status='completed' if r.returncode==0 else 'failed'
                except subprocess.TimeoutExpired:status='timeout'
            if status!='completed':failures.append(dict(**job,status=status,log=str(d/'run.log')))
        write_json(out/'failures.json',failures)
        print(f'Finished queue; {len(failures)} incomplete jobs. No failed results imputed.',flush=True)
        return
    job=jobs[a.job_index];t=job['task'];spec=specs[t];sub=f[f.task==t].copy();masks={s:(sub.split==s).to_numpy() for s in ['train','validation','test']}
    x,names=feature_matrix(sub,spec,job['representation']);prep=TrainPreprocessor().fit(x[masks['train']]);x=prep.transform(x)
    y=transformed(sub.loc[masks['train'],'value'],spec);mean=float(y.mean());std=max(float(y.std()),1e-6);yn=((y-mean)/std).astype(np.float32)
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);start=time.time()
    predictions,info=predict_model(a,job['model'],job['seed'],x[masks['train']],yn,[x[masks[s]] for s in ['validation','test']])
    d=out/t/job['model']/job['representation']/f"seed_{job['seed']}";d.mkdir(parents=True,exist_ok=True)
    result=dict(**job,data_sha256=manifest['data_sha256'],task_schema='ablation_v1',elapsed_seconds=time.time()-start,model_info=info,
       feature_names=[n for n,k in zip(names,prep.keep) if k],preprocessing=prep.state(),target_scaler={'mean':mean,'std':std},
       package_versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','rdkit','scikit-learn']})
    if job['model']=='tabpfn':result['package_versions']['tabpfn']=importlib.metadata.version('tabpfn')
    for split,normalized in zip(['validation','test'],predictions):
        frame=sub.loc[masks[split]].copy();pt=np.asarray(normalized).reshape(-1)*std+mean
        if len(pt)!=len(frame):raise ValueError('Prediction count mismatch')
        metric,raw=scores(frame,pt,spec,std);result[split]=metric;frame['prediction_transformed']=pt;frame['prediction']=raw;frame.to_csv(d/f'{split}_predictions.csv',index=False)
    write_json(d/'metrics.json',result);print(json.dumps({s:result[s] for s in ['validation','test']},indent=2))
if __name__=='__main__':main()
