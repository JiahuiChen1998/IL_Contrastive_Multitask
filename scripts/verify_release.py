"""Verify the release and recompute held-out metrics without fitting or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from il_contrastive.train import fit_scalers, transformed
from scripts.audit_coverage_predictions import audit_predictions

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def restore_precision(frame, spec, std, metrics):
    """Recover short-decimal legacy CSV round trips only with four-metric proof.

    Same rule as the archived downstream-readiness audit. Source files remain
    untouched; a float32/16 interpretation must reproduce ALL four metrics.
    """
    y = frame.value.to_numpy(float); yt = transformed(y,spec)
    raw = frame.prediction.to_numpy(float); pt = frame.prediction_transformed.to_numpy(float)
    for dtype in [np.float64,np.float32,np.float16]:
        if max(abs(pt).max(),abs(raw).max()) > np.finfo(dtype).max:continue
        p=raw.astype(dtype).astype(float); t=pt.astype(dtype).astype(float)
        values=dict(mae=np.mean(abs(y-p)),rmse=np.sqrt(np.mean((y-p)**2)),
            transformed_mae=np.mean(abs(yt-t)),standardized_mse=np.mean((yt-t)**2)/std**2)
        if all(np.isclose(v,metrics[k],rtol=1e-12,atol=5e-14) for k,v in values.items()):
            return p,t,np.dtype(dtype).name,values
    raise ValueError('Saved predictions do not reproduce all four archived metrics')


def verify(predictions=False, output=None):
    manifest_path=ROOT/'provenance/release_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else json.loads((ROOT/'provenance/source_files.json').read_text())
    for r in manifest:
        if digest(ROOT/r['path']) != r['sha256']:raise ValueError('Release checksum mismatch: '+r['path'])
    frozen=ROOT/'data/benchmarks/il_v1'
    m=json.loads((frozen/'manifest.json').read_text())
    for name,sha in m['files'].items():
        if digest(frozen/name)!=sha:raise ValueError('Frozen data changed: '+name)
    truth=pd.read_csv(frozen/'observations.csv.gz',low_memory=False)
    if len(truth)!=74887 or truth.smiles.nunique()!=5458:raise ValueError('Unexpected dataset coverage')
    keys=['smiles','task','source_file','source_sheet','source_row']
    if truth.duplicated(keys).any() or truth.groupby('smiles').split.nunique().max()!=1:raise ValueError('Invalid frozen identities/splits')
    splits=truth[['smiles','split']].drop_duplicates().groupby('split').size().to_dict()
    if splits!=dict(train=4376,validation=546,test=536):raise ValueError('Frozen split counts changed')
    schema=json.loads((frozen/'task_schema.json').read_text())
    for p in (ROOT/'runs/pretraining_full_coverage_v1/ssl').glob('*/final.pt'):
        if digest(p)!=(p.parent/'final_sha256.txt').read_text().strip():raise ValueError('Encoder changed: '+str(p))
    registry=json.loads((ROOT/'configs/fit_registry.json').read_text())
    result=dict(passed=True,files=len(manifest),observations=len(truth),ils=truth.smiles.nunique(),
        splits=splits,fits=len(registry),prediction_audits=0,model_inference=False,training=False)
    records=[]
    if predictions:
        scalers=fit_scalers(truth[truth.split.eq('train')],schema)
        groups={(t,s):g for (t,s),g in truth.groupby(['task','split'])}
        for i,r in enumerate(registry):
            folder=ROOT/r['path']; metrics=json.loads((folder/'metrics.json').read_text())
            if metrics['data_sha256']!=m['data_sha256'] or metrics.get('subset_run'):raise ValueError('Data/smoke contamination: '+r['path'])
            for split in ['validation','test']:
                pred=pd.read_csv(folder/(split+'_predictions.csv.gz'),low_memory=False)
                if set(pred.task)!=set(r['tasks']):raise ValueError('Prediction task coverage changed')
                for task,frame in pred.groupby('task',sort=False):
                    audit=audit_predictions(frame,groups[task,split])
                    archive=metrics[split] if 'task' in metrics else metrics[split][task]
                    p,t,dtype,values=restore_precision(frame,schema[task],scalers[task]['std'],archive)
                    primary=abs(transformed(frame.value,schema[task])-t) if task=='viscosity' else abs(frame.value.to_numpy(float)-p)
                    macro=float(pd.DataFrame(dict(smiles=frame.smiles.to_numpy(),error=primary)).groupby('smiles').error.mean().mean())
                    records.append(dict(path=r['path'],seed=r['seed'],task=task,split=split,
                        families=';'.join(r['families']),primary_mae=float(primary.mean()),il_macro_mae=macro,
                        csv_roundtrip_dtype=dtype,**values,**audit))
                    if 'primary_mae' in archive and not np.isclose(primary.mean(),archive['primary_mae'],rtol=1e-12,atol=5e-14):raise ValueError('Primary metric differs')
                    if 'il_macro_mae' in archive and not np.isclose(macro,archive['il_macro_mae'],rtol=1e-12,atol=5e-14):raise ValueError('Equal-IL metric differs')
            if (i+1)%100==0:print(f'Checked {i+1}/{len(registry)} fits',flush=True)
        result['prediction_audits']=len(records)
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        if records:pd.DataFrame(records).to_csv(output/'recomputed_metrics.csv',index=False)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',action='store_true',help='Audit every held-out prediction row and recompute metrics')
    p.add_argument('--output',default=None,help='Optional destination for verification and recomputed tables')
    a=p.parse_args();verify(a.predictions,a.output)


if __name__=='__main__':main()
