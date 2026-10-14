"""Rebuild per-configuration seed summaries and validation candidates from metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.verify_release import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reproduced/summaries')
    args=parser.parse_args();rows=[]
    registry=json.loads((ROOT/'configs/fit_registry.json').read_text())
    for r in registry:
        m=json.loads((ROOT/r['path']/'metrics.json').read_text())
        for family in r['families']:
            if family=='historical_scratch':
                label=r['labels'][-1];label=dict(variant=label['variant'],seed=label['seed'])
            else:
                # The registry stores labels in the same order as families;
                # repeated aliases are consolidated by path below.
                candidates=[j for j in json.loads((ROOT/'configs/experiments'/(family+'.json')).read_text())['jobs'] if j['output']==r['path']] if family!='tabular_final_v2' else [dict(model=m['model'],representation=m['representation'],seed=m['seed'])]
                label=candidates[0]
            config={k:label[k] for k in ['scale','initialization','scope','head_family','variant','model','representation','arm','application','target'] if k in label}
            if family.startswith(('heat_transfer','co2_absorbent')):config['application']='heat_transfer' if family.startswith('heat') else 'co2_absorbent'
            for split in ['validation','test']:
                for task in r['tasks']:
                    v=m[split] if 'task' in m else m[split][task]
                    rows.append(dict(study=family,configuration=json.dumps(config,sort_keys=True),task=task,seed=r['seed'],split=split,
                        primary_mae=v['transformed_mae'] if task=='viscosity' else v['mae'],standardized_mse=v['standardized_mse'],path=r['path']))
    frame=pd.DataFrame(rows);keys=['study','configuration','task','seed','split']
    if frame.duplicated(keys).any():raise ValueError('Duplicate fitted seed in a published configuration')
    agg=frame.groupby(keys[:3]+['split'])[['primary_mae','standardized_mse']].agg(['mean','std','count']).reset_index()
    agg.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in agg.columns]
    validation=agg[agg.split.eq('validation')].copy()
    validation=validation.sort_values(['study','task','standardized_mse_mean','configuration'])
    # Target specialists choose allocations ONLY within the specified target.
    eligible=validation.apply(lambda r:'target' not in json.loads(r.configuration) or json.loads(r.configuration)['target']==r.task,axis=1)
    validation=validation[eligible]
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'per_seed.csv',index=False);agg.to_csv(out/'aggregate_statistics.csv',index=False)
    validation.to_csv(out/'validation_candidates.csv',index=False)
    print(f'Wrote {len(frame)} seed/property/split records and {len(agg)} summaries to {out}')


if __name__=='__main__':main()
