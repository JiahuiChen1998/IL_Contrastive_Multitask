"""Predict using either ordinary graph or historical plain heat-model checkpoints."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from il_mixer import predict as original
from scripts.heat_transfer_attention.model import HeatTransferMixer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    config=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['config']
    if 'feature_mode' in config:
        original.ILMixer=HeatTransferMixer
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    original.predict(a.checkpoint,pd.read_csv(a.input)).to_csv(out,index=False)
    print(f'Wrote predictions to {out}')


if __name__=='__main__':main()
