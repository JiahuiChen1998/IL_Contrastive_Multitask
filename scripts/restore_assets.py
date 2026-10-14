"""Verify and restore the separately uploaded supporting assets."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import zipfile
from scripts.verify_release import ROOT,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretraining-inputs',type=Path,help='Folder containing the two exact .txt.gz assets')
    p.add_argument('--checkpoints',type=Path,help='Path to graph_checkpoints_v1.zip')
    a=p.parse_args()
    if not a.pretraining_inputs and not a.checkpoints:p.error('Select an asset to restore')
    assets={r['filename']:r for r in json.loads((ROOT/'provenance/large_assets.json').read_text())}
    if a.pretraining_inputs:
        for name in ['pretrain_train_set.txt.gz','pretrain_val_set.txt.gz']:
            r=assets[name];source=a.pretraining_inputs/name;dest=ROOT/r['destination']
            if digest(source)!=r['sha256']:raise ValueError('Compressed input checksum mismatch: '+name)
            if dest.exists():
                if digest(dest)==r['uncompressed_sha256']:print('Already restored',dest);continue
                raise FileExistsError(dest)
            dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix(dest.suffix+'.partial')
            try:
                with gzip.open(source,'rb') as f,partial.open('xb') as g:shutil.copyfileobj(f,g,2**20)
                if digest(partial)!=r['uncompressed_sha256']:raise ValueError('Decompressed input checksum mismatch: '+name)
                partial.replace(dest)
            except BaseException:
                partial.unlink(missing_ok=True);raise
            print('Restored',dest)
    if a.checkpoints:
        if digest(a.checkpoints)!=assets['graph_checkpoints_v1.zip']['sha256']:raise ValueError('Checkpoint archive changed')
        expected={r['path']:r['sha256'] for r in json.loads((ROOT/'configs/optional_checkpoints.json').read_text())}
        with zipfile.ZipFile(a.checkpoints) as z:
            if set(z.namelist())!=set(expected) or len(z.namelist())!=len(expected):raise ValueError('Checkpoint archive membership changed')
            for name,sha in expected.items():
                dest=ROOT/name
                if not dest.resolve().is_relative_to(ROOT):raise ValueError('Unsafe archive path')
                if dest.exists():
                    if digest(dest)==sha:continue
                    raise FileExistsError(dest)
                dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix('.partial')
                try:
                    with z.open(name) as f,partial.open('xb') as g:shutil.copyfileobj(f,g,2**20)
                    if digest(partial)!=sha:raise ValueError('Individual checkpoint checksum mismatch: '+name)
                    partial.replace(dest)
                except BaseException:
                    partial.unlink(missing_ok=True);raise
            print(f'Restored {len(expected)} verified checkpoints')


if __name__=='__main__':main()
