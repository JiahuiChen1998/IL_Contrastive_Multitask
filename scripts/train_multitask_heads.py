"""Run the unchanged trainer with resource/provenance recording for the new study."""
import json
from pathlib import Path
import platform
import resource
import runpy
import sys
import time

import numpy as np
import pandas as pd
import torch
from il_contrastive.rdkit_logging import suppress_isolated_hydrogen_warning
from scripts.pretrain_full_coverage import atomic_json


def main():
    argv=sys.argv[1:]
    out=Path(argv[argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
    device=argv[argv.index('--device')+1]
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use native WSL for production')
        torch.cuda.set_per_process_memory_fraction(.7)
    suppress_isolated_hydrogen_warning()
    from scripts.run_multitask_heads_study import training_fingerprint
    state=dict(started_unix=time.time(),state='running',device=device,
        source_sha256=training_fingerprint(),versions=dict(python=platform.python_version(),
        torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__))
    atomic_json(state,out/'execution.json')
    try:
        runpy.run_module('il_contrastive.train',run_name='__main__')
        state['state']='complete'
    except BaseException as error:
        state.update(state='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state.update(finished_unix=time.time(),wall_seconds=time.time()-state['started_unix'],
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else 0,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.startswith('cuda') else 0)
        atomic_json(state,out/'execution.json')


if __name__=='__main__':main()
