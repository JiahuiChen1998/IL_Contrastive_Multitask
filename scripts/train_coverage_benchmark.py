"""Run the existing downstream trainer with a conservative per-process GPU cap."""
import runpy
import torch

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the production benchmark')
    torch.cuda.set_per_process_memory_fraction(.7)
    runpy.run_module('il_mixer.train', run_name='__main__')
