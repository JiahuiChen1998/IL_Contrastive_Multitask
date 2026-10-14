"""Full-pass contrastive pretraining with recoverable within-pass checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import torch
from torch import nn
from il_mixer.full_coverage import FullCoverageSampler, corpus_metadata
from il_mixer.graph import pack, graph
from il_mixer.model import ILMixer
from il_mixer.pretraining import masked_view, contrastive_loss

SOURCE_FILES = ['il_mixer/full_coverage.py', 'il_mixer/graph.py', 'il_mixer/model.py',
                'il_mixer/pretraining.py', 'scripts/pretrain_full_coverage.py']


def source_hash():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def atomic_json(value, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def atomic_checkpoint(value, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def memory():
    rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    return dict(rss_gib=rss/2**30, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                available_gib=available/2**30,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', default='data/pretraining/corpus_v1.sqlite')
    p.add_argument('--output', required=True)
    p.add_argument('--structures', type=int, required=True)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--buffer-size', type=int, default=8192)
    p.add_argument('--validation-structures', type=int, default=2560)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mask-rate', type=float, default=.15)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--gpu-memory-fraction', type=float, default=.7)
    p.add_argument('--rss-limit-gib', type=float, default=24.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-steps', type=int, default=0, help='Controlled interruption for smoke/resume verification')
    return p


def run(a):
    if min(a.structures, a.batch_size, a.validation_structures) < 2 or min(a.passes, a.checkpoint_every, a.log_every) < 1:
        raise ValueError('Invalid pool, batch or training budget')
    if a.hidden % 4 or a.layers < 1 or not 0 < a.gpu_memory_fraction <= .9:
        raise ValueError('Invalid encoder or GPU limit')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = corpus_metadata(a.corpus)
    counts = {s: sum(x['n'] for x in meta['counts'] if x['split'] == s) for s in ['train', 'validation']}
    if a.structures > counts['train'] or a.validation_structures > counts['validation']:
        raise ValueError('Requested subset exceeds the prepared corpus')
    settings = {k: v for k, v in vars(a).items() if k not in ['resume', 'stop_after_steps']}
    manifest = dict(settings=settings, corpus=meta, source_sha256=source_hash(), torch_version=str(torch.__version__),
                    protocol='Full nested subsets; each structure once per pass; final-pass encoder used for downstream; no property labels')
    if (out/'plan.json').exists():
        if not a.resume or json.loads((out/'plan.json').read_text()) != manifest:
            raise ValueError('Use --resume with the identical plan, or a new output')
    else:
        if a.resume:
            raise ValueError('No run to resume')
        atomic_json(manifest, out/'plan.json')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    encoder = ILMixer({}, hidden=a.hidden, layers=a.layers).to(a.device)
    projector = nn.Sequential(nn.Linear(2*a.hidden, 2*a.hidden), nn.GELU(), nn.Linear(2*a.hidden, 128)).to(a.device)
    parameters = list(encoder.parameters())+list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=1e-4)
    epoch = 0; step = 0; history = []; sampler_state = None
    loss_sum = 0.; epoch_steps = 0; best = float('inf'); elapsed = 0.
    if a.resume and (out/'last.pt').exists():
        ck = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        encoder.load_state_dict(ck['encoder']); projector.load_state_dict(ck['projector'])
        optimizer.load_state_dict(ck['optimizer'])
        epoch, step, history = ck['completed_passes'], ck['step'], ck['history']
        sampler_state = ck['sampler']; loss_sum, epoch_steps = ck['loss_sum'], ck['epoch_steps']
        best, elapsed = ck['best'], ck['elapsed_seconds']
        torch.set_rng_state(ck['torch_rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif a.resume:
        # Plan exists but first checkpoint was never written. Deterministic restart.
        print('No checkpoint yet; restarting the untouched run from initialization.', flush=True)
    start = time.monotonic(); sampler = None; stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    previous = {s: signal.signal(s, request_stop) for s in [signal.SIGINT, signal.SIGTERM]}

    def save(filename='last.pt'):
        state = dict(format='ilmixer_ssl_v1', encoder=encoder.state_dict(), projector=projector.state_dict(),
                     optimizer=optimizer.state_dict(), manifest=manifest, completed_passes=epoch, step=step,
                     history=history, sampler=sampler.state_dict() if sampler else None,
                     loss_sum=loss_sum, epoch_steps=epoch_steps, best=best,
                     elapsed_seconds=elapsed+time.monotonic()-start, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [])
        atomic_checkpoint(state, out/filename)

    def status(state):
        seen = epoch*a.structures+(sampler.emitted if sampler else 0)
        value = dict(state=state, completed_passes=epoch, pass_in_progress=min(epoch+1, a.passes),
                     pass_structures_seen=sampler.emitted if sampler else 0, unique_structures_seen=min(a.structures, seen),
                     structures=a.structures, presentations=seen, target_presentations=a.passes*a.structures,
                     optimizer_steps=step, elapsed_seconds=elapsed+time.monotonic()-start, **memory())
        atomic_json(value, out/'status.json')
        return value

    def ssl_loss(smiles):
        g = pack(smiles, a.device, require_ions=False)
        z1 = projector(encoder.encode(masked_view(g, a.mask_rate)))
        z2 = projector(encoder.encode(masked_view(g, a.mask_rate)))
        return contrastive_loss(z1, z2, a.temperature), z1

    def validate():
        encoder.eval(); projector.eval(); values = []; spread = []
        validation = FullCoverageSampler(a.corpus, 'validation', a.validation_structures, a.batch_size, 9917, a.buffer_size)
        devices = [torch.device(a.device).index or 0] if a.device.startswith('cuda') else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(9917)
                for smiles in validation:
                    loss, z = ssl_loss(smiles)
                    values.append(float(loss)); spread.append(float(z.std(dim=0, unbiased=False).mean()))
        finally:
            validation.close()
        return sum(values)/len(values), sum(spread)/len(spread)

    try:
        while epoch < a.passes:
            sampler = FullCoverageSampler(a.corpus, 'train', a.structures, a.batch_size,
                                          a.seed+epoch, a.buffer_size, sampler_state)
            sampler_state = None
            encoder.train(); projector.train()
            for smiles in sampler:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = ssl_loss(smiles)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite SSL loss')
                loss.backward(); nn.utils.clip_grad_norm_(parameters, 5.); optimizer.step()
                step += 1; epoch_steps += 1; loss_sum += float(loss.detach())
                if step % a.log_every == 0:
                    progress = status('running')
                    print(json.dumps(progress | {'train_loss': loss_sum/epoch_steps}), flush=True)
                if step % a.checkpoint_every == 0:
                    save()
                    mem = memory()
                    if mem['rss_gib'] > a.rss_limit_gib or mem['available_gib'] < 4:
                        status('memory_stop')
                        raise MemoryError('Host memory guard stopped training after saving a checkpoint')
                if stop_requested or (a.stop_after_steps and step >= a.stop_after_steps):
                    save(); status('paused'); return False
            if sampler.emitted != a.structures:
                raise AssertionError('Pass did not visit every selected structure')
            val, spread = validate()
            if not torch.isfinite(torch.tensor(val)):
                raise FloatingPointError('Nonfinite SSL validation loss')
            history.append(dict(pass_number=epoch+1, train_loss=loss_sum/epoch_steps, validation_loss=val,
                                projection_std=spread, structures_seen=sampler.emitted, optimizer_steps=epoch_steps))
            improved = val < best; best = min(best, val)
            sampler.close(); sampler = None; epoch += 1; loss_sum = 0.; epoch_steps = 0
            save(); save(f'pass_{epoch:02d}.pt')
            if improved:
                save('best_ssl.pt')
            atomic_json(history, out/'history.json')
            print(json.dumps(status('running') | {'validation_loss': val, 'projection_std': spread}), flush=True)
            graph.cache_clear()
        save('final.pt')
        status('complete')
        return True
    except Exception as error:
        # Keep the previous valid checkpoint. Do not save a partially updated batch.
        atomic_json(dict(error=type(error).__name__, message=str(error), step=step,
                         last_checkpoint=str(out/'last.pt')), out/'error.json')
        raise
    finally:
        if sampler is not None:
            sampler.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    finished = run(parser().parse_args())
    if not finished:
        raise SystemExit(75)
