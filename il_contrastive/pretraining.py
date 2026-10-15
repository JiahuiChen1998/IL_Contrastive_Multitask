"""Small, checkpoint-compatible graph contrastive pretraining utilities.

MolCLR-inspired, independently implemented for ILMixer; no upstream weights used.
"""
import hashlib
import itertools
import random
import sqlite3
from pathlib import Path
import torch
from torch.nn import functional as F

TRUNK = ('embeddings.', 'edge_embedding.', 'layers.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_encoder(model, path, include_cross=False):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('format') != 'ilmixer_ssl_v1':
        raise ValueError('Expected an ILMixer SSL checkpoint, not upstream MolCLR/MOFormer weights')
    prefixes = TRUNK + (('cross.', 'cross_norm.') if include_cross else ())
    own = model.state_dict()
    expected = {k for k in own if k.startswith(prefixes)}
    state = {k: v for k, v in ck['encoder'].items() if k.startswith(prefixes)}
    if set(state) != expected or any(state[k].shape != own[k].shape for k in expected):
        raise ValueError('Encoder architecture mismatch: hidden/layers/cross must match checkpoint')
    model.load_state_dict(state, strict=False)
    return {'path': str(Path(path).resolve()), 'sha256': digest(path),
            'include_cross': include_cross, 'loaded_tensors': len(state),
            'pretraining_manifest': ck['manifest']}


def masked_view(g, rate):
    """Mask element identity only; retain charges, stereo, all fragments and bonds."""
    if not 0 < rate < 1:
        raise ValueError('Mask rate must be between zero and one')
    out = dict(g)
    out['x'] = g['x'].clone()
    # Mask at least one atom in every molecule. Atom 0 is the existing unknown slot.
    for i in range(int(g['batch'].max()) + 1):
        ids = (g['batch'] == i).nonzero().flatten()
        chosen = ids[torch.randperm(len(ids), device=ids.device)[:max(1, round(rate * len(ids)))]]
        out['x'][chosen, 0] = 0
    return out


def contrastive_loss(a, b, temperature=.1):
    if len(a) < 2 or temperature <= 0:
        raise ValueError('Contrastive loss needs at least two distinct structures and positive temperature')
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
    labels = (torch.arange(len(z), device=z.device) + len(a)) % len(z)
    return F.cross_entropy(logits, labels)


def smiles_batches(db, split, batch_size, seed, pairs_only=False, limit=0, buffer_size=4096):
    """Read an indexed, hash-ordered corpus with bounded memory; no full-file list."""
    rng = random.Random(seed)
    uri = Path(db).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as conn:
        sql = 'SELECT smiles FROM molecules WHERE split=?'
        if pairs_only:
            sql += ' AND pair=1'
        count = conn.execute(sql.replace('SELECT smiles', 'SELECT count(*)'), (split,)).fetchone()[0]
        count = min(count, limit) if limit else count
        start = rng.randrange(count) if count else 0
        sql += ' ORDER BY shuffle_key'
        # Rotate each epoch before buffer shuffling, so a step budget shorter than
        # a full traversal does not repeatedly expose only the beginning of the corpus.
        cursor = itertools.chain(conn.execute(sql+' LIMIT ? OFFSET ?', (split, count-start, start)),
                                 conn.execute(sql+' LIMIT ?', (split, start)))
        buffer = []
        for _ in range(buffer_size):
            row = next(cursor, None)
            if row is None: break
            buffer.append(row[0])
        batch = []
        while buffer:
            idx = rng.randrange(len(buffer))
            batch.append(buffer[idx])
            row = next(cursor, None)
            if row is None: buffer.pop(idx)
            else: buffer[idx] = row[0]
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) >= 2:
            yield batch
