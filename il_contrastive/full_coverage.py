"""Bounded-memory, exactly resumable traversal of nested hash-selected subsets."""
import copy
import json
from pathlib import Path
import random
import sqlite3


def corpus_metadata(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
        return json.loads(db.execute('SELECT json FROM metadata').fetchone()[0])


class FullCoverageSampler:
    """Each pass emits each selected row once, including the last partial batch.

    Checkpoints include the shuffle buffer, random state and last *fetched* SQL
    key. Resuming uses an indexed keyset query, not a replay from row zero.
    """
    def __init__(self, path, split, count, batch_size=128, seed=42, buffer_size=8192, state=None):
        if count < 2 or batch_size < 2 or buffer_size < 1:
            raise ValueError('At least two structures per contrastive batch/pool are required')
        self.config = dict(split=split, count=count, batch_size=batch_size, seed=seed, buffer_size=buffer_size)
        self.db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
        self.db.execute('PRAGMA cache_size=-65536')  # 64 MiB SQLite cache.
        self.db.execute('PRAGMA mmap_size=0')
        self.rng = random.Random(seed)
        self.buffer = []
        self.fetched = self.emitted = 0
        self.last_key = None
        if state is not None:
            if state['config'] != self.config:
                raise ValueError('Sampler configuration changed on resume')
            self.buffer = copy.deepcopy(state['buffer'])
            self.fetched, self.emitted = state['fetched'], state['emitted']
            self.last_key = state['last_key']
            self.rng.setstate(state['rng'])
            if self.emitted+len(self.buffer) != self.fetched or not 0 <= self.fetched <= count:
                raise ValueError('Inconsistent sampler checkpoint')
        remaining = count-self.fetched
        if self.last_key is None:
            self.cursor = self.db.execute('SELECT shuffle_key, smiles FROM molecules WHERE split=? '
                                          'ORDER BY shuffle_key, smiles LIMIT ?', (split, remaining))
        else:
            self.cursor = self.db.execute('SELECT shuffle_key, smiles FROM molecules WHERE split=? '
                'AND (shuffle_key,smiles)>(?,?) ORDER BY shuffle_key,smiles LIMIT ?',
                (split, *self.last_key, remaining))
        if state is None:
            for _ in range(min(buffer_size, count)):
                self.buffer.append(self._fetch())

    def _fetch(self):
        row = self.cursor.fetchone()
        if row is None:
            raise ValueError('Corpus has fewer rows than its declared subset')
        self.fetched += 1
        self.last_key = tuple(row)
        return tuple(row)

    def __iter__(self):
        return self

    def __next__(self):
        remaining = self.config['count']-self.emitted
        if not remaining:
            raise StopIteration
        b = self.config['batch_size']
        # Merge an otherwise unusable singleton into the preceding batch.
        target = b+1 if remaining == b+1 else min(b, remaining)
        result = []
        for _ in range(target):
            i = self.rng.randrange(len(self.buffer))
            _, smiles = self.buffer[i]
            result.append(smiles)
            if self.fetched < self.config['count']:
                self.buffer[i] = self._fetch()
            else:
                self.buffer.pop(i)
            self.emitted += 1
        return result

    def state_dict(self):
        return copy.deepcopy(dict(config=self.config, buffer=self.buffer, fetched=self.fetched,
                                  emitted=self.emitted, last_key=self.last_key, rng=self.rng.getstate()))

    def close(self):
        self.db.close()
