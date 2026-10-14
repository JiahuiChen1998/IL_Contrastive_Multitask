"""Stream SMILES into a deduplicated disk database; never load the corpus into RAM."""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import pandas as pd
from rdkit import Chem, RDLogger
from il_mixer.graph import canonicalize, graph
from il_mixer.pretraining import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='pre-train strategy/pretrian data/pretrain_train_set.txt')
    p.add_argument('--validation', default='pre-train strategy/pretrian data/pretrain_val_set.txt')
    p.add_argument('--downstream', default='data/processed/observations.csv.gz')
    p.add_argument('--output', default='data/pretraining/corpus.sqlite')
    p.add_argument('--max-lines', type=int, default=0, help='Per-file prefix limit, smoke only; zero scans entire files')
    p.add_argument('--max-atoms', type=int, default=160)
    a = p.parse_args()
    if a.max_lines < 0 or a.max_atoms < 2: p.error('Invalid limits')
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix('.partial').exists():
        raise FileExistsError('Choose a fresh output; partial preparation must be removed explicitly before restarting')
    # Exclude every downstream complete structure from both SSL splits. Labels are not read.
    observed = pd.read_csv(a.downstream, usecols=['smiles'])
    excluded = {canonicalize(s) for s in observed.smiles.unique()}
    conn = sqlite3.connect(out.with_suffix('.partial'))
    conn.execute('CREATE TABLE molecules (smiles TEXT PRIMARY KEY, split TEXT, pair INTEGER, shuffle_key TEXT)')
    report = {'arguments': vars(a), 'downstream_sha256': digest(a.downstream),
              'excluded_complete_downstream_structures': len(excluded), 'sources': {},
              'scope': 'prefix_smoke' if a.max_lines else 'full',
              'overlap_policy': 'exclude all downstream complete structures; validation wins SSL duplicates; shared ions allowed'}
    RDLogger.DisableLog('rdApp.*')
    # Validation first gives it priority in exact canonical train/validation collisions.
    for split, path in [('validation', a.validation), ('train', a.train)]:
        counts = Counter(); content = hashlib.sha256(); source = Path(path)
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rb') as f:
            rows = itertools.islice(f, a.max_lines) if a.max_lines else f
            for line in rows:
                counts['lines'] += 1; content.update(line)
                fields = line.decode('utf-8').strip().split()
                if not fields: counts['blank'] += 1; continue
                s = fields[0] if len(fields) == 1 else fields[1] if len(fields) == 2 and fields[0].isdigit() else None
                if s is None: counts['unsupported_record'] += 1; continue
                try:
                    s = canonicalize(s); mol = Chem.MolFromSmiles(s)
                    if mol.GetNumAtoms() > a.max_atoms: counts['too_large'] += 1; continue
                    if any(x.GetAtomicNum() == 0 for x in mol.GetAtoms()): counts['wildcard'] += 1; continue
                    # Check encoder feature compatibility, without retaining millions of graphs.
                    graph(s, require_ions=False); graph.cache_clear()
                except (ValueError, RuntimeError): counts['invalid_or_unsupported'] += 1; continue
                if s in excluded: counts['downstream_excluded'] += 1; continue
                charges = [Chem.GetFormalCharge(x) for x in Chem.GetMolFrags(mol, asMols=True)]
                pair = int(min(charges) < 0 < max(charges) and sum(charges) == 0)
                key = hashlib.sha256(s.encode()).hexdigest()
                c = conn.execute('INSERT OR IGNORE INTO molecules VALUES (?, ?, ?, ?)', (s, split, pair, key))
                counts['accepted_pair' if pair else 'accepted_other'] += c.rowcount
                counts['duplicates'] += 1 - c.rowcount
                if counts['lines'] % 100000 == 0:
                    conn.commit(); print(split, dict(counts), flush=True)
        conn.commit()
        report['sources'][split] = {'path': str(source.resolve()), 'bytes': source.stat().st_size,
            'processed_uncompressed_sha256': content.hexdigest(), 'counts': dict(counts)}
    conn.execute('CREATE INDEX sample_order ON molecules(split, shuffle_key)')
    conn.execute('CREATE INDEX pair_order ON molecules(split, pair, shuffle_key)')
    report['counts'] = [dict(split=s, pair=pair, n=n) for s,pair,n in conn.execute('SELECT split,pair,count(*) FROM molecules GROUP BY split,pair')]
    conn.execute('CREATE TABLE metadata (json TEXT)')
    conn.execute('INSERT INTO metadata VALUES (?)', (json.dumps(report, sort_keys=True),))
    conn.commit(); conn.close()
    out.with_suffix('.partial').replace(out)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
