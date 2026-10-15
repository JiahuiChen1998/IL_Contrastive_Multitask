"""Auditable XLSX import and globally consistent structure-grouped splits."""
import argparse
import hashlib
import json
import math
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from .graph import canonicalize, graph
from .tasks import TASKS

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook_rows(path, sheet):
    """Stream real cells, ignoring Excel's inflated used range and blank styling."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//s:t', NS))
                       for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = {s.attrib['name']: s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
                  for s in wb.find('s:sheets', NS)}
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        target = rels[sheets[sheet]]
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        headers = None
        for _, row in ET.iterparse(z.open(member), events=['end']):
            if row.tag != '{' + NS['s'] + '}row':
                continue
            values = {}
            for cell in row:
                col = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                v = cell.find('s:v', NS)
                if cell.attrib.get('t') == 's' and v is not None:
                    values[col] = strings[int(v.text)]
                elif cell.attrib.get('t') == 'inlineStr':
                    values[col] = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                elif v is not None:
                    values[col] = v.text
            if headers is None:
                headers = values
            elif values:
                yield int(row.attrib['r']), {name: values.get(col) for col, name in headers.items()}
            row.clear()


def split_for(key, seed=42):
    value = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if value < .8 else 'validation' if value < .9 else 'test'


def prepare(raw, output, seed=42):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, rejected, files, audit = [], [], {}, {}
    cache = {}
    for name, spec in TASKS.items():
        path = Path(raw) / spec['file']
        files[spec['file']] = hashlib.sha256(path.read_bytes()).hexdigest()
        count, blanks = 0, 0
        for rownum, row in workbook_rows(path, spec['sheet']):
            # Blank/formatting-only spreadsheet rows are not measurements.
            if not row.get('IL SMILES') and row.get(spec['target']) is None:
                blanks += 1; continue
            count += 1
            try:
                raw_smiles = (row.get('IL SMILES') or '').strip()
                if raw_smiles not in cache:
                    try:
                        canonical = canonicalize(raw_smiles)
                        graph(canonical)
                        cache[raw_smiles] = canonical
                    except ValueError as exc:
                        cache[raw_smiles] = exc
                smi = cache[raw_smiles]
                if isinstance(smi, Exception): raise ValueError(str(smi))
                value = float(row[spec['target']])
                if not math.isfinite(value): raise ValueError('Non-finite target')
                if spec['transform'] == 'log' and value <= 0: raise ValueError('Non-positive log target')
                t = float(row[spec['temperature']]) if spec['temperature'] else None
                p = float(row[spec['pressure']]) * spec['pressure_factor'] if spec['pressure'] else None
                if t is not None and (not math.isfinite(t) or t <= 0): raise ValueError('Invalid temperature (must be positive Kelvin)')
                if name == 'viscosity' and t < 150: raise ValueError('Viscosity below VFT domain (150 K)')
                if p is not None and (not math.isfinite(p) or p < 0): raise ValueError('Invalid pressure')
                if name == 'co2_solubility' and not 0 <= value <= 1: raise ValueError('Mole fraction outside [0,1]')
                rows.append(dict(smiles=smi, task=name, value=value, temperature_K=t, pressure_input=p,
                                 source_file=spec['file'], source_sheet=spec['sheet'], source_row=rownum,
                                 source_dataset_id=row.get('Dataset ID'),
                                 source_accepted=row.get('Accepted dataset'), source_excluded=row.get('Excluded IL'),
                                 split=split_for(smi, seed)))
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append(dict(task=name, source_row=rownum, smiles=row.get('IL SMILES'), reason=str(exc)))
        audit[name] = dict(source_measurements=count, ignored_empty_rows=blanks)
        print(f'{name}: {count} measurement rows', flush=True)
    df = pd.DataFrame(rows)
    # Keep all replicates in the archive, remove exact duplicate observations for fitting.
    df.to_csv(output / 'observations_all.csv.gz', index=False)
    keys = ['smiles', 'task', 'temperature_K', 'pressure_input', 'value']
    dup = df.duplicated(keys)
    df[dup].to_csv(output / 'duplicates.csv', index=False)
    df = df[~dup].copy()
    df.to_csv(output / 'observations.csv.gz', index=False)
    pd.DataFrame(rejected, columns=['task', 'source_row', 'smiles', 'reason']).to_csv(output / 'rejected.csv', index=False)
    for name, sub in df.groupby('task'):
        audit[name].update(kept=len(sub), unique_ils=sub.smiles.nunique(), splits=sub.split.value_counts().to_dict(),
                          temperature_range_K=range_or_none(sub.temperature_K), pressure_input_range=range_or_none(sub.pressure_input))
    report = dict(seed=seed, raw_sha256=files, tasks=TASKS, audit=audit, rejected=len(rejected),
                  exact_duplicates_removed=int(dup.sum()), total_observations=len(df), unique_ils=df.smiles.nunique(),
                  upstream_commit='ff5e55cfb8162b0706fc2d88ca5cd384705686b1')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    return report


def range_or_none(series):
    return None if series.dropna().empty else [float(series.min()), float(series.max())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw/iltransr')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.seed)
