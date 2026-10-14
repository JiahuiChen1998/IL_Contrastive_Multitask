"""Extend raw tabular baselines to seeds 42–46, with parallel seed workers.

Default is preparation/reporting only. --run explicitly launches the new fits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from il_mixer.pretraining import digest
from scripts.tabular_benchmark_archive import ROOTS, freeze, index_results

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
OUT = Path('runs/tabular_five_seeds_v1')
REPORT = Path('reports/tabular_five_seeds_v1')


def job_path(root, job):
    return root / job['task'] / job['model'] / job['representation'] / 'raw' / f"seed_{job['seed']}"


def run_one(root, args, index, job, timeout):
    path = job_path(root, job)
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'metrics.json').exists():
        return None
    print(f"START {job['task']} / {job['model']} / {job['representation']} / seed {job['seed']}", flush=True)
    with (path / 'run.log').open('w') as log:
        try:
            result = subprocess.run([sys.executable, '-m', 'scripts.run_tabular_all', *args,
                '--job-index', str(index)], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            status = 'completed' if result.returncode == 0 and (path / 'metrics.json').exists() else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
    print(f"{status.upper()} {job['task']} / {job['model']} / {job['representation']} / seed {job['seed']}", flush=True)
    return None if status == 'completed' else dict(**job, status=status, log=str(path / 'run.log'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--workers', type=int, default=5, help='Maximum concurrent seeds in one task/model/representation')
    p.add_argument('--job-timeout', type=int, default=7200)
    a = p.parse_args()
    if a.workers < 1 or a.workers > 5 or a.job_timeout <= 0:
        p.error('workers must be 1–5 and timeout must be positive')
    import os
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Five-seed launcher is already active')
    # Keep the previous completion wrapper from writing the reused roots concurrently.
    legacy_lock = Path(ROOTS[-1], 'queue.lock').open('a')
    try:
        fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('The three-seed completion queue is still running; wait for it to finish')
    snapshot = Path('data/benchmarks/il_v1')
    frozen = freeze(Path('data/processed/observations.csv.gz'), snapshot)
    # Preserve the old locked plans and scientific code; add two disjoint seed roots.
    roots = list(ROOTS); plans = []
    for seed in [45, 46]:
        root = OUT / f'seed_{seed}'
        args = ['--models', 'extratrees,tabpfn,limix', '--modes', 'raw', '--seeds', str(seed),
            '--device', 'cuda', '--threads', '4', '--output', str(root)]
        subprocess.run([sys.executable, '-m', 'scripts.run_tabular_all', *args], check=True)
        plan = json.loads((root / 'plan.json').read_text())
        plans.append((root, args, plan)); roots.append(str(root))
    protocol = dict(schema='tabular_five_seeds_v1', seeds=SEEDS, data_sha256=frozen['data_sha256'],
        expected_fits=330, roots=roots, modes=['raw'],
        legacy_manifest_sha256=digest(snapshot / 'manifest.json'),
        policy='Reuse seeds 42–44 including explicit failed attempts; extend with seeds 45–46. Fixed split.',
        plan_sha256={str(root): digest(root / 'plan.json') for root, _, _ in plans})
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Five-seed protocol changed; use a new version')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    print('330 total fits; preserve 198 prior attempts and add 132 fits. No fits run without --run.', flush=True)
    if a.run:
        failures = {str(root): [] for root, _, _ in plans}
        # Each seed writes its own plan root. Only one job per seed/root is active,
        # avoiding concurrent writes to run_tabular_all's manifest temporary file.
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i in range(len(plans[0][2]['jobs'])):
                group = [(root, args, plan['jobs'][i]) for root, args, plan in plans]
                if len({(j['task'], j['model'], j['representation']) for _, _, j in group}) != 1:
                    raise ValueError('Seed job ordering differs')
                futures = [(root, pool.submit(run_one, root, args, i, j, a.job_timeout)) for root, args, j in group]
                for root, future in futures:
                    failure = future.result()
                    if failure:
                        failures[str(root)].append(failure)
                for root, _, _ in plans:
                    (root / 'failures.json').write_text(json.dumps(failures[str(root)], indent=2))
    index_results(roots, snapshot, REPORT, frozen, SEEDS)
    subprocess.run([sys.executable, '-m', 'scripts.summarize_tabular_all', '--roots', *roots,
        '--seeds', ','.join(map(str, SEEDS)), '--output', str(REPORT)], check=True)
    print(f'Combined report: {REPORT}/COMPARISON.md; full coverage: {REPORT}/coverage.csv', flush=True)
    if a.run:
        import pandas as pd
        incomplete = (pd.read_csv(REPORT / 'coverage.csv').status != 'completed').sum()
        print(f'Queue ended; {incomplete} of 330 fits lack successful results. See coverage and per-fit logs.', flush=True)
        if incomplete:
            sys.exit(1)


if __name__ == '__main__':
    main()
