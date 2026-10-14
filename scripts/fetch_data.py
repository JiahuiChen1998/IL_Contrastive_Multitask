"""Download the eleven workbooks at a pinned upstream commit; retain its license."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from il_mixer.tasks import TASKS

COMMIT = 'ff5e55cfb8162b0706fc2d88ca5cd384705686b1'
BASE = f'https://raw.githubusercontent.com/GuzhongChen/ILTransR/{COMMIT}/'

if __name__ == '__main__':
    output = Path('data/raw/iltransr'); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path('data/raw/source_manifest.json').read_text())
    files = {s['file']: 'datasets/il properties/' + s['file'] for s in TASKS.values()}
    files['LICENSE'] = 'LICENSE'
    for name, remote in files.items():
        path = output / name
        expected = manifest['sha256'][name]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f'Verified {name}'); continue
        with urlopen(BASE + quote(remote), timeout=180) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch for {name}')
        tmp = path.with_suffix(path.suffix + '.part'); tmp.write_bytes(content); tmp.replace(path)
        print(f'Downloaded and verified {name}')
