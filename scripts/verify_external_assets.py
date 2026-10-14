"""Check the exact provider weights and LimiX source used by the paper."""
import json
import subprocess
from scripts.verify_release import ROOT,digest


def main():
    config=json.loads((ROOT/'configs/external_dependencies.json').read_text())
    for name,path in config['checkpoint_destinations'].items():
        if digest(ROOT/path) not in config['checkpoint_sha256'][name]:raise ValueError('Provider checkpoint differs from the paper: '+name)
        print(name,'checkpoint PASS')
    head=subprocess.check_output(['git','-C',str(ROOT/'external/LimiX'),'rev-parse','HEAD'],text=True).strip()
    if head!=config['limix_source']['commit']:raise ValueError('LimiX source commit differs from the paper')
    print('LimiX source commit PASS')


if __name__=='__main__':main()
