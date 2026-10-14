import json
import sqlite3
from pathlib import Path

import pytest
import torch
from il_mixer.full_coverage import FullCoverageSampler
from scripts.pretrain_full_coverage import parser, run


def corpus(path, n=33, validation=8):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE molecules (smiles TEXT PRIMARY KEY, split TEXT, pair INTEGER, shuffle_key TEXT)')
        db.execute('CREATE INDEX sample_order ON molecules(split,shuffle_key)')
        db.executemany('INSERT INTO molecules VALUES (?,?,?,?)',
                       [('C'*(i+1), 'train', 0, f'{i:08d}') for i in range(n)]+
                       [('C'*(i+1)+'O', 'validation', 0, f'{i:08d}') for i in range(validation)])
        db.execute('CREATE TABLE metadata (json TEXT)')
        db.execute('INSERT INTO metadata VALUES (?)', (json.dumps({'counts': [dict(split='train',n=n,pair=0),
                   dict(split='validation',n=validation,pair=0)]}),))


@pytest.mark.parametrize('n,b,buffer', [(33,8,5),(17,8,100),(8,8,1),(2,128,5),(32,8,7)])
def test_exact_coverage_nestedness_and_sampler_resume(tmp_path,n,b,buffer):
    p=tmp_path/'corpus.sqlite';corpus(p)
    s=FullCoverageSampler(p,'train',n,b,42,buffer)
    first=next(s);state=s.state_dict();tail=list(s);s.close()
    resumed=FullCoverageSampler(p,'train',n,b,42,buffer,state)
    assert list(resumed)==tail
    resumed.close()
    rows=first+[v for batch in tail for v in batch]
    assert len(rows)==len(set(rows))==n
    assert set(rows)=={'C'*(i+1) for i in range(n)}
    assert all(len(batch)>=2 for batch in [first]+tail)


def test_training_resumes_exactly_with_partial_pass(tmp_path):
    p=tmp_path/'corpus.sqlite';corpus(p,17)
    def args(out, extra=()):
        return parser().parse_args(['--corpus',str(p),'--output',str(out),'--structures','17','--passes','2',
            '--device','cpu','--hidden','8','--layers','1','--batch-size','8','--buffer-size','5',
            '--validation-structures','8','--threads','1','--checkpoint-every','1','--log-every','1',*extra])
    uninterrupted=tmp_path/'full';interrupted=tmp_path/'resumed'
    assert run(args(uninterrupted))
    assert not run(args(interrupted,['--stop-after-steps','1']))
    assert run(args(interrupted,['--resume']))
    x=torch.load(uninterrupted/'final.pt',weights_only=True)
    y=torch.load(interrupted/'final.pt',weights_only=True)
    assert x['history']==y['history']
    for section in ['encoder','projector']:
        assert all(torch.equal(value,y[section][key]) for key,value in x[section].items())
    status=json.loads((interrupted/'status.json').read_text())
    assert status['unique_structures_seen']==17 and status['presentations']==34
    assert status['completed_passes']==2 and status['state']=='complete'


def test_reject_changed_resume_configuration(tmp_path):
    p=tmp_path/'corpus.sqlite';corpus(p)
    s=FullCoverageSampler(p,'train',17,8,42,5);next(s);state=s.state_dict();s.close()
    with pytest.raises(ValueError,match='configuration'):
        FullCoverageSampler(p,'train',18,8,42,5,state)
