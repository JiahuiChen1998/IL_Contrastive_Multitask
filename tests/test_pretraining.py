import torch
import pytest
from il_mixer.graph import graph, pack
from il_mixer.model import ILMixer
from il_mixer.pretraining import masked_view, contrastive_loss, load_encoder


def test_general_graph_keeps_strict_downstream_and_handles_no_bonds():
    with pytest.raises(ValueError):graph('CCO')
    g=pack(['[Na+]','CCO'],require_ions=False)
    model=ILMixer({},hidden=16,layers=1)
    out=model.encode(g)
    assert out.shape==(2,32) and torch.isfinite(out).all()
    out.square().sum().backward()


def test_masks_preserve_ionic_graph_and_cached_input():
    g=pack(['CC[N+](C)(C)C.[Cl-]','C[N+](C)(C)C.[Br-]'])
    original=g['x'].clone();v=masked_view(g,.3)
    assert torch.equal(g['x'],original)
    assert torch.equal(v['x'][:,1:],original[:,1:])
    assert torch.equal(v['edge_index'],g['edge_index'])
    assert torch.equal(v['fragment'],g['fragment'])
    assert (v['x'][:,0]==0).any()


def test_contrastive_gradients_and_encoder_transfer(tmp_path):
    model=ILMixer({},hidden=16,layers=1)
    a=torch.randn(4,8,requires_grad=True);b=a.detach().clone().requires_grad_()
    loss=contrastive_loss(a,b);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    path=tmp_path/'ssl.pt'
    torch.save(dict(format='ilmixer_ssl_v1',encoder=model.state_dict(),manifest={}),path)
    other=ILMixer({'x':{'temperature':None,'pressure':None,'head':'mlp'}},hidden=16,layers=1,cross_ion=True)
    readout={k:v.clone() for k,v in other.readouts.state_dict().items()}
    load_encoder(other,path)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,other.readouts.state_dict()[k]) for k,v in readout.items())
    with pytest.raises(ValueError):load_encoder(ILMixer({},hidden=32,layers=1),path)
    with pytest.raises(ValueError):load_encoder(other,path,include_cross=True)


def test_corpus_preparation_excludes_downstream_and_deduplicates(tmp_path, monkeypatch):
    import json
    import sqlite3
    import pandas as pd
    from scripts.prepare_pretraining import main
    train=tmp_path/'train.txt';val=tmp_path/'val.txt';down=tmp_path/'down.csv';db=tmp_path/'corpus.sqlite'
    train.write_text('CCO\nOCC\nCC[N+](C)(C)C.[Cl-]\nCCC\n')
    val.write_text('OCC\nC[N+](C)(C)C.[Br-]\n')
    pd.DataFrame({'smiles':['[Cl-].CC[N+](C)(C)C']}).to_csv(down,index=False)
    monkeypatch.setattr('sys.argv',['prepare','--train',str(train),'--validation',str(val),'--downstream',str(down),'--output',str(db)])
    main()
    with sqlite3.connect(db) as conn:rows=conn.execute('SELECT smiles,split FROM molecules').fetchall()
    assert ('CCO','validation') in rows and ('CCO','train') not in rows
    assert len(rows)==3 and all('Cl' not in s for s,_ in rows)
    report=json.loads(db.with_suffix('.json').read_text())
    assert report['sources']['train']['counts']['downstream_excluded']==1


def test_sqlite_batches_are_unique_and_respect_hash_subset(tmp_path):
    import sqlite3
    from il_mixer.pretraining import smiles_batches
    db=tmp_path/'corpus.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE molecules (smiles TEXT,split TEXT,pair INTEGER,shuffle_key TEXT)')
        conn.executemany('INSERT INTO molecules VALUES (?,?,?,?)',[(str(i),'train',i%2,f'{i:04d}') for i in range(20)])
    rows=[s for batch in smiles_batches(db,'train',2,17,limit=10,buffer_size=3) for s in batch]
    assert len(rows)==len(set(rows))==10 and set(rows)=={str(i) for i in range(10)}
