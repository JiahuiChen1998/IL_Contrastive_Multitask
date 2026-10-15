import numpy as np
import pandas as pd
import pytest
import torch
from il_contrastive.graph import canonicalize, graph, pack
from il_contrastive.model import ILMixer
from il_contrastive.tasks import TASKS
from il_contrastive.data import split_for
from il_contrastive.train import fit_scalers, loss_for

S = 'CCCC[n+]1ccn(C)c1.[Cl-]'

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    torch.manual_seed(10)


def test_fragments_and_charge_validation():
    g = graph(S)
    assert g['fragment'].unique().numel() == 2
    src, dst = g['edge_index']
    assert torch.equal(g['fragment'][src], g['fragment'][dst])
    for bad in ['bad_smiles', 'CCO', 'C[N+](C)(C)C', 'C[N+](C)(C)C.[O-]S(=O)(=O)[O-]']:
        with pytest.raises(ValueError): graph(bad)
    assert canonicalize(S) == canonicalize('.'.join(reversed(S.split('.'))))
    assert len(graph('C[N+](C)(C)C.C[N+](C)(C)C.[O-]S(=O)(=O)[O-]')['fragment'].unique()) == 3


@pytest.mark.parametrize('cross', [False, True])
def test_permutation_and_batch_independence(cross):
    tasks = {'density': TASKS['density']}
    m = ILMixer(tasks, hidden=16, layers=2, cross_ion=cross).eval()
    g = pack([S]); n = len(g['x']); perm = torch.randperm(n); inv = torch.argsort(perm)
    shuffled = dict(g, x=g['x'][perm], fragment=g['fragment'][perm], batch=g['batch'][perm],
                    edge_index=inv[g['edge_index']])
    t, p = torch.tensor([300.]), torch.tensor([.1])
    a = m(g, t, p)['density']; b = m(shuffled, t, p)['density']
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    together = m(pack([S, '[Na+].[Cl-]']), torch.tensor([300., 320.]), torch.tensor([.1, 1.]))['density']
    torch.testing.assert_close(a, together[:1], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(together).all()


def test_physics_and_pressure_gradients():
    tasks = {k: TASKS[k] for k in ['viscosity', 'surface_tension', 'density', 'melting_point']}
    m = ILMixer(tasks, hidden=16, layers=1)
    t = torch.tensor([280., 330.], requires_grad=True)
    p = torch.tensor([.1, .1], requires_grad=True)
    out = m(pack([S, S]), t, p)
    assert out['viscosity'][0] > out['viscosity'][1]
    assert out['surface_tension'][0] > out['surface_tension'][1]
    torch.testing.assert_close(out['melting_point'][0], out['melting_point'][1])
    grad = torch.autograd.grad(out['density'].sum(), p, retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    sum(x.sum() for x in out.values()).backward()
    assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
    with pytest.raises(ValueError): m(pack([S]), torch.tensor([100.]), torch.tensor([0.]))


def test_train_scaling_and_partial_labels():
    tasks = {k: TASKS[k] for k in ['density', 'viscosity']}
    df = pd.DataFrame({'task':['density', 'density', 'viscosity'], 'value':[10., 20., np.e]})
    scalers = fit_scalers(df, tasks)
    assert scalers['density']['mean'] == 15.
    predictions = {'density':torch.tensor([-1., 1., 100.], requires_grad=True),
                   'viscosity':torch.tensor([100., 100., 0.], requires_grad=True)}
    loss = loss_for(predictions, df, scalers, tasks, 'cpu')
    assert loss.item() == pytest.approx(0)
    loss.backward()
    assert predictions['density'].grad[-1] == 0
    assert (predictions['viscosity'].grad[:2] == 0).all()


def test_global_split():
    assert split_for(canonicalize(S)) == split_for(canonicalize('.'.join(reversed(S.split('.')))))
    assert len({split_for(str(i)) for i in range(100)}) == 3


def test_mixed_task_conditions_do_not_contaminate_vft():
    tasks = {k: TASKS[k] for k in ['viscosity', 'heat_capacity']}
    model = ILMixer(tasks, hidden=16, layers=1)
    outputs = model(pack([S, S]), torch.tensor([300., 10.]), torch.tensor([0., 101.325]),
                    ['viscosity', 'heat_capacity'])
    assert all(torch.isfinite(v).all() for v in outputs.values())


def test_all_edgeless_graph_backward():
    model = ILMixer({'density':TASKS['density']}, hidden=16, layers=1, cross_ion=True)
    output = model(pack(['[Na+].[Cl-]']), torch.tensor([300.]), torch.tensor([.1]))['density']
    output.sum().backward()
    assert torch.isfinite(output).all()
