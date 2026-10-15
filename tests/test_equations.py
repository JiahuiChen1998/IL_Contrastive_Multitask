import math
import numpy as np
import pandas as pd
import pytest
import torch
from il_contrastive.equations import EquationHead, choose_cp_degree, study_tasks, CONDITION_TASKS
from il_contrastive.graph import pack
from il_contrastive.model import ILMixer
from scripts.run_ablation import variants


@pytest.mark.parametrize('name', CONDITION_TASKS)
def test_requested_equations_and_physical_coefficients(name):
    torch.manual_seed(2)
    head=EquationHead(name,8,16,mean=2.,std=3.,cp_degree=3)
    h=torch.randn(3,8)
    t=torch.tensor([270.,300.,350.],requires_grad=True);p=torch.tensor([.1,2.,5.])
    z=head(h,t,p); c=head.physical_coefficients(h,p)
    if name=='viscosity':expected=c['a']+c['B']*(1/(t-c['T0'])-1/(298.15-c['T0']))
    elif name=='surface_tension':expected=c['a']*t+c['b']
    elif name=='density':expected=c['A']+c['B']*t+c['C']*p
    elif name=='heat_capacity':expected=sum(c[f'A{i}']*(t/100)**i for i in range(4))
    elif name=='co2_solubility':expected=-c['delta_h_J_per_mol']/8.314/t+c['C']
    elif name=='refractive_index':
        expected=c['a']+c['b']*t;assert (c['b']<0).all()
    else:expected=c['a1']+c['a2']*t
    torch.testing.assert_close(2.+3.*z,expected,atol=2e-4,rtol=2e-4)
    z.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in head.parameters() if x.grad is not None)


def test_linear_slopes_are_free_except_refractive_index():
    for name in ['surface_tension','thermal_conductivity']:
        head=EquationHead(name,8,16)
        with torch.no_grad():
            head.net[-1].weight.zero_();head.net[-1].bias.copy_(torch.tensor([0.,2.]))
        y=head(torch.zeros(2,8),torch.tensor([280.,320.]),torch.zeros(2))
        assert y[1]>y[0]


def test_co2_fixed_pressure_inverse_temperature():
    head=EquationHead('co2_solubility',8,16)
    h=torch.ones(3,8);u=torch.tensor([1/280.,1/300.,1/340.])
    p=torch.ones(3,requires_grad=True)
    y=head(h,1/u,p)
    torch.testing.assert_close((y[1]-y[0])/(u[1]-u[0]),(y[2]-y[1])/(u[2]-u[1]),rtol=1e-4,atol=1e-4)
    assert torch.autograd.grad(y.sum(),p)[0].abs().sum()>0


def test_cp_selection_and_common_target_space():
    rows=[dict(task='heat_capacity',smiles=str(i),temperature_K=200+j) for i in range(30) for j in range(8)]
    assert choose_cp_degree(pd.DataFrame(rows))[0]==3
    assert choose_cp_degree(pd.DataFrame(rows[:8]))[0]==1
    assert study_tasks()['heat_capacity']['transform']=='identity'
    assert study_tasks()['co2_solubility']['transform']=='log'
    assert sum(len(variants(n)) for n in study_tasks())==36


def test_model_equation_roundtrip(tmp_path):
    torch.set_num_threads(1)
    config=dict(tasks=study_tasks(),head_mode='equations',hidden=16,layers=1,cross_ion=True,cp_degree=3)
    m=ILMixer(**config).eval();g=pack(['CCCC[n+]1ccn(C)c1.[Cl-]']*2)
    args=(g,torch.tensor([5.,300.]),torch.tensor([101.325,1.]),['heat_capacity','co2_solubility'])
    a=m(*args)
    torch.save(dict(config=config,state=m.state_dict()),tmp_path/'model.pt')
    saved=torch.load(tmp_path/'model.pt',weights_only=True)
    restored=ILMixer(**saved['config']);restored.load_state_dict(saved['state']);restored.eval()
    for n,b in restored(*args).items():torch.testing.assert_close(a[n],b)
