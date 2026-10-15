import numpy as np
import pandas as pd
from il_contrastive.conditions import condition_basis, EmpiricalTrend, condition_domains
from il_contrastive.equations import study_tasks


def test_pressure_trend_extrapolates_and_does_not_refit_on_query():
    f=pd.DataFrame({'temperature_K':[280.,300.,320.,340.], 'pressure_input':[0.,10.,0.,10.]})
    spec=study_tasks()['density'];b,n=condition_basis(f,spec,'density')
    model=EmpiricalTrend().fit(b,1000-0.5*f.temperature_K+0.1*f.pressure_input)
    state=model.state();q=pd.DataFrame({'temperature_K':[360.], 'pressure_input':[100.]})
    qb,_=condition_basis(q,spec,'density');pred=model.predict(qb)
    assert np.isfinite(pred).all() and model.state()==state
    assert all(flag[0] for flag in condition_domains(f,q,spec).values())


def test_co2_basis_keeps_both_conditions_and_cp_pressure():
    f=pd.DataFrame({'temperature_K':[300.,300.,330.], 'pressure_input':[1.,2.,2.]})
    b,n=condition_basis(f,study_tasks()['co2_solubility'],'co2_solubility')
    assert b.shape==(3,4) and not np.array_equal(b[0],b[1])
    cp,n=condition_basis(f,study_tasks()['heat_capacity'],'heat_capacity')
    assert 'pressure_native' in n and cp.shape==(3,4)
