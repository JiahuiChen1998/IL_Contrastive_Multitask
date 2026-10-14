import numpy as np
import pandas as pd
from il_mixer.tabular import TrainPreprocessor,structure_features,feature_matrix,train_only_limix_preprocessing
from il_mixer.equations import study_tasks
from scripts.run_tabular_benchmark import scores

def test_fragment_order_invariance():
    a,n=structure_features('CC[N+](C)(C)C.[Cl-]',True)
    b,m=structure_features('[Cl-].CC[N+](C)(C)C',True)
    assert n==m and np.allclose(a,b,equal_nan=True)

def test_train_only_imputation_and_variance_selection():
    train=np.array([[1.,np.nan,4.],[3.,2.,4.],[5.,4.,4.]])
    prep=TrainPreprocessor().fit(train);before=prep.state()
    result=prep.transform(np.array([[np.nan,1000.,999.]]))
    assert prep.state()==before and result.shape==(1,2)
    assert result[0,0]==0 and result[0,1]>100

def test_structure_only_does_not_receive_conditions():
    f=pd.DataFrame({'smiles':['C[N+](C)(C)C.[Cl-]']*2,'temperature_K':[200,400],'pressure_input':[0,100]})
    x,n=feature_matrix(f,study_tasks()['toxicity'],'descriptors')
    assert np.array_equal(x[0],x[1]) and 'temperature_K' not in n and 'pressure_input' not in n
    y,n=feature_matrix(f,study_tasks()['heat_capacity'],'descriptors')
    assert n[-2:]==['temperature_K','pressure_input'] and not np.array_equal(y[0],y[1])

def test_limix_adapter_uses_only_training_rows_and_restores():
    class FilterValidFeatures:
        def fit(self,x,categorical_features,seed,**kw):self.mean=x.mean(axis=0)
        def transform(self,x,**kw):return x-self.mean,[]
        def fit_transform(self,x,categorical_features,seed,**kw):self.fit(x,categorical_features,seed);return self.transform(x)
    class Model:
        def __init__(self):self.preprocess_pipelines=[[FilterValidFeatures()]]
        def get_categorical_features_indices(self,x):return [len(x)]
    model=Model();step=model.preprocess_pipelines[0][0];original=step.fit_transform
    with train_only_limix_preprocessing(model,2):
        a,_=step.fit_transform(np.array([[0.],[2.],[1000.]]),[],42,y=np.array([1.,2.]))
        assert a[0]==-1 and a[1]==1 and model.get_categorical_features_indices(np.zeros((3,1)))==[2]
    assert step.fit_transform==original and model.get_categorical_features_indices(np.zeros((3,1)))==[3]

def test_co2_metrics_and_equal_il_weighting():
    f=pd.DataFrame({'task':['co2_solubility']*3,'smiles':['a','a','b'],'value':[.1,.1,.4]})
    r,p=scores(f,np.log([.2,.2,.4]),study_tasks()['co2_solubility'],1.)
    assert np.isclose(r['mae'],.2/3) and np.isclose(r['il_macro_mae'],.05)
    assert np.isclose(r['standardized_mse'],2*np.log(2)**2/3)

def test_actual_limix_preprocessing_ignores_other_query_values():
    import sys
    from pathlib import Path
    import pytest
    repo=Path(__file__).resolve().parents[1]/'external/LimiX'
    if not repo.exists():pytest.skip('Optional upstream LimiX checkout unavailable')
    pytest.importorskip('kditransform')
    sys.path.insert(0,str(repo))
    from inference.preprocess import FilterValidFeatures,CategoricalFeatureEncoder,FeatureShuffler
    class Model:
        def __init__(self):self.preprocess_pipelines=[[FilterValidFeatures(),CategoricalFeatureEncoder('ordinal_strict_feature_shuffled'),FeatureShuffler(mode='shuffle')]]
        def get_categorical_features_indices(self,x):return [0]
    model=Model();train=np.column_stack([np.tile([0.,1.],20),np.arange(40.)]);query=np.array([[0.,10.],[99.,1e8]])
    outputs=[]
    for q in [query[:1],query]:
        z=np.vstack([train,q]);cats=[0]
        with train_only_limix_preprocessing(model,len(train)):
            for step in model.preprocess_pipelines[0]:z,cats=step.fit_transform(z,cats,42,y=np.arange(40.))
        outputs.append(z)
    assert np.allclose(outputs[0],outputs[1][:-1],equal_nan=True)
