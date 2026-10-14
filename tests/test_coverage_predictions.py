import pandas as pd
import pytest
from scripts.audit_coverage_predictions import audit_predictions


def frames():
    truth=pd.DataFrame(dict(smiles=['C','CC'],task=['melting_point']*2,source_file=['x']*2,
        source_sheet=['Sheet1']*2,source_row=[1,2],split=['validation']*2,value=[300.,400.],
        temperature_K=[float('nan')]*2,pressure_input=[float('nan')]*2))
    return truth,truth.assign(prediction=[301.,399.],prediction_transformed=[301.,399.])


def test_audit_allows_order_change_but_not_observation_change():
    truth,pred=frames()
    assert audit_predictions(pred.iloc[::-1],truth)['rows']==2
    pred.loc[0,'value']=400.
    with pytest.raises(ValueError,match='values changed'):
        audit_predictions(pred,truth)


def test_audit_rejects_missing_or_reassigned_rows():
    truth,pred=frames()
    with pytest.raises(ValueError,match='held-out rows'):
        audit_predictions(pred.iloc[:1],truth)
    pred.loc[0,'split']='train'
    with pytest.raises(ValueError,match='split labels'):
        audit_predictions(pred,truth)


def test_audit_numeric_strings_are_compared_without_weakening_checks():
    truth,pred=frames()
    pred['value']=pred.value.astype(str)
    pred['prediction']=pred.prediction.astype(str)
    assert audit_predictions(pred,truth)['status']=='pass'
    pred.loc[0,'value']='301.0'
    with pytest.raises(ValueError,match='values changed'):
        audit_predictions(pred,truth)
    pred.loc[0,'value']='malformed'
    with pytest.raises(ValueError,match='Non-numeric'):
        audit_predictions(pred,truth)
