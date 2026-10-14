"""Check row identities, targets and conditions against the frozen benchmark."""
import hashlib
import numpy as np
import pandas as pd

KEYS=['smiles','task','source_file','source_sheet','source_row']


def numeric_values(values, label):
    """Accept numeric representations; reject malformed values without coercion."""
    try:
        return pd.to_numeric(values, errors='raise').to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Non-numeric values in {label}') from error


def audit_predictions(predicted, expected):
    if predicted.duplicated(KEYS).any() or expected.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction/frozen row identities')
    joined=predicted.merge(expected,on=KEYS,how='outer',suffixes=('_pred','_frozen'),indicator=True,validate='one_to_one')
    if not (joined['_merge']=='both').all():
        raise ValueError('Prediction rows do not equal the frozen held-out rows')
    if not (joined.split_pred==joined.split_frozen).all():
        raise ValueError('Prediction split labels changed')
    for column in ['value','temperature_K','pressure_input']:
        actual=numeric_values(joined[column+'_pred'],column+' prediction file')
        reference=numeric_values(joined[column+'_frozen'],column+' frozen data')
        if not np.allclose(actual,reference,rtol=1e-10,atol=1e-12,equal_nan=True):
            raise ValueError(f'Frozen target/condition values changed: {column}')
    for column in ['prediction','prediction_transformed']:
        if not np.isfinite(numeric_values(predicted[column],column)).all():
            raise ValueError('Nonfinite prediction')
    identity=expected[KEYS].sort_values(KEYS).to_csv(index=False)
    return dict(rows=len(expected),unique_ils=expected.smiles.nunique(),
                frozen_row_identity_sha256=hashlib.sha256(identity.encode()).hexdigest(),status='pass')
