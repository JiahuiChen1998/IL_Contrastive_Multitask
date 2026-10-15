"""Fixed ionic descriptors and train-only preprocessing for tabular benchmarks."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

DESCRIPTORS = ['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds',
 'RingCount','NumAromaticRings','NumAliphaticRings','FractionCSP3','HeavyAtomCount',
 'NHOHCount','NOCount','NumHeteroatoms','LabuteASA','BalabanJ','BertzCT','HallKierAlpha',
 'Kappa1','Kappa2','Kappa3','Chi0v','Chi1v','MaxPartialCharge']
FP_BITS=128

def structure_features(smiles, fingerprints=False):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f'Invalid SMILES: {smiles}')
    frags=Chem.GetMolFrags(mol,asMols=True)
    cats=[f for f in frags if Chem.GetFormalCharge(f)>0]
    ans=[f for f in frags if Chem.GetFormalCharge(f)<0]
    def combine(fs):
        return Chem.MolFromSmiles('.'.join(sorted(Chem.MolToSmiles(f) for f in fs))) if fs else None
    parts=[combine(cats),combine(ans),mol];vals=[];names=[]
    for role,part in zip(['cation','anion','whole'],parts):
        for d in DESCRIPTORS:
            try: value=float(getattr(Descriptors,d)(part)) if part is not None else 0.
            except (ValueError,RuntimeError,ZeroDivisionError): value=np.nan
            vals.append(value if np.isfinite(value) else np.nan);names.append(f'{role}_{d}')
    vals.extend([len(cats),len(ans),sum(Chem.GetFormalCharge(f) for f in cats),sum(Chem.GetFormalCharge(f) for f in ans)])
    names.extend(['n_cations','n_anions','cation_charge','anion_charge'])
    if fingerprints:
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=FP_BITS,includeChirality=True)
        for role,part in zip(['cation','anion'],parts[:2]):
            bits=gen.GetFingerprintAsNumPy(part) if part is not None else np.zeros(FP_BITS)
            vals.extend(bits);names.extend(f'{role}_morgan_{i}' for i in range(FP_BITS))
    return np.asarray(vals,dtype=np.float64),names

def feature_matrix(frame,spec,representation):
    fp=representation=='descriptors_fp';cache={s:structure_features(s,fp) for s in sorted(frame.smiles.unique())}
    names=next(iter(cache.values()))[1].copy();x=np.stack([cache[s][0] for s in frame.smiles])
    for col,required in [('temperature_K',spec['temperature']),('pressure_input',spec['pressure'])]:
        if required:
            values=frame[col].to_numpy(float)
            if not np.isfinite(values).all():raise ValueError(f'Missing required condition: {col}')
            x=np.column_stack([x,values]);names.append(col)
    return x,names

class TrainPreprocessor:
    def fit(self,x):
        x=np.where(np.isfinite(x),x,np.nan)
        self.median=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in x.T])
        filled=np.where(np.isfinite(x),x,self.median)
        self.keep=filled.std(axis=0)>1e-12
        if not self.keep.any():raise ValueError('No variable training features')
        self.mean=filled[:,self.keep].mean(axis=0);self.std=filled[:,self.keep].std(axis=0)
        return self
    def transform(self,x):
        x=np.where(np.isfinite(x),x,self.median)[:,self.keep]
        return ((x-self.mean)/self.std).astype(np.float32)
    def state(self):return {k:getattr(self,k).tolist() for k in ['median','keep','mean','std']}

@contextmanager
def train_only_limix_preprocessing(model,n_train):
    """Adapt upstream's numeric preprocessing to fit only the context rows.

    No retrieval or feature-generation steps: audited no-retrieval config only.
    Upstream already passes eval_pos to the transformer. No source files mutated.
    """
    restored=[]
    def bind(step):
        original_fit_transform=step.fit_transform
        def fit_transform(self,x,categorical_features,seed,**kwargs):
            if type(self).__name__=='CategoricalFeatureEncoder':
                _, cats=original_fit_transform(x[:n_train],categorical_features,seed,**kwargs)
                if self.transformer is None:return x,cats
                z=self.transformer.transform(x)
                if self.encoding_strategy.startswith('ordinal') and self.category_mappings:
                    for col,perm in self.category_mappings.items():
                        valid=np.isfinite(z[:,col]);z[valid,col]=perm[z[valid,col].astype(int)]
                return z,cats
            fit_kwargs=kwargs.copy()
            # Upstream checks all-NaN query columns; a train-only fit has no query rows.
            if type(self).__name__=='FilterValidFeatures':fit_kwargs.pop('y',None)
            self.fit(x[:n_train],categorical_features,seed,**fit_kwargs)
            return self.transform(x,**kwargs)
        return MethodType(fit_transform,step)
    original=model.get_categorical_features_indices
    model.get_categorical_features_indices=lambda x:original(x[:n_train])
    try:
        for pipe in model.preprocess_pipelines:
            for step in pipe:
                if type(step).__name__ not in {'FilterValidFeatures','RebalanceFeatureDistribution','CategoricalFeatureEncoder','FeatureShuffler'}:
                    raise ValueError(f'Unaudited LimiX preprocessing step {type(step).__name__}')
                restored.append((step,step.fit_transform));step.fit_transform=bind(step)
        yield
    finally:
        model.get_categorical_features_indices=original
        for step,method in restored:step.fit_transform=method
