"""Run the corrected exploratory pipeline on synthetic features to catch selector/coefficient errors."""
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'
import tempfile, nbformat, numpy as np,pandas as pd,json
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GroupKFold,GroupShuffleSplit,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
rng=np.random.default_rng(142)
N=90
pred=pd.DataFrame({'cbsa_code':[str(11000+i) for i in range(N)],
                   'geographic_group':np.repeat([f'g{x}' for x in range(9)],10)})
core=['qcew2018to2019q2_growth','log_qcew2019q2_employment',
      'acs2018_bachelors_plus_share','acs2018_poverty_share',
      'acs2018_unemployment_share','acs2018_broadband_subscription_share']
industry=[f'qcew2019q2_share_naics_{i:02}' for i in range(19)]+[
         'qcew2019q2_private_share_all_covered','qcew2019q2_private_goods_producing_share']
for c in core+industry:
    v=rng.normal(size=N)
    v[rng.uniform(size=N)<.1]=np.nan
    pred[c]=v
pred['qcew2019q2_share_naics_18']=np.nan  # all missing, should be removed in each fold
pred['qcew2019q2_share_naics_00']=1.0   # constant, should be removed in each fold
merged=pred.copy()
for target in ('resistance_shortfall_pct_aprjun2020','persistence_shortfall_pct_octdec2023','cumulative_shortfall_46_pp_months'):
    merged[target]=rng.uniform(0,30,N)
merged['recovery_gain_from_trough_pp']=rng.normal(size=N)*15
with tempfile.TemporaryDirectory() as directory:
    OUT=Path(directory)
    context=globals()
    n=nbformat.read(NOTEBOOKS_DIR/'03_Stage2_Resilience_and_Initial_Prediction.ipynb',as_version=4)
    exec(n.cells[7].source,context)
    scores=pd.read_csv(OUT/'stage2_omb2018_predictive_model_scores.csv')
    coef=pd.read_csv(OUT/'stage2_omb2018_standardized_coefficients.csv')
    assert len(scores)==12
    assert scores.n_features.max()<=len(core+industry)-2
    assert scores.nonzero_coefficients.ge(0).all()
    assert coef.feature.str.contains('naics_18|naics_00').sum()==0
    print('PASS fold-local selection, imputation, coefficient names, baseline metrics (12 fits)')
