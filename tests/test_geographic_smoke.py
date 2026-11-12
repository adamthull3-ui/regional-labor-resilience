"""Optional offline mini-integration test for 04, uses synthetic regions, no actual QCEW data."""
import nbformat, numpy as np, pandas as pd, tempfile, warnings, time, json, hashlib
from pathlib import Path

# Resolve the notebooks from tests/../code/ rather than assuming the same directory.
NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold,GridSearchCV,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
import os
rng=np.random.default_rng(22)
with tempfile.TemporaryDirectory() as tmp:
 data={}
 n=60
 df=pd.DataFrame({'cbsa_code':[f'{i+10000:05}' for i in range(n)],
                  'geographic_group':np.repeat([f'Div-{d}' for d in range(10)],6)})
 CORE=['qcew2018to2019q2_growth','log_qcew2019q2_employment','acs2018_bachelors_plus_share',
       'acs2018_poverty_share','acs2018_unemployment_share','acs2018_broadband_subscription_share']
 INDUSTRY=[f'qcew2019q2_share_naics_{i}' for i in range(19)]+[
       'qcew2019q2_private_share_all_covered','qcew2019q2_private_goods_producing_share']
 for name in CORE+INDUSTRY:df[name]=rng.normal(size=n)
 df.loc[df.index%4==0,'acs2018_poverty_share']=np.nan
 df['qcew2019q2_employment']=rng.uniform(1000,100000,size=n)
 for target in ['resistance_shortfall_pct_aprjun2020','persistence_shortfall_pct_octdec2023']:
  df[target]=np.abs(rng.normal(size=n)*10)
 df['recovery_gain_from_trough_pp']=rng.normal(size=n)*20
 df['cumulative_shortfall_12_pp_months']=rng.uniform(0,50,n)
 df['cumulative_shortfall_24_pp_months']=df['cumulative_shortfall_12_pp_months']+rng.uniform(0,100,n)
 df['cumulative_shortfall_46_pp_months']=df['cumulative_shortfall_24_pp_months']+rng.uniform(0,300,n)
 df['signed_cumulative_gap_46_pp_months']=df['cumulative_shortfall_46_pp_months']-rng.uniform(0,60,n)
 for name in ['ensemble','damped']:
  df['cumulative_shortfall_46_pp_months_'+name]=df['cumulative_shortfall_46_pp_months']*rng.uniform(.8,1.1,n)
 TARGETS={
     'resistance':'resistance_shortfall_pct_aprjun2020',
     'recovery':'recovery_gain_from_trough_pp',
     'persistence':'persistence_shortfall_pct_octdec2023',
     'c12':'cumulative_shortfall_12_pp_months',
     'c24':'cumulative_shortfall_24_pp_months',
     'c46':'cumulative_shortfall_46_pp_months',
     'signed46':'signed_cumulative_gap_46_pp_months'}
 for origin in ['2013_06','2014_06','2014_12']:
  col='cumulative_shortfall_46_pp_months_stress_'+origin
  df[col]=df['cumulative_shortfall_46_pp_months']*rng.uniform(.75,1.25,n)
  TARGETS['stress_'+origin]=col
 FEATURE_SETS={'core':CORE,'industry':INDUSTRY,'combined':CORE+INDUSTRY}
 N_JOBS=1;SEED=2026;MIN_FEATURE_COVERAGE=.50;BOOTSTRAP_REPS=30
 ENABLE_HGB=False;INCLUDE_HISTORICAL_STRESS=True;MAX_OUTER_DIVISIONS=1;PREFIX='smoke_'
 OUT=Path(tmp)
 pd.DataFrame({'cbsa_code':df.cbsa_code,'model':rng.choice(['ets','sarima','seasonal_naive'],n),
               'specification':['fixture']*n}).to_csv(OUT/'stage1_omb2018_selected_models.csv',index=False)
 def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 s1meta={'selected_models_sha256':sha256(OUT/'stage1_omb2018_selected_models.csv')};s2meta={}
 nbook=nbformat.read(NOTEBOOKS_DIR/'04_Cumulative_Shortfall_Geographic_Validation.ipynb',as_version=4)
 g=globals()
 print(f'Run 04 synthetic CV smoke test (1 held-out division, {len(TARGETS)} outcomes)...')
 for i in (7,9,11,13):
  print('  CELL',i,flush=True)
  exec(nbook.cells[i].source,g)
  if i==7:
   fitted=ridge_pipeline().fit(df[CORE],df['cumulative_shortfall_46_pp_months'])
   names=fitted.named_steps['impute'].get_feature_names_out(fitted.named_steps['coverage'].cols_)
   assert any(x.startswith('missingindicator_') for x in names), names
   print('  PASS transformed missingness indicator names')
 assert set(predictions.target)==set(TARGETS)
 assert len(predictions)==6*len(TARGETS)
 assert not pooled['oos_r2_vs_training_mean'].isna().all()
 assert set(criteria_df.target)==set(TARGETS)
 assert len(weighted)==len(TARGETS)
 assert 'transformed_feature' in outer_coefs.columns
 assert boot.empty or (boot['bootstrap_mean_advantage_pp_months']-boot['mae_advantage_over_median_pp_months']).abs().max()>=0
 print('PASS nested CV, pooled/fold R2, bootstrap, weighted metrics, criteria, scenario diagnostics')
