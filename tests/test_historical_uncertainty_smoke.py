"""End-to-end synthetic Stage-1 pseudo-origin stress paths -> Stage-2 outcome test.
No QCEW downloads; does not establish calibrated confidence intervals.
"""
import hashlib
import tempfile
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
import warnings
from joblib import Parallel,delayed
rng=np.random.default_rng(1303)
NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'
n1=nbformat.read(NOTEBOOKS_DIR/'01_Stage1_Counterfactual_OMB2018.ipynb',as_version=4)
n3=nbformat.read(NOTEBOOKS_DIR/'03_Stage2_Resilience_and_Initial_Prediction.ipynb',as_version=4)
with tempfile.TemporaryDirectory() as tmp:
    PROC=Path(tmp)/'data'/'processed';OUT=Path(tmp)/'output'
    PROC.mkdir(parents=True);OUT.mkdir()
    pre_dates=pd.date_range('2010-01-01','2019-06-01',freq='MS')
    dates=pd.date_range('2010-01-01','2023-12-01',freq='MS')
    future_dates=pd.date_range('2019-07-01','2023-12-01',freq='MS')
    assert len(future_dates)==54
    tasks=[];panels=[];forecasts=[]
    for i,cbsa in enumerate(('11111','22222','33333')):
        y=1200+600*i+np.arange(len(dates))*(.5+0.15*i)+20*np.sin(2*np.pi*np.arange(len(dates))/12)
        y=y.astype(float)
        pre_y=y[:len(pre_dates)];tasks.append((cbsa,pre_y))
        panels.append(pd.DataFrame({'cbsa_code':cbsa,'date':dates,'employment':y}))
        base=np.tile(pre_y[-12:],5)[:54]
        forecasts.append(pd.DataFrame({'cbsa_code':cbsa,'date':future_dates,
            'counterfactual_employment':base,
            'counterfactual_ensemble_employment':base*.995,
            'counterfactual_damped_employment':base*.998}))
    panel=pd.concat(panels,ignore_index=True)
    forecast=pd.concat(forecasts,ignore_index=True)
    baselines=[{'family':'seasonal_naive','spec':'seasonal_naive','config':{}},
               {'family':'seasonal_drift','spec':'seasonal_drift','config':{}},
               {'family':'trend_seasonal','spec':'trend_seasonal_log','config':{}}]
    c={'np':np,'pd':pd,'PROC':PROC,'OUT':OUT,'panel':panel,'forecast':forecast,
       'tasks':tasks,'selected':pd.DataFrame({'cbsa_code':[a for a,_ in tasks],
                                           'specification':['seasonal_naive']*3}),
       'candidates':baselines,'BASELINES':baselines,'LONG_CV':True,'RETRO_SPEC':False,
       'N_JOBS':1,'Parallel':Parallel,'delayed':delayed,
       'threadpool_limits':threadpool_limits,'warnings':warnings,'LinearRegression':LinearRegression,
       'RUN_ETS':False,'RUN_SARIMA':False,'ETS_GRID':'fast','SARIMA_GRID':'smoke',
       'LOG_BIAS_ADJUST':True,'FORECAST_MAX_LEVEL_MULTIPLE':4.0,
       'FORECAST_MIN_LEVEL_MULTIPLE':.25}
    exec(n1.cells[5].source,c)
    exec(n1.cells[11].source,c)
    assert len(c['historical_monthly'])==3*3*54
    assert len(c['stress'])==3*3*46
    assert len(c['stress_outcomes'])==3*3
    assert c['stress_outcomes'].groupby('cbsa_code').stress_origin.nunique().eq(3).all()
    assert c['stress'].counterfactual_stress_employment.gt(0).all()
    assert c['historic_cv'].loc[c['historic_cv'].strategy.eq('preorigin_selected_baseline'),'nmae'].notna().all()
    print('PASS pre-origin baseline selection, 54-month extrapolation, 46-month stress paths')
    # Build synthetic Stage-1 monthly pandemic file, then compute Stage-2 outcomes.
    gaps=panel.loc[panel.date.ge('2020-03-01')].merge(
        forecast,on=['cbsa_code','date'],validate='one_to_one')
    for suffix, col in (('', 'counterfactual_employment'),
                        ('_ensemble','counterfactual_ensemble_employment'),
                        ('_damped','counterfactual_damped_employment')):
        gaps['gap_pct'+suffix]=100*(gaps.employment/gaps[col]-1)
        gaps['shortfall_pct'+suffix]=(-gaps['gap_pct'+suffix]).clip(lower=0)
    filepath=PROC/'stage1_omb2018_observed_vs_counterfactual_2020_2023.csv'
    gaps.to_csv(filepath,index=False)
    stressfile=PROC/'stage1_omb2018_historical_stress_outcomes.csv'
    sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
    d={'DATA':filepath,'PROC':PROC,'np':np,'pd':pd,'sha256':sha,
       'STAGE1':{'historical_stress_paths_completed':True,
                 'historical_stress_origins':['2013_06','2014_06','2014_12'],
                 'historical_stress_outcomes_sha256':sha(stressfile)}}
    exec(n3.cells[3].source,d)
    for origin in d['STAGE1']['historical_stress_origins']:
        assert f'cumulative_shortfall_46_pp_months_stress_{origin}' in d['outcomes']
    assert len(d['outcomes'])==3
    print('PASS Stage-2 persistence of all three historical counterfactual stress targets')
print('ALL HISTORICAL STRESS TESTS PASSED')
