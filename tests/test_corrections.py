"""Focused offline smoke tests; not a substitute for a full QCEW/Census rerun."""
from pathlib import Path
import nbformat, pandas as pd, numpy as np, tempfile, json
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
import warnings
NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'

def get_cell(name,i):return nbformat.read(NOTEBOOKS_DIR/name,as_version=4).cells[i].source

# Correct annual seasonal base offset; same twelve base months receive the same first-year drift.
code=get_cell('01_Stage1_Counterfactual_OMB2018.ipynb',5)
slice_=code.split('def trend_seasonal')[0]
ctx={'np':np};exec(slice_,ctx)
y=100+np.arange(120)*0.5+np.tile(np.sin(np.arange(12)),10)
p=ctx['seasonal_drift'](y,24)
assert np.allclose(p[:12],y[-12:]+ctx['seasonal_drift'](y,1)[0]-y[-12])
assert np.allclose(p[12:24]-p[:12],(y[-12:].mean()-y[:12].mean())/9)
print('PASS seasonal drift annual stepping')

# BLS missing cell logic: one missing industry is a structural zero; explicit N is missing.
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)
 geo=pd.DataFrame({'cbsa_code':['11111','22222'], 'county_fips':['01001','01003']})
 cols=['county_fips','own_code','industry_code','average_employment','disclosure_code']
 q18=pd.DataFrame([['01001','0','10',100,''],['01003','0','10',110,'']],columns=cols)
 q19=pd.DataFrame([['01001','0','10',110,''],['01003','0','10',120,''],
                   ['01001','5','10',75,''],['01003','5','10',80,''],
                   ['01001','5','11',10,''],['01003','5','11',11,''],
                   ['01001','5','22',np.nan,'N'],['01003','5','22',np.nan,'N']],columns=cols)
 core=get_cell('02_Download_PreFeb2020_Predictors.ipynb',7)
 private=['11','21','22'];broad=['101','102']
 c={'pd':pd,'np':np,'geo':geo,'q18':q18,'q19':q19,'PRIVATE_SECTORS':private,
    'BROAD_PRIVATE_SECTORS':broad,'OUT':out,'P19':None}
 exec(core,c)
 sector=c['status'].set_index(['cbsa_code','industry_code'])
 assert sector.loc[('11111','21'),'industry_employment']==0
 assert pd.isna(sector.loc[('11111','22'),'industry_employment'])
 assert sector.loc[('11111','21'),'n_structural_zero']==1
 assert sector.loc[('11111','22'),'n_suppressed_N']==1
 print('PASS absent QCEW cell zero / explicit N missing')

# Recovery eligibility and timing: pre-trough nonnegative observations cannot count.
code=get_cell('03_Stage2_Resilience_and_Initial_Prediction.ipynb',3)
code=code[code.index('def measure(s):'):code.index('rows=[]')]
c={'np':np,'pd':pd};exec(code,c)
months=pd.date_range('2020-03-01','2023-12-01',freq='MS')
gap=np.full(46,-3.)
gap[0:2]=2.;gap[2]=-15. # Apr trough? month 2 May trough; prior two months >0
# Three positives begin in June (month 3) and confirm August (month 5)
gap[3:6]=1.
s=pd.DataFrame({'date':months,'gap_pct':gap,'shortfall_pct':np.maximum(0,-gap),
                'employment':100.,'shortfall_pct_ensemble':np.maximum(0,-gap),
                'shortfall_pct_damped':np.maximum(0,-gap)})
r=c['measure'](s)
assert r['trough_date_2020']=='2020-05-01'
assert r['recovery_episode_start_date']=='2020-06-01'
assert r['recovery_episode_confirm_date']=='2020-08-01'
assert r['months_from_trough_to_recovery_start']==1
assert r['months_to_sustained_recovery']==3
s['gap_pct']=1;s['shortfall_pct']=0
r=c['measure'](s)
assert np.isnan(r['recovered_by_dec2023'])
print('PASS trough-dependent recovery start/confirmation and not-at-risk')

print('ALL BASIC TESTS PASSED')
