"""Offline deterministic safeguards for the 12 vs 24/36/48-month Stage-1 CV designs."""
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'
import nbformat,os,pandas as pd
code=nbformat.read(NOTEBOOKS_DIR/'01_Stage1_Counterfactual_OMB2018.ipynb',as_version=4).cells[1].source
selection_config=code[code.index('# The original prespecified design'):code.index('RUN_ETS =')]
old_a=os.environ.get('RL_STAGE1_SELECTION_HORIZON')
old_b=os.environ.get('RL_ALLOW_ALTERNATE_SELECTION')
try:
    for horizon,origin in [(12,'2016-06-01'),(24,'2015-06-01'),(36,'2014-06-01'),(48,'2013-06-01')]:
        os.environ['RL_STAGE1_SELECTION_HORIZON']=str(horizon)
        os.environ['RL_ALLOW_ALTERNATE_SELECTION']='1'
        ns={'pd':pd,'os':os};exec(selection_config,ns)
        assert ns['VALIDATION_MONTHS']==horizon
        assert str(ns['VALIDATION_ORIGINS'][0].date())==origin
        assert ns['VALIDATION_ORIGINS'][-1]+pd.DateOffset(months=horizon)<=pd.Timestamp('2019-06-01')
    os.environ['RL_STAGE1_SELECTION_HORIZON']='48'
    os.environ['RL_ALLOW_ALTERNATE_SELECTION']='0'
    try:
        exec(selection_config,{'pd':pd,'os':os})
        raise AssertionError('Unsafe alternate selection was permitted without opt-in')
    except ValueError as e:
        assert 'RL_ALLOW_ALTERNATE_SELECTION' in str(e)
finally:
    if old_a is None:os.environ.pop('RL_STAGE1_SELECTION_HORIZON',None)
    else:os.environ['RL_STAGE1_SELECTION_HORIZON']=old_a
    if old_b is None:os.environ.pop('RL_ALLOW_ALTERNATE_SELECTION',None)
    else:os.environ['RL_ALLOW_ALTERNATE_SELECTION']=old_b
print('PASS 12/24/36/48 month CV origins, cutoff boundaries and explicit alternate-design opt-in')
