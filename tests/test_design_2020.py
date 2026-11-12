"""Offline boundary tests for the revised design. No empirical models/downloads."""
from pathlib import Path
from unittest.mock import patch
import sys
import unittest
import ast
import json
import time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import design_predictors as dp
import design_models as dm

class DesignChecks(unittest.TestCase):
    def test_geographic_partition_uses_whole_blocks(self):
        frame=pd.DataFrame({'geographic_group':np.repeat(list('ABCDEFGHIJ'),[10,20,30,40,50,60,70,80,90,100])})
        tr,te=dm.geographic_partition(frame)
        self.assertFalse(set(frame.iloc[tr].geographic_group)&set(frame.iloc[te].geographic_group))
        self.assertEqual(len(te),110)

    def test_release_gate(self):
        self.assertTrue(all(s['available_by']<='2020-02-29' for s in dp.SOURCES.values()))
        self.assertIn('/2018/',dp.SOURCES['acs']['url'])
        self.assertIn('24_1',dp.SOURCES['onet']['url'])

    def test_county_missing_is_not_partial_total(self):
        geo=pd.DataFrame({'cbsa_code':['A','A','B'],'county_fips':['1','2','3']})
        county=pd.DataFrame({'county_fips':['1','2','3'],'x':[10,np.nan,3],'y':[1,2,4]})
        result=dp.aggregate_counts(geo,county,['x','y'])
        self.assertTrue(pd.isna(result.loc['A','x']))
        self.assertEqual(result.loc['A','y'],3)
        self.assertEqual(result.loc['B','x'],3)

    def test_grouped_income_bins(self):
        counts=np.zeros(16);counts[8]=100
        self.assertEqual(dp.grouped_median(counts),47500)
        counts[:]=0;counts[-1]=100
        self.assertTrue(np.isnan(dp.grouped_median(counts)))

    def test_planned_outcomes_and_undefined_recovery(self):
        gap=np.full(46,-5.);gap[1:4]=[-10,-20,-5];gap[11]=5;gap[23]=-30
        t=pd.DataFrame({'date':pd.date_range('2020-03-01',periods=46,freq='MS'),'gap_pct':gap})
        result=dm.planned_outcomes(t)
        self.assertEqual(result['resistance_peak_shortfall_pct_aprjun2020'],20)
        self.assertEqual(result['recovery_fraction_closed_12'],1.25)
        self.assertEqual(result['recovery_fraction_closed_24'],-.5)
        self.assertEqual(result['persistence_shortfall_pct_12'],0)
        t['gap_pct']=5
        self.assertTrue(np.isnan(dm.planned_outcomes(t)['recovery_fraction_closed_46']))
        with self.assertRaises(ValueError):dm.planned_outcomes(t.iloc[:-1])

    def test_fold_only_preprocessing(self):
        train=pd.DataFrame({'is_micropolitan':[0]*30+[1]*30,
                            'x':[1.]*29+[np.nan]+[10.]*29+[np.nan],
                            'too_missing':[np.nan]*20+list(range(40))})
        prep=dm.DesignPreprocessor().fit(train)
        self.assertNotIn('too_missing',prep.columns_)
        extreme=pd.DataFrame({'is_micropolitan':[0,1], 'x':[np.nan,np.nan], 'too_missing':[999,999]})
        got=prep.transform(extreme)
        j=list(prep.get_feature_names_out()).index('x')
        np.testing.assert_allclose(got[:,j],[1,10])
        # Test values cannot alter stored medians or coverage decisions.
        extreme['x']=1e10;prep.transform(extreme)
        self.assertEqual(prep.medians_['x'],5.5)

    def test_laus_county_layout(self):
        table=pd.DataFrame([['header']+[None]*8,
            ['CN0100100000000','01','001','Autauga, AL','2018','10,000','9,500','500','5.0']])
        with patch.object(dp.pd,'read_excel',return_value=table):
            got=dp.read_laus(Path('fixture.xlsx'))
        self.assertEqual(got.iloc[0].county_fips,'01001')
        self.assertEqual(got.iloc[0].laus_unemployed,500)

    def test_oes_county_layout(self):
        table=pd.DataFrame([['Area Code','State FIPS','County FIPS'],['10100','46','013'],['10100','46','045']])
        with patch.object(dp.pd,'read_excel',return_value=table):
            got=dp.oes_county_sets(Path('fixture.xlsx'))
        self.assertEqual(got.loc['10100'],{'46013','46045'})

    def test_stage1_selects_rmse_not_mae_without_fitting_models(self):
        notebook=next((Path(__file__).resolve().parents[1]/'code').glob('01*.ipynb'))
        n=json.loads(notebook.read_text(encoding='utf-8'))
        tree=ast.parse(''.join(n['cells'][7]['source']))
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate_cbsa')
        candidates=[{'family':'seasonal_naive','spec':name,'config':{}} for name in ('A','B')]
        def fake_fit(y,h,c):
            error=np.full(h,2.) if c['spec']=='B' else np.r_[12.,np.zeros(h-1)]
            return 100+error
        ctx={'pd':pd,'np':np,'json':json,'time':time,'candidates':candidates,
             'origin_locs':[(pd.Timestamp('2015-12-01'),12)],'VALIDATION_MONTHS':12,
             'FORECAST_DATES':pd.date_range('2019-07-01',periods=54,freq='MS'),
             'TRAIN_END':pd.Timestamp('2019-06-01'),'fit_forecast':fake_fit,
             'seasonal_naive':lambda y,h:np.full(h,100.)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<selection-test>','exec'),ctx)
        result=ctx['evaluate_cbsa']('00001',np.full(114,100.))
        # A has lower MAE but higher RMSE; B must win the revised rule.
        self.assertEqual(result[0]['specification'],'B')
        self.assertEqual(len(result[3]),54)
        self.assertIn('counterfactual_common_ets_employment',result[3][0])

if __name__=='__main__':unittest.main()
