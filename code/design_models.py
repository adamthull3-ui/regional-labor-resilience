"""Prior-design Stage 2 specification with strict pre-2020 predictor whitelist.

Notebook 04 calls run(). No training, download or writing occurs on import.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import subprocess
import warnings
from functools import partial
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn

SCHEMA='released_by_20200229_v2'
SEED=2026
PRIMARY_FEATURES=[
 'qcew2019q2_covered_share_naics_72','qcew2019q2_covered_share_naics_31_33',
 'qcew2019q2_covered_share_naics_62','qcew2019q2_covered_share_naics_54',
 'acs2018_bachelors_plus_share','acs2018_log_grouped_median_household_income',
 'acs2018_labor_force_participation_share','laus2018_unemployment_share',
 'qcew2015to2018_annualized_log_growth','log_qcew2018_annual_employment',
 'is_micropolitan','qcew2019q2_industry_diversification',
 'oes2018_pre2020_task_telework_proxy']
PRIMARY_TARGETS={
 'resistance':'resistance_peak_shortfall_pct_aprjun2020',
 'recovery12':'recovery_fraction_closed_12',
 'recovery24':'recovery_fraction_closed_24',
 'recovery46':'recovery_fraction_closed_46',
 'persistence12':'persistence_shortfall_pct_12',
 'persistence24':'persistence_shortfall_pct_24',
 'persistence46':'persistence_shortfall_pct_46',
 'c46':'cumulative_shortfall_46_pp_months',
 'c12':'cumulative_shortfall_12_pp_months',
 'c24':'cumulative_shortfall_24_pp_months'}

class DesignPreprocessor(BaseEstimator,TransformerMixin):
    """Fold-local 80% coverage, group medians, >=5% missingness indicators.

    is_micropolitan is carried into each pipeline solely as observed metadata
    for imputation as well as a prespecified predictor. It must never be missing.
    """
    def __init__(self,min_coverage=.8,min_group_observed=20,indicator_threshold=.05):
        self.min_coverage=min_coverage
        self.min_group_observed=min_group_observed
        self.indicator_threshold=indicator_threshold
    def fit(self,X,y=None):
        if not isinstance(X,pd.DataFrame): raise TypeError('Named DataFrame required')
        if 'is_micropolitan' not in X or X.is_micropolitan.isna().any():
            raise ValueError('Observed CBSA type required for group imputation')
        numeric=X.astype(float).replace([np.inf,-np.inf],np.nan)
        self.columns_=[c for c in numeric if numeric[c].notna().mean()>=self.min_coverage
                       and numeric[c].nunique(dropna=True)>1]
        if not self.columns_: raise ValueError('No predictors satisfy training-only coverage rules')
        self.medians_=numeric[self.columns_].median()
        self.indicators_=[c for c in self.columns_ if numeric[c].isna().mean()>=self.indicator_threshold]
        self.group_medians_={}
        for group in (0,1):
            t=numeric.loc[numeric.is_micropolitan.eq(group),self.columns_]
            self.group_medians_[group]=t.median().where(t.count()>=self.min_group_observed,self.medians_)
        return self
    def transform(self,X):
        numeric=X.astype(float).replace([np.inf,-np.inf],np.nan)
        missing=numeric[self.indicators_].isna().astype(float)
        filled=numeric[self.columns_].copy()
        for group in (0,1):
            mask=numeric.is_micropolitan.eq(group)
            filled.loc[mask]=filled.loc[mask].fillna(self.group_medians_[group])
        filled=filled.fillna(self.medians_)
        if filled.isna().any().any(): raise ValueError('Imputation left missing values')
        return np.column_stack([filled.to_numpy(float),missing.to_numpy(float)])
    def get_feature_names_out(self,input_features=None):
        return np.array(self.columns_+['missingindicator_'+c for c in self.indicators_],dtype=object)

def planned_outcomes(s):
    """Index 0 is March 2020. Recovery fractions are unbounded and may be negative."""
    s=s.sort_values('date').reset_index(drop=True)
    expected=pd.date_range('2020-03-01','2023-12-01',freq='MS')
    if not pd.DatetimeIndex(s.date).equals(expected): raise ValueError('Incorrect outcome month grid')
    gap=s.gap_pct.to_numpy(float); shortfall=np.maximum(-gap,0)
    peak=float(shortfall[1:4].max())
    result={'resistance_peak_shortfall_pct_aprjun2020':peak}
    for h in (12,24,46):
        # Signed improvement retains >100% recovery after overshoot and
        # negative recovery when the fixed-horizon gap is worse than the peak.
        result[f'recovery_fraction_closed_{h}']=(peak+gap[h-1])/peak if peak>0 else np.nan
        result[f'persistence_shortfall_pct_{h}']=float(shortfall[h-1])
    result['recovery_initial_peak_positive']=int(peak>0)
    return pd.Series(result)

def feature_sets(pred):
    absent=set(PRIMARY_FEATURES)-set(pred)
    if absent: raise ValueError(f'Rerun notebook 02; missing planned predictors: {sorted(absent)}')
    industry=sorted(c for c in pred if c.startswith('qcew2019q2_share_naics_') and not c.endswith('_81'))
    occupations=sorted(c for c in pred if c.startswith('acs2018_occupation_share_')
                       and not c.endswith('production_transport_material'))
    oes=sorted(c for c in pred if c.startswith('oes2018_share_soc_') and not c.endswith('_53'))
    return {'primary':PRIMARY_FEATURES,
            'expanded':list(dict.fromkeys(PRIMARY_FEATURES+industry+occupations)),
            'occupation_supplement':list(dict.fromkeys(PRIMARY_FEATURES+occupations+oes))}

def candidates(include_expanded=True):
    # RF/HGB compete as named comparisons, not substitutes for Elastic Net.
    specs=[('mean',DummyRegressor(strategy='mean'),{}),
           ('median',DummyRegressor(strategy='median'),{}),
           ('ols',LinearRegression(),{}),
           ('elasticnet',ElasticNet(max_iter=50000,random_state=SEED),
            {'model__alpha':[.001,.01,.1,1.,10.,100.], 'model__l1_ratio':[.1,.5,.9,1.]}),
           ('random_forest',RandomForestRegressor(n_estimators=300,n_jobs=1,random_state=SEED),
            {'model__max_depth':[None,8], 'model__min_samples_leaf':[5,20], 'model__max_features':[.5,1.]}),
           ('gradient_boosting',HistGradientBoostingRegressor(loss='squared_error',max_iter=160,
                 early_stopping=False,random_state=SEED),
            {'model__max_leaf_nodes':[7,15], 'model__l2_regularization':[1.,10.]})]
    for name,model,params in specs:
        if name in ('mean','median'):
            yield name,model,params
        else:
            yield name,Pipeline([('prepare',DesignPreprocessor()),('scale',StandardScaler()),('model',model)]),params

def scores(y,p,bmean,bmedian,weights=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    mae=float(np.average(abs(y-p),weights=weights))
    med=float(np.average(abs(y-bmedian),weights=weights))
    denom=np.average((y-bmean)**2,weights=weights)
    return {'n':len(y),'mae':mae,'rmse':float(np.sqrt(np.average((y-p)**2,weights=weights))),
            'r2':float(r2_score(y,p,sample_weight=weights)) if len(y)>1 else np.nan,
            'oos_r2_vs_training_mean':float(1-np.average((y-p)**2,weights=weights)/denom) if denom>0 else np.nan,
            'baseline_median_mae':med,'skill_vs_median_pct':100*(1-mae/med) if med>0 else np.nan,
            'bias':float(np.average(p-y,weights=weights))}

def original_scale_rmse(estimator,X,y,nonnegative):
    prediction=np.asarray(estimator.predict(X),float)
    if nonnegative: prediction=np.maximum(prediction,0)
    return -float(np.sqrt(np.mean((np.asarray(y,float)-prediction)**2)))

def original_scale_mae(estimator,X,y,nonnegative):
    prediction=np.asarray(estimator.predict(X),float)
    if nonnegative: prediction=np.maximum(prediction,0)
    return -float(np.mean(abs(np.asarray(y,float)-prediction)))

def geographic_partition(frame):
    """Choose whole blocks nearest 20% of CBSAs, using membership counts only."""
    counts=frame.geographic_group.value_counts()
    n_groups=max(1,int(round(.2*len(counts))))
    if len(counts)-n_groups<5: raise ValueError('Too few development groups after final partition')
    choices=list(combinations(sorted(counts.index),n_groups))
    held=min(choices,key=lambda groups:(abs(sum(counts[g] for g in groups)-.2*len(frame)),groups))
    mask=frame.geographic_group.isin(held).to_numpy()
    return np.flatnonzero(~mask),np.flatnonzero(mask)

def evaluate(frame, targets, features, out, *, n_jobs=2, final_holdout=True):
    """Development outer CV then one revised holdout assessment, never called untouched."""
    groups=frame.geographic_group.astype(str)
    required_groups=6 if final_holdout else 5
    if groups.nunique()<required_groups: raise ValueError(f'Need >={required_groups} geographic groups')
    tr,te=geographic_partition(frame) if final_holdout else (np.arange(len(frame)),np.array([],dtype=int))
    development=frame.iloc[tr].copy() if final_holdout else frame.copy()
    final=frame.iloc[te].copy() if final_holdout else frame.iloc[:0]
    membership=frame[['cbsa_code','geographic_group','cbsa_type']].copy()
    membership['partition']=np.where(membership.cbsa_code.isin(final.cbsa_code),'revised_final_holdout','development')
    membership.to_csv(out/'design2020_geographic_partitions.csv',index=False)
    rows=[]; decisions=[]; coeffs=[]; importance=[]; failures=[]
    phases=[('development',development,LeaveOneGroupOut().split(development,groups=development.geographic_group))]
    if final_holdout:
        both=pd.concat([development,final],ignore_index=True)
        phases.append(('revised_final_holdout',both,[(np.arange(len(development)),np.arange(len(development),len(both)))]))
    for phase,data,splits in phases:
        for fold,(train_ix,test_ix) in enumerate(splits):
            training=data.iloc[train_ix]; testing=data.iloc[test_ix]
            if set(training.geographic_group)&set(testing.geographic_group): raise RuntimeError('Group leakage')
            for key,column in targets.items():
                tr=training[training[column].notna()]; te=testing[testing[column].notna()]
                if len(tr)<20 or len(te)<2 or tr.geographic_group.nunique()<4: continue
                xtr=tr[features].astype(float); xte=te[features].astype(float)
                ytr=tr[column].to_numpy(float); yte=te[column].to_numpy(float)
                inner=GroupKFold(n_splits=min(4,tr.geographic_group.nunique()))
                winners=[]
                for name,est,grid in candidates():
                    fit=GridSearchCV(est,grid,cv=inner,scoring=partial(original_scale_rmse,nonnegative=not key.startswith('recovery')),
                                      n_jobs=n_jobs,error_score='raise',refit=True)
                    try:
                        fit.fit(xtr,ytr,groups=tr.geographic_group)
                        predicted=fit.predict(xte)
                        if not key.startswith('recovery'): predicted=np.maximum(predicted,0)
                    except Exception as exc:
                        failures.append({'phase':phase,'fold':fold,'target':key,'model':name,
                                         'error':f'{type(exc).__name__}: {exc}'})
                        # Required primary models must run; do not silently
                        # compare a reduced candidate set as the planned design.
                        pd.DataFrame(failures).to_csv(out/'design2020_fit_failures.csv',index=False)
                        raise RuntimeError(f'Required model failed: {phase}/{fold}/{key}/{name}; see fit audit') from None
                    cv=float(-fit.best_score_)
                    decisions.append({'phase':phase,'fold':fold,'target':key,'model':name,
                                      'inner_rmse':cv,'params':json.dumps(fit.best_params_)})
                    winners.append((cv,name,predicted))
                    for cbsa,obs,pred,btype,weight,group in zip(te.cbsa_code,yte,predicted,te.cbsa_type,
                            te.qcew2019q2_employment,te.geographic_group):
                        rows.append({'phase':phase,'fold':fold,'target':key,'model':name,'cbsa_code':cbsa,
                            'geographic_group':group,'cbsa_type':btype,'observed':obs,'predicted':pred,
                            'baseline_mean':ytr.mean(),'baseline_median':np.median(ytr),'employment_weight':weight})
                    if name not in ('mean','median'):
                        pipe=fit.best_estimator_
                        prep=pipe.named_steps['prepare']; model=pipe.named_steps['model']
                        if hasattr(model,'coef_'):
                            coeffs += [{'phase':phase,'fold':fold,'target':key,'model':name,'feature':f,'coefficient':float(v)}
                                       for f,v in zip(prep.get_feature_names_out(),np.ravel(model.coef_))]
                        # Prediction inspection is terminal: never feeds back into selection.
                        pi=permutation_importance(pipe,xte,yte,n_repeats=5,random_state=SEED,
                            scoring=partial(original_scale_mae,nonnegative=not key.startswith('recovery')),n_jobs=1)
                        importance += [{'phase':phase,'fold':fold,'target':key,'model':name,'feature':f,
                                        'mae_increase':float(v)} for f,v in zip(features,pi.importances_mean)]
                # Selection rule fixed: RMSE, then model name for deterministic ties.
                cv,name,estimate=min(winners,key=lambda item:(item[0],item[1]))
                selected=[r.copy() for r in rows if r['phase']==phase and r['fold']==fold
                          and r['target']==key and r['model']==name]
                for r in selected: r['model']='inner_selected'; r['selected_model']=name
                rows.extend(selected)
            pd.DataFrame(rows).to_csv(out/'design2020_predictions_CHECKPOINT.csv',index=False)
    predictions=pd.DataFrame(rows)
    predictions.to_csv(out/'design2020_predictions.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'design2020_inner_decisions.csv',index=False)
    pd.DataFrame(coeffs).to_csv(out/'design2020_standardized_coefficients.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'design2020_heldout_permutation_importance.csv',index=False)
    pooled=[]; folds=[]
    for (phase,target,model),t in predictions.groupby(['phase','target','model']):
        for kind,s in [('all',t),*list(t.groupby('cbsa_type'))]:
            for weighting in ('equal_cbsa','employment'):
                w=s.employment_weight if weighting=='employment' else None
                pooled.append({'phase':phase,'target':target,'model':model,'cbsa_type':kind,'weighting':weighting,
                               **scores(s.observed,s.predicted,s.baseline_mean,s.baseline_median,w)})
        for group,s in t.groupby('geographic_group'):
            folds.append({'phase':phase,'target':target,'model':model,'geographic_group':group,
                          **scores(s.observed,s.predicted,s.baseline_mean,s.baseline_median)})
    pd.DataFrame(pooled).to_csv(out/'design2020_scores.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'design2020_group_scores.csv',index=False)
    # Paired group-resampling intervals, conditional on saved predictions.
    rng=np.random.default_rng(SEED); boot=[]
    for (phase,target,model),s in predictions.groupby(['phase','target','model']):
        names=s.geographic_group.unique()
        if len(names)<4: continue
        improvement=abs(s.observed-s.baseline_median)-abs(s.observed-s.predicted)
        t=s.assign(improvement=improvement)
        values=[]
        for _ in range(1000):
            chosen=rng.choice(names,len(names),replace=True)
            values.append(np.concatenate([t.loc[t.geographic_group.eq(g),'improvement'] for g in chosen]).mean())
        boot.append({'phase':phase,'target':target,'model':model,'point_mae_improvement':improvement.mean(),
                     'p2_5':np.quantile(values,.025),'p97_5':np.quantile(values,.975),'n_groups':len(names),
                     'interpretation':'descriptive; conditional on estimated counterfactual and fitted outer models'})
    pd.DataFrame(boot).to_csv(out/'design2020_group_bootstrap.csv',index=False)
    return predictions

def run(root):
    root=Path(root); proc=root/'data/processed'; out=root/'output'
    manifest=json.loads((out/'stage2_omb2018_modeling_manifest.json').read_text())
    if manifest.get('design_schema')!=SCHEMA: raise ValueError('Rerun revised notebooks 01--03 before revised 04')
    for filename,key in [('stage2_omb2018_pre2020_predictors.csv','predictors_sha256'),
                         ('stage2_omb2018_resilience_outcomes.csv','resilience_outcomes_sha256')]:
        if hashlib.sha256((proc/filename).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'Input hash mismatch: {filename}')
    pred=pd.read_csv(proc/'stage2_omb2018_pre2020_predictors.csv',dtype={'cbsa_code':'string'})
    target=pd.read_csv(proc/'stage2_omb2018_resilience_outcomes.csv',dtype={'cbsa_code':'string'})
    frame=target.merge(pred,on='cbsa_code',validate='one_to_one')
    sets=feature_sets(pred)
    for c in set(sum(sets.values(),[])):
        frame[c]=pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    coverage=pd.DataFrame({'feature':sets['primary'],'coverage':frame[sets['primary']].notna().mean().values})
    coverage.to_csv(out/'design2020_primary_feature_coverage.csv',index=False)
    print('Primary feature coverage (actual exclusion is learned inside each fold):')
    print(coverage.to_string(index=False))
    predictions=evaluate(frame,PRIMARY_TARGETS,sets['primary'],out,n_jobs=int(os.getenv('RL_N_JOBS','2')))
    metadata={'design_schema':SCHEMA,'information_cutoff':'2020-02-29','geography_vintage':'2018-09',
        'primary_targets':PRIMARY_TARGETS,'feature_sets':sets,'primary_model':'elasticnet',
        'inner_selection':'RMSE','preprocessing_min_coverage':.8,
        'final_holdout_note':'Revised holdout; original data and earlier CV results have already been inspected. Not an untouched confirmation set.',
        'spatial_groups':'nine Census divisions plus Puerto Rico, assigned by largest pre-cutoff county employment',
        'spatial_design_amendment':'Uses fixed state-division blocks; operational principal county is largest Q2 2019 county employment, not a purported official OMB principal-county designation.',
        'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,
        'prediction_sha256':hashlib.sha256((out/'design2020_predictions.csv').read_bytes()).hexdigest(),
        'limitations':['Current historical downloads can include subsequent revisions.',
                      'Task proxy is not published Dingel--Neiman classification.',
                      'Grouped income median is approximate.',
                      'Group bootstrap is conditional on estimated outcomes; not calibrated forecast uncertainty.']}
    (out/'design2020_manifest.json').write_text(json.dumps(metadata,indent=2))
    versions=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=False)
    if versions.returncode:
        raise RuntimeError('Models finished, but pip freeze failed; export the actual environment before reporting results')
    (out/'design2020_environment.txt').write_text(versions.stdout,encoding='utf-8')
    return predictions

def run_sensitivities(root):
    """Separately labeled experiments; never choose primary design by their scores."""
    root=Path(root); proc=root/'data/processed'; out=root/'output/design2020_sensitivities'
    main=json.loads((root/'output/design2020_manifest.json').read_text())
    if main.get('design_schema')!=SCHEMA: raise ValueError('Run revised notebook 04 first')
    pred=pd.read_csv(proc/'stage2_omb2018_pre2020_predictors.csv',dtype={'cbsa_code':'string'})
    outcomes=pd.read_csv(proc/'stage2_omb2018_resilience_outcomes.csv',dtype={'cbsa_code':'string'})
    frame=outcomes.merge(pred,on='cbsa_code',validate='one_to_one')
    sets=feature_sets(pred)
    experiments={
      'expanded_predictors':({'c46':PRIMARY_TARGETS['c46']},sets['expanded']),
      'occupation_supplement':({'c46':PRIMARY_TARGETS['c46']},sets['occupation_supplement']),
      'without_growth':({'c46':PRIMARY_TARGETS['c46']},[c for c in sets['primary'] if c!='qcew2015to2018_annualized_log_growth']),
      'counterfactuals':({name:f'cumulative_shortfall_46_pp_months_{name}' for name in ('ensemble','damped','common_ets','short_window')},sets['primary']),
      'historical_stress':({c.removeprefix('cumulative_shortfall_46_pp_months_'):c for c in frame
                           if c.startswith('cumulative_shortfall_46_pp_months_stress_')},sets['primary'])}
    for name,(targets,features) in experiments.items():
        dest=out/name; dest.mkdir(parents=True,exist_ok=True)
        if not targets or set(targets.values())-set(frame):
            raise ValueError(f'Missing sensitivity outcomes: {name}; rerun revised 01--03')
        print('Sensitivity:',name,flush=True)
        evaluate(frame,targets,features,dest,n_jobs=int(os.getenv('RL_N_JOBS','2')))
    # Alternative state block assignment; principal county remains fixed.
    # Merge Census divisions into four Census-region-like contiguous state blocks,
    # leaving Puerto Rico separate. Five groups permit nested development CV only.
    alt=frame.copy()
    regions={'New England':'Northeast','Middle Atlantic':'Northeast',
             'East North Central':'Midwest','West North Central':'Midwest',
             'South Atlantic':'South','East South Central':'South','West South Central':'South',
             'Mountain':'West','Pacific':'West','Puerto Rico':'Puerto Rico'}
    alt['geographic_group']=alt.geographic_group.map(regions)
    dest=out/'alternative_blocks';dest.mkdir(parents=True,exist_ok=True)
    # Preserve honest group count; five blocks are too few for a separate final
    # partition plus four inner training groups. Evaluate nested outer folds only.
    evaluate(alt,{'c46':PRIMARY_TARGETS['c46']},sets['primary'],dest,
             n_jobs=int(os.getenv('RL_N_JOBS','2')),final_holdout=False)
