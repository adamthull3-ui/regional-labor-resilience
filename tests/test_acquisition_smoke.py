"""Synthetic audit and provenance test for notebook 00 section 3."""
from pathlib import Path

# Locate code/ relative to this script, regardless of the current VS Code working directory.
NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'code'
import nbformat,pandas as pd,numpy as np,hashlib,shutil,json,tempfile
from datetime import datetime,timezone
with tempfile.TemporaryDirectory() as tmp:
 ROOT=Path(tmp);PROC=ROOT/'data'/'processed';OUT=ROOT/'output'
 PROC.mkdir(parents=True);OUT.mkdir()
 CROSSWALK_FILE=PROC/'county_cbsa_crosswalk_sep2018.csv'
 COUNTY_FILE=PROC/'qcew_county_monthly_2010_2023.csv'
 PANEL_FILE=PROC/'qcew_cbsa_monthly_2010_2023_omb2018.csv'
 cw=pd.DataFrame({'cbsa_code':['11111','22222','33333'],
 'county_fips':['01001','01003','01005'],'cbsa_type':['Metropolitan','Micropolitan','Micropolitan'],
 'state_fips':['01','01','01']})
 cw.to_csv(CROSSWALK_FILE,index=False)
 EXPECTED_DATES=pd.date_range('2010-01-01','2023-12-01',freq='MS')
 county=pd.concat([pd.DataFrame({'county_fips':id,'date':EXPECTED_DATES,
   'employment':1000.0 if id!='01005' else 2000.}) for id in cw.county_fips],ignore_index=True)
 county.loc[(county.county_fips=='01005')&(county.date==pd.Timestamp('2021-03-01')),'employment']=np.nan
 county.to_csv(COUNTY_FILE,index=False)
 def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 cel=nbformat.read(NOTEBOOKS_DIR/'00_Acquire_QCEW_OMB2018.ipynb',as_version=4).cells[7]
 ctx=dict(globals());exec(cel.source,ctx)
 assert int(ctx['ca'].pre_cutoff_eligible.sum())==3
 assert int(ctx['ca'].full_history_included.sum())==2
 assert int(ctx['ca'].excluded_only_due_to_post_cutoff.sum())==1
 assert int(ctx['manifest']['n_cbsa'])==2
 assert sha256(PANEL_FILE)==ctx['manifest']['panel_sha256']
 # Manifest changed even if panel is unchanged: previous manifest is archived.
 exec(cel.source,ctx)
 assert any((OUT/'derived_history').rglob('omb2018_geography_manifest.json'))
 print('PASS balanced panel completeness/post-cutoff selection audits and versioned manifests')
