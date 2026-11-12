"""Release-gated additions for notebook 02. Importing this file downloads nothing.

Only pre-2020 statistical inputs are requested. Current BLS historical downloads
can nevertheless contain revisions; release eligibility is not vintage proof.
"""
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import re
import zipfile
import os
import shutil
import time
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import requests

CUTOFF = '2020-02-29'
SCHEMA = 'released_by_20200229_v2'
SOURCES = {
    'acs': {'url': 'https://api.census.gov/data/2018/acs/acs5',
            'available_by': '2019-12-19', 'reference': '2014-2018'},
    'laus': {'url': 'https://www.bls.gov/lau/laucnty18.xlsx',
             'available_by': '2019-12-31', 'reference': '2018 annual average',
             'note': 'Conservative eligibility bound, not an asserted exact first-release date.'},
    'oes': {'url': 'https://www.bls.gov/oes/special.requests/oesm18ma.zip',
            'available_by': '2019-03-29', 'reference': 'May 2018'},
    'oes_geography': {'url': 'https://www.bls.gov/oes/2018/may/area_definitions_m2018.xlsx',
                      'available_by': '2019-03-29', 'reference': 'May 2018 area definitions'},
    'onet': {'url': 'https://www.onetcenter.org/dl_files/database/db_24_1_text.zip',
             'available_by': '2019-11-30', 'reference': 'O*NET 24.1 November 2019'},
}

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def _validate_source(path, kind=None):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError('File is missing or empty')
    if kind in ('zip','xlsx') or path.suffix.lower()=='.xlsx':
        if not zipfile.is_zipfile(path): raise ValueError('Expected an archive, received a non-ZIP file (possibly an HTML error page)')
        with zipfile.ZipFile(path) as archive:
            names=set(archive.namelist())
            if kind=='xlsx' and not {'[Content_Types].xml','xl/workbook.xml'}.issubset(names):
                raise ValueError('ZIP content is not an Excel workbook')

def _browser_download(url, path, kind):
    """Adopt only exact-name downloads of the requested source, never a newer year."""
    folders=[]
    configured=os.getenv('RL_SOURCE_DOWNLOAD_DIR','').strip()
    if configured: folders.append(Path(configured).expanduser())
    folders.append(Path.home()/'Downloads')
    names=list(dict.fromkeys([path.name,Path(urlsplit(url).path).name]))
    for folder in folders:
        for name in names:
            candidate=folder/name
            if candidate.resolve()==path.resolve() or not candidate.is_file(): continue
            try: _validate_source(candidate,kind)
            except (ValueError,OSError,zipfile.BadZipFile): continue
            temporary=path.with_suffix(path.suffix+'.part')
            shutil.copyfile(candidate,temporary)
            _validate_source(temporary,kind)
            temporary.replace(path)
            print('REUSE BROWSER DOWNLOAD:',candidate,'->',path)
            return True
    return False

def fetch(url, path, kind=None):
    path = Path(path)
    kind='xlsx' if path.suffix.lower()=='.xlsx' else kind
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try: _validate_source(path,kind)
        except (ValueError,OSError,zipfile.BadZipFile) as exc:
            raise ValueError(f'Invalid cached source {path}: {exc}. Move it aside and download the exact original file again.') from None
        return path
    if _browser_download(url,path,kind): return path
    part = path.with_suffix(path.suffix + '.part')
    failure='unknown error'
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 240),
                              headers={'User-Agent': 'Regional labor resilience academic research'}) as response:
                response.raise_for_status()
                with part.open('wb') as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block: handle.write(block)
            _validate_source(part,kind)
            part.replace(path)
            return path
        except requests.exceptions.SSLError:
            failure='TLS certificate verification failed. Configure REQUESTS_CA_BUNDLE with your trusted CA bundle, or use the browser download below. TLS checks were not disabled.'
            break
        except requests.exceptions.HTTPError as exc:
            status=getattr(exc.response,'status_code',None)
            failure=f'HTTP {status}'
            if status not in (429,500,502,503,504): break
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout) as exc:
            failure=type(exc).__name__
        except (ValueError,OSError,zipfile.BadZipFile) as exc:
            failure=str(exc)
            break
        finally:
            part.unlink(missing_ok=True)
        if attempt<2: time.sleep(2**attempt)
    raise RuntimeError(
        f'Download unavailable: {failure}\n'
        f'Open the official source in your normal browser: {url}\n'
        f'Save the file as {Path(urlsplit(url).path).name} in your Downloads folder, '
        f'or save/copy it directly to: {path}\n'
        'Then rerun this cell. Exact-name files in Downloads are detected automatically. '
        'For a different download folder, set RL_SOURCE_DOWNLOAD_DIR in the kernel. '
        'The source year is unchanged; no alternative/newer dataset is substituted.'
    ) from None

def acs_counties(raw, variables):
    """Separate versioned cache; never reuse the smaller old ACS table."""
    digest = hashlib.sha256(','.join(sorted(variables)).encode()).hexdigest()[:12]
    dest = raw / 'acs' / f'acs2018_design_{digest}.csv'
    if dest.exists():
        result = pd.read_csv(dest, dtype='string')
    else:
        frames = []
        for offset in range(0, len(variables), 45):
            args = {'get': ','.join(variables[offset:offset+45]), 'for': 'county:*', 'in': 'state:*'}
            key = __import__('os').getenv('CENSUS_API_KEY', '').strip()
            if key: args['key'] = key
            # Never allow request URLs, response excerpts or chained exceptions to disclose the key.
            try:
                response = requests.get(SOURCES['acs']['url'], params=args, timeout=(30, 180))
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    raise ValueError('Unexpected Census payload')
                frame = pd.DataFrame(payload[1:], columns=payload[0])
                required = variables[offset:offset+45] + ['state', 'county']
                if not set(required).issubset(frame) or frame.duplicated(['state', 'county']).any():
                    raise ValueError('Invalid Census schema or duplicate county')
                frames.append(frame[required])
            except Exception:
                raise RuntimeError('ACS 2018 request failed. Check CENSUS_API_KEY and connectivity; request details withheld.') from None
        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(frame, on=['state', 'county'], how='outer', validate='one_to_one')
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix('.csv.part')
        result.to_csv(part, index=False)
        part.replace(dest)
    if not set(variables + ['state', 'county']).issubset(result):
        raise ValueError('Incomplete ACS design cache; remove invalid cache and retry')
    result['county_fips'] = result.state.str.zfill(2) + result.county.str.zfill(3)
    for variable in variables:
        result[variable] = pd.to_numeric(result[variable], errors='coerce')
        result.loc[result[variable] < 0, variable] = np.nan
    return result, dest

def aggregate_counts(geo, county, columns):
    merged = geo[['cbsa_code', 'county_fips']].merge(
        county[['county_fips'] + columns], on='county_fips', how='left', validate='one_to_one')
    group = merged.groupby('cbsa_code')[columns]
    expected = merged.groupby('cbsa_code').size()
    return group.sum(min_count=1).where(group.count().eq(expected, axis=0))

def grouped_median(counts):
    """Approximate median from additive household income bins, in 2018 dollars.

    County medians cannot be summed/averaged into a CBSA median. Open-ended
    median bins are left missing rather than assigned an arbitrary endpoint.
    """
    values = np.asarray(counts, float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        return np.nan
    edges = np.array([0,10000,15000,20000,25000,30000,35000,40000,
                      45000,50000,60000,75000,100000,125000,150000,200000,np.inf])
    total = values.sum(); cumulative = values.cumsum()
    j = int(np.searchsorted(cumulative, total/2))
    if j in (0,15) or values[j] == 0: return np.nan
    before = cumulative[j-1]
    return edges[j] + (total/2-before)/values[j]*(edges[j+1]-edges[j])

def read_laus(path):
    # Headers/footers vary across revisions. Identify data by LAUS county ID.
    raw = pd.read_excel(path, header=None, dtype=str, engine='openpyxl')
    rows = []
    for values in raw.itertuples(index=False, name=None):
        cells = [str(v).strip() for v in values]
        pos = next((i for i,v in enumerate(cells) if re.fullmatch(r'CN\d{13}', v)), None)
        if pos is None: continue
        # Published layout: LAUS code, state FIPS, county FIPS, title, year, LF, E, U, rate.
        if len(cells) <= pos+8: raise ValueError('LAUS county workbook layout changed')
        state = cells[pos+1].removesuffix('.0').zfill(2)
        county = cells[pos+2].removesuffix('.0').zfill(3)
        if cells[pos+4].removesuffix('.0') != '2018': continue
        def number(v): return pd.to_numeric(v.replace(',',''), errors='coerce')
        rows.append({'county_fips':state+county,'laus_labor_force':number(cells[pos+5]),
                     'laus_unemployed':number(cells[pos+7])})
    result = pd.DataFrame(rows)
    if result.empty or result.county_fips.duplicated().any():
        raise ValueError('No unique 2018 county rows parsed from LAUS source')
    bad=(result.laus_labor_force<=0)|(result.laus_unemployed<0)|(result.laus_unemployed>result.laus_labor_force)
    result.loc[bad,['laus_labor_force','laus_unemployed']]=np.nan
    return result

def oes_county_sets(path):
    """Use published FIPS fields, never an area-code-only geographic match."""
    raw = pd.read_excel(path, header=None, dtype=str, engine='openpyxl')
    raw = raw.map(lambda x: re.sub(r'[_\s]+',' ',str(x)).strip() if pd.notna(x) else x)
    header = next((i for i,row in raw.iterrows() if
                   any('county' in str(v).lower() and ('fips' in str(v).lower() or 'code' in str(v).lower()) for v in row)
                   and any(str(v).lower()=='area' or ('area' in str(v).lower() and 'code' in str(v).lower()) for v in row)), None)
    if header is None:
        raise ValueError('OES geography workbook FIPS headers not recognized; inspect the cached official workbook')
    data = raw.iloc[header+1:].copy()
    data.columns = [str(v).strip().lower() for v in raw.iloc[header]]
    def col(words):
        match = [c for c in data.columns if all(word in c for word in words)]
        if len(match) != 1: raise ValueError(f'Ambiguous OES geography header: {words}')
        return match[0]
    ac = 'area' if 'area' in data else col(['area','code'])
    cc = col(['county','code']) if any('county' in c and 'code' in c for c in data) else col(['county','fips'])
    norm = lambda s,n: s.astype('string').str.replace(r'\.0$','',regex=True).str.strip().str.zfill(n)
    area = norm(data[ac].ffill(),5)
    codes=data[cc].astype('string').str.replace(r'\.0$','',regex=True).str.strip()
    if codes.dropna().str.len().eq(5).all():
        fips=norm(codes,5)
    else:
        sc = col(['state','fips']) if any('state' in c and 'fips' in c for c in data) else col(['state','code'])
        fips=norm(data[sc].ffill(),2)+norm(codes,3)
    out = pd.DataFrame({'area':area,'county_fips':fips})
    out = out[out.area.str.fullmatch(r'\d{5}',na=False)&out.county_fips.str.fullmatch(r'\d{5}',na=False)]
    return out.groupby('area').county_fips.agg(lambda x:set(x))

def add_occupations(root, geo, result):
    """OES composition only where county sets agree exactly; no nonmetro allocation."""
    raw = root/'data/raw'; out = root/'output'
    oes = fetch(SOURCES['oes']['url'], raw/'oes/oesm18ma.zip', 'zip')
    defs = fetch(SOURCES['oes_geography']['url'], raw/'oes/area_definitions_m2018.xlsx')
    with zipfile.ZipFile(oes) as archive:
        books = [n for n in archive.namelist() if n.lower().endswith(('.xlsx','.xls')) and not Path(n).name.startswith('~')]
        tables = [pd.read_excel(BytesIO(archive.read(n)), dtype=str,
                               engine='openpyxl' if n.lower().endswith('.xlsx') else 'xlrd') for n in books]
    if not tables: raise ValueError('OES archive has no Excel data')
    data = pd.concat(tables, ignore_index=True)
    data.columns = data.columns.str.strip().str.upper()
    area_col = 'AREA' if 'AREA' in data else 'AREA_CODE'
    required = {area_col,'OCC_CODE','TOT_EMP'}
    if not required.issubset(data): raise ValueError(f'OES schema lacks {required-set(data)}')
    data['area'] = data[area_col].str.replace(r'\.0$','',regex=True).str.zfill(5)
    data['employment'] = pd.to_numeric(data.TOT_EMP.str.replace(',','',regex=False), errors='coerce')
    sets = oes_county_sets(defs)
    audit=[]; matched={}
    for cbsa, counties in geo.groupby('cbsa_code').county_fips.agg(set).items():
        exact = cbsa in sets and sets[cbsa] == counties
        audit.append({'cbsa_code':cbsa,'exact_oes_county_match':exact,
                      'reason':'exact county match' if exact else 'absent or different geography; left missing'})
        if exact: matched[cbsa] = data[data.area.eq(cbsa)].copy()
    pd.DataFrame(audit).to_csv(out/'stage2_omb2018_oes_geography_audit.csv',index=False)
    major = data.OCC_CODE.dropna().loc[lambda x:x.str.fullmatch(r'\d{2}-0000')].unique()
    for code in sorted(set(major)-{'00-0000'}):
        values={}
        for cbsa,t in matched.items():
            total=t.loc[t.OCC_CODE.eq('00-0000'),'employment']
            num=t.loc[t.OCC_CODE.eq(code),'employment']
            if len(total)==1 and len(num)==1 and total.iloc[0]>0:
                values[cbsa]=num.iloc[0]/total.iloc[0]
        result['oes2018_share_soc_'+code[:2]]=pd.Series(values)
    # Derive an explicitly labeled task proxy from archived inputs, rather than
    # downloading a post-cutoff published teleworkability score.
    onet = fetch(SOURCES['onet']['url'],raw/'onet/db_24_1_text.zip','zip')
    scores = task_telework_proxy(onet)
    scores.to_csv(out/'stage2_omb2018_onet_task_proxy_by_soc.csv',index=False)
    result['oes2018_pre2020_task_telework_proxy']=np.nan
    proxy_audit=[]
    for cbsa,t in matched.items():
        total=t.loc[t.OCC_CODE.eq('00-0000'),'employment']
        if 'O_GROUP' not in t: raise ValueError('OES O_GROUP field is required to distinguish detailed occupations')
        detail=t[t.O_GROUP.str.lower().eq('detailed')][['OCC_CODE','employment']]
        if detail.OCC_CODE.duplicated().any(): raise ValueError('Duplicate detailed OES occupations')
        detail=detail.merge(scores,left_on='OCC_CODE',right_on='soc',how='left',validate='one_to_one')
        valid=detail.employment.notna() & detail.task_remote_feasible.notna()
        denominator=float(total.iloc[0]) if len(total)==1 else np.nan
        represented=float(detail.loc[valid,'employment'].sum())
        coverage=represented/denominator if denominator>0 else np.nan
        if .90<=coverage<=1.02 and represented>0:
            result.loc[cbsa,'oes2018_pre2020_task_telework_proxy']=np.average(
                detail.loc[valid,'task_remote_feasible'],weights=detail.loc[valid,'employment'])
        proxy_audit.append({'cbsa_code':cbsa,'classified_employment_coverage':coverage,
                            'included':bool(.90<=coverage<=1.02)})
    pd.DataFrame(proxy_audit,columns=['cbsa_code','classified_employment_coverage','included']).to_csv(
        out/'stage2_omb2018_telework_coverage.csv',index=False)
    return [oes,defs,onet]

def task_telework_proxy(path):
    """Recreate Dingel--Neiman task rules using archived O*NET 24.1 inputs.

    Method reference (not a downloaded predictor dataset):
    github.com/jdingel/DingelNeiman-workathome, occ_onet_scores/code/onet_characteristics.do.
    Published classification appeared after the cutoff, so no scores or later
    employment inputs are imported. This adaptation uses the earlier 24.1 data,
    not the authors' 24.2 inputs. Missing required items remain missing; do not
    infer feasible work from the absence of an exclusion item.
    """
    with zipfile.ZipFile(path) as archive:
        def table(name):
            match=[n for n in archive.namelist() if Path(n).name.lower()==name.lower()]
            if len(match)!=1: raise ValueError(f'O*NET archive lacks {name}')
            return pd.read_csv(BytesIO(archive.read(match[0])),sep='\t',dtype=str)
        context=table('Work Context.txt'); activities=table('Work Activities.txt')
    items=[(activities,code,'IM',lambda x:x>=4) for code in
           ('4.A.3.a.1','4.A.3.a.2','4.A.3.a.3','4.A.3.a.4',
            '4.A.4.a.8','4.A.3.b.4','4.A.3.b.5','4.A.1.b.2')]
    items += [(context,'4.C.1.a.2.h','CX',lambda x:x<=2)]
    items += [(context,code,'CX',lambda x:x>=4.5) for code in ('4.C.2.a.1.c','4.C.2.a.1.d')]
    items += [(context,code,'CX',lambda x:x>=4) for code in ('4.C.1.d.3','4.C.2.c.1.f','4.C.2.c.1.b')]
    items += [(context,code,'CX',lambda x:x>=3.5) for code in ('4.C.2.e.1.d','4.C.2.e.1.e','4.C.2.d.1.d')]
    flags=[]
    for frame,label,scale,rule in items:
        t=frame[frame['Element ID'].eq(label)&frame['Scale ID'].eq(scale)]
        if t.empty: raise ValueError(f'Missing archived O*NET task item: {label}/{scale}')
        values=pd.to_numeric(t['Data Value'],errors='coerce')
        s=pd.Series(np.where(values.notna(),rule(values).astype(float),np.nan),index=t['O*NET-SOC Code'])
        if s.index.duplicated().any(): raise ValueError('Duplicate O*NET task score')
        flags.append(s.rename(label))
    matrix=pd.concat(flags,axis=1)
    score=(~matrix.gt(0).any(axis=1)).astype(float).where(matrix.notna().all(axis=1))
    score.index=score.index.str.slice(0,7)
    # O*NET specializations share one SOC; average their binary feasibility flags.
    out=score.groupby(level=0).agg(lambda x:x.mean() if x.notna().all() else np.nan)
    return out.rename('task_remote_feasible').rename_axis('soc').reset_index()

def extend_predictors(root, predictors, geo, shares, q19):
    root=Path(root); raw=root/'data/raw'; out=root/'output'; proc=root/'data/processed'
    if any(v['available_by']>CUTOFF for v in SOURCES.values()):
        raise ValueError('Post-cutoff predictor source prohibited')
    ids = predictors.cbsa_code.astype('string').str.zfill(5)
    result = pd.DataFrame(index=pd.Index(ids,name='cbsa_code'))
    # Broad occupational counts are additive and cover metropolitan and micro areas.
    occ={'management_business_science_arts':(3,39),'service':(19,55),
         'sales_office':(27,63),'natural_resources_construction_maintenance':(30,66),
         'production_transport_material':(34,70)}
    variables=['B23025_001E','B23025_003E','C24010_001E']
    variables += [f'B19001_{i:03}E' for i in range(1,18)]
    variables += [f'C24010_{i:03}E' for pair in occ.values() for i in pair]
    acs,acs_path=acs_counties(raw,variables)
    summed=aggregate_counts(geo,acs,variables).reindex(result.index)
    result['acs2018_labor_force_participation_share']=summed.B23025_003E/summed.B23025_001E.where(summed.B23025_001E>0)
    med=summed[[f'B19001_{i:03}E' for i in range(2,18)]].apply(grouped_median,axis=1)
    bins=summed[[f'B19001_{i:03}E' for i in range(2,18)]].sum(axis=1,min_count=16)
    med=med.where(np.isclose(bins,summed.B19001_001E,rtol=.01,atol=1))
    result['acs2018_log_grouped_median_household_income']=np.log(med.where(med>0))
    for label,pair in occ.items():
        numerator=summed[[f'C24010_{i:03}E' for i in pair]].sum(axis=1,min_count=2)
        result['acs2018_occupation_share_'+label]=numerator/summed.C24010_001E.where(summed.C24010_001E>0)
    county=pd.read_csv(proc/'qcew_county_monthly_2010_2023.csv',dtype={'county_fips':'string'},parse_dates=['date'])
    for year in (2015,2018):
        t=county[county.date.dt.year.eq(year)]
        valid=t.groupby('county_fips').employment.agg(['mean','count'])
        valid.loc[valid['count'].ne(12),'mean']=np.nan
        totals=aggregate_counts(geo,valid['mean'].rename('emp').reset_index(),['emp']).emp
        result[f'qcew{year}_annual_employment']=totals
    result['qcew2015to2018_annualized_log_growth']=(np.log(result.qcew2018_annual_employment)-np.log(result.qcew2015_annual_employment))/3
    result['log_qcew2018_annual_employment']=np.log(result.qcew2018_annual_employment.where(result.qcew2018_annual_employment>0))
    cw=pd.read_csv(proc/'county_cbsa_crosswalk_sep2018.csv',dtype={'cbsa_code':'string'})
    kinds=cw[['cbsa_code','cbsa_type']].drop_duplicates('cbsa_code').set_index('cbsa_code').cbsa_type
    result['is_micropolitan']=kinds.reindex(result.index).eq('Micropolitan Statistical Area').astype(int)
    result['cbsa_type']=kinds.reindex(result.index)
    sector=[c for c in shares if c.startswith('qcew2019q2_share_naics_')]
    s=shares[sector].reindex(result.index)
    complete=s.notna().all(axis=1)&s.sum(axis=1).between(.98,1.02)
    result['qcew2019q2_industry_diversification']=1-s.pow(2).sum(axis=1)
    result.loc[~complete,'qcew2019q2_industry_diversification']=np.nan
    result['industry_diversification_complete']=complete.astype(int)
    # Primary industry exposures use covered-employment denominators, with
    # government and private ownerships represented in the numerator.
    for industry in ('72','31-33','62','54'):
        records=q19[q19.industry_code.eq(industry)]
        ownerships=pd.DataFrame({'own_code':['1','2','3','5']})
        grid=geo[['cbsa_code','county_fips']].merge(ownerships,how='cross').merge(
            records[records.own_code.isin(['1','2','3','5'])][['county_fips','own_code','average_employment','disclosure_code']],
            on=['county_fips','own_code'],how='left',indicator=True,validate='one_to_one')
        grid.loc[grid['_merge'].eq('left_only'),'average_employment']=0.
        grid.loc[grid.disclosure_code.fillna('').eq('N'),'average_employment']=np.nan
        bycounty=grid.groupby('county_fips').average_employment.agg(['sum','count'])
        bycounty.loc[bycounty['count'].ne(4),'sum']=np.nan
        total=aggregate_counts(geo,bycounty['sum'].rename('emp').reset_index(),['emp']).emp
        denominator=predictors.set_index('cbsa_code').qcew2019q2_employment
        result['qcew2019q2_covered_share_naics_'+industry.replace('-','_')]=total/denominator.where(denominator>0)
    laus_path=fetch(SOURCES['laus']['url'],raw/'laus/laucnty18.xlsx')
    laus=aggregate_counts(geo,read_laus(laus_path),['laus_labor_force','laus_unemployed'])
    result['laus2018_unemployment_share']=laus.laus_unemployed/laus.laus_labor_force.where(laus.laus_labor_force>0)
    files=[acs_path,laus_path]+add_occupations(root,geo,result)
    (out/'stage2_omb2018_telework_status.json').write_text(json.dumps({
        'status':'Dingel_Neiman_rule_recreation_archived_inputs','reason':'Published task rules applied to archived O*NET 24.1, weighted by May 2018 OES detailed occupations; exact geography only and >=90% coverage.',
        'not_equivalent_to':'Authors published scores based on O*NET 24.2; measured actual remote work',
        'post_cutoff_Dingel_Neiman_scores_downloaded':False},indent=2))
    merged=predictors.set_index('cbsa_code').join(result,how='left',validate='one_to_one').reset_index()
    metadata={'schema':SCHEMA,'information_cutoff':CUTOFF,'sources':SOURCES,
              'raw_sha256':{str(p.relative_to(root)):sha256(p) for p in files},
              'historical_revision_vintage_verified':False,
              'income_method':'Linear within-bin median from aggregated county B19001 counts; approximate, 2018 dollars. Not average county medians.',
              'telework_status':'archived-input task feasibility proxy; supplemental OES exact-match sample only',
              'reference_categories':'Omit NAICS 81 and production/transport/material occupational share in primary regression.'}
    (out/'stage2_omb2018_extended_predictors_manifest.json').write_text(json.dumps(metadata,indent=2))
    return merged, metadata
