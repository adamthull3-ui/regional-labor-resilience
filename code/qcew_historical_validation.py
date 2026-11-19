"""Build a separate, fixed-geography QCEW history and matched C46 backtests."""
import csv
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from c46_forecast_validity import (
    FIXED_ENSEMBLE, NAMES, PANEL_FILE, TRAIN_MONTHS, accuracy,
    cbsa_csv, guarded_paths, select_name, sha, target_values,
)

HISTORY_START = "2001-01-01"
HISTORY_END = "2019-06-01"
HISTORY_DATES = pd.date_range(HISTORY_START, HISTORY_END, freq="MS")
CROSSWALK = "data/processed/county_cbsa_crosswalk_sep2018.csv"
COUNTIES = "data/processed/qcew_county_monthly_2010_2023.csv"
GEO_MANIFEST = "data/processed/omb2018_geography_manifest.json"
OUTPUT_DIR = "output/qcew_historical_validation"
PANEL_NAME = "cbsa_monthly_2001_2019_omb2018.csv"
BLS_ARCHIVE = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_by_area.zip"
CENSUS_2000 = "https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2000.html"
CENSUS_2010 = "https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html"

# Exact territory replacements. The two independent cities merged whole into their respective counties.
ALIASES = {"02158": "02270", "46102": "46113"}
MERGERS = {"51005": ("51560", "2001-07-01"), "51019": ("51515", "2013-07-01")}
# The affected source units cross 2018 CBSA boundaries or include territory outside a CBSA.
# Allow the next full reporting year after these mid-year changes.
BOUNDARY_GAPS = (
    (("08001", "08013", "08014", "08059", "08123"), "2002-01-01", "Broomfield creation"),
    (("02105", "02230"), "2008-01-01", "Skagway and Hoonah-Angoon split"),
    (("02130", "02195", "02198", "02275"), "2009-01-01", "Southeast Alaska split and annexation"),
    (("02105", "02195", "02198"), "2014-01-01", "Petersburg and adjacent boundary changes"),
)
EXCHANGE = ("51199", "51700")
RECENT_ORIGINS = pd.to_datetime(["2012-12-01", "2013-12-01", "2014-12-01"])
RECESSION_ORIGINS = pd.to_datetime(["2006-12-01", "2007-06-01", "2007-09-01"])
OUTER_HORIZON = 54


def read_county_archive(path, year, source_codes):
    """Read total-covered employment from the official QCEW by-area archive."""
    months = [f"month{i}_emplvl" for i in (1, 2, 3)]
    columns = {"area_fips", "own_code", "industry_code", "year", "qtr", "disclosure_code", *months}
    quarters = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            match = re.search(r"(?<!\d)(\d{5})(?!\d)", Path(member).name)
            if match and match[1] not in source_codes:
                continue
            with archive.open(member) as binary:
                from io import TextIOWrapper

                reader = csv.DictReader(TextIOWrapper(binary, encoding="latin1", newline=""))
                if not columns.issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected QCEW columns in {path.name}/{member}")
                for row in reader:
                    code = row["area_fips"].strip()
                    if (code in source_codes and row["year"].strip() == str(year)
                            and row["own_code"].strip() == "0" and row["industry_code"].strip() == "10"
                            and row["qtr"].strip() in {"1", "2", "3", "4"}):
                        quarters.append((code, int(row["qtr"]), row["disclosure_code"].strip().upper(),
                                         *(row[column] for column in months)))
    if not quarters:
        raise ValueError(f"No requested county records in {path.name}")
    frame = pd.DataFrame(quarters, columns=["county_fips", "quarter", "disclosure", *months])
    if frame.duplicated(["county_fips", "quarter"]).any():
        raise ValueError(f"Duplicate total-covered county quarters in {path.name}")
    rows = []
    for i, month in enumerate(months, start=1):
        values = pd.to_numeric(frame[month], errors="coerce")
        rows.append(pd.DataFrame({
            "county_fips": frame.county_fips,
            "date": pd.to_datetime({"year": year, "month": (frame.quarter - 1) * 3 + i, "day": 1}),
            "employment": values.where(frame.disclosure.ne("N") & values.gt(0)),
        }))
    return pd.concat(rows, ignore_index=True)


def harmonize(counties, crosswalk):
    """Return effective-date source assignments and CBSA totals on the 2018 footprint."""
    vintages = counties.pivot(index="date", columns="county_fips", values="employment").reindex(HISTORY_DATES)
    vintages = vintages.astype(float).where(lambda x: (x > 0) & np.isfinite(x))
    geographies = crosswalk.set_index("county_fips").cbsa_code
    if not geographies.index.is_unique:
        raise ValueError("County-to-CBSA mapping is ambiguous")
    harmonized, exceptions = [], []
    missing = pd.Series(np.nan, index=HISTORY_DATES)
    for target, cbsa in geographies.items():
        current = vintages.get(target, missing).copy()
        rule = pd.Series("direct", index=HISTORY_DATES)
        inputs = pd.Series(target, index=HISTORY_DATES)
        if target in ALIASES:
            old = ALIASES[target]
            previous = vintages.get(old, missing)
            duplicate = current.notna() & previous.notna()
            substitute = current.isna() & previous.notna()
            inputs.loc[substitute] = old
            inputs.loc[duplicate] = f"{target}+{old}"
            rule.loc[substitute] = "exact_code_alias"
            rule.loc[duplicate] = "ambiguous_alias"
            current = current.combine_first(previous).mask(duplicate)
        if target in MERGERS:
            old, change = MERGERS[target]
            previous = vintages.get(old, missing)
            before = HISTORY_DATES < pd.Timestamp(change)
            current.loc[before] = (current + previous).loc[before]
            inputs.loc[before] = f"{target}+{old}"
            rule.loc[before] = "exact_merger"
            ambiguous = ~before & previous.notna()
            rule.loc[ambiguous] = "ambiguous_merger"
            inputs.loc[ambiguous] = f"{target}+{old}"
            current.loc[ambiguous] = np.nan
        for affected, end, event in BOUNDARY_GAPS:
            if target in affected:
                before = HISTORY_DATES < pd.Timestamp(end)
                current.loc[before] = np.nan
                rule.loc[before] = event
        # The two units were in the same 2018 CBSA, so summing both restores the fixed footprint.
        if target in EXCHANGE:
            before = HISTORY_DATES < pd.Timestamp("2007-07-01")
            if all(fips in geographies for fips in EXCHANGE) and geographies[EXCHANGE[0]] == geographies[EXCHANGE[1]]:
                rule.loc[before] = "exchange_within_same_cbsa"
            else:
                current.loc[before] = np.nan
                rule.loc[before] = "cross_cbsa_exchange"
        rows = pd.DataFrame({"cbsa_code": cbsa, "county_fips_2018": target, "date": HISTORY_DATES,
                             "source_counties": inputs.to_numpy(), "rule": rule.to_numpy(),
                             "employment": current.to_numpy()})
        harmonized.append(rows)
        exceptions.append(rows[rows.rule.ne("direct") | rows.employment.isna()])

    county = pd.concat(harmonized, ignore_index=True)
    grouped = county.groupby(["cbsa_code", "date"]).employment
    panel = grouped.agg(employment="sum", counties_observed="count", counties_required="size").reset_index()
    panel["employment"] = panel.employment.where(panel.counties_observed.eq(panel.counties_required))
    return panel, pd.concat(exceptions, ignore_index=True)


def acquire_history(root, output=None):
    """Step 00. Download 2001–2009 files and build a companion historical panel."""
    from design_predictors import fetch

    root = Path(root)
    output = Path(output) if output else root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    geo = json.loads((root / GEO_MANIFEST).read_text(encoding="utf-8"))
    sources = {PANEL_FILE: "panel_sha256", CROSSWALK: "crosswalk_sha256", COUNTIES: "county_sha256"}
    if geo["geography_vintage"] != "2018-09" or any(
            sha(root / path) != geo[key] for path, key in sources.items()):
        raise ValueError("Run the existing Step 00 cells to restore a verified September 2018 base panel.")
    base = cbsa_csv(root / PANEL_FILE, parse_dates=["date"])
    crosswalk = cbsa_csv(root / CROSSWALK, usecols=["cbsa_code", "county_fips"])
    # Preserve FIPS leading zeroes even if read with integer inference.
    crosswalk["county_fips"] = crosswalk.county_fips.astype(str).str.zfill(5)
    crosswalk = crosswalk[crosswalk.cbsa_code.isin(base.cbsa_code)]
    if set(crosswalk.cbsa_code) != set(base.cbsa_code):
        raise ValueError("A base-panel CBSA is missing from the verified crosswalk")
    source_codes = set(crosswalk.county_fips)
    source_codes.update(old for new, old in ALIASES.items() if new in source_codes)
    source_codes.update(old for new, (old, _) in MERGERS.items() if new in source_codes)
    current = pd.read_csv(root / COUNTIES, dtype={"county_fips": "string"}, parse_dates=["date"],
                          usecols=["county_fips", "date", "employment"])
    current = current[current.county_fips.isin(source_codes) & current.date.le(HISTORY_END)]
    parts, hashes = [current], {}
    for year in range(2001, 2010):
        url = BLS_ARCHIVE.format(year=year)
        archive = root / "data/raw/qcew" / Path(url).name
        print(f"Reading QCEW county totals for {year}", flush=True)
        path = fetch(url, archive, "zip")
        parts.append(read_county_archive(path, year, source_codes))
        hashes[str(year)] = {"url": url, "sha256": sha(path)}

    panel, exceptions = harmonize(pd.concat(parts, ignore_index=True), crosswalk)
    overlap = panel.merge(base[["cbsa_code", "date", "employment"]].rename(columns={"employment": "existing_employment"}),
                          on=["cbsa_code", "date"], how="inner", validate="one_to_one")
    changed = overlap[~np.isclose(overlap.employment, overlap.existing_employment)]
    panel["year"] = panel.date.dt.year
    coverage = panel.groupby(["cbsa_code", "year"], as_index=False).employment.agg(
        months_required="size", months_observed="count")
    files = {PANEL_NAME: panel.drop(columns="year"), "county_geography_audit.csv": exceptions,
             "cbsa_year_coverage.csv": coverage, "overlap_changes_from_base.csv": changed}
    for filename, data in files.items():
        data.to_csv(output / filename, index=False)
    manifest = {
        "geography_vintage": "2018-09", "start": HISTORY_START, "end": HISTORY_END,
        "base_n_cbsa": base.cbsa_code.nunique(), "base_sources": {path: geo[key] for path, key in sources.items()},
        "bls_archives": hashes, "county_change_sources": [CENSUS_2000, CENSUS_2010],
        "files": {filename: sha(output / filename) for filename in files},
    }
    (output / "historical_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def score_path(actual, predicted, anchor):
    rmse, mae = accuracy(actual, predicted, anchor)
    c46, signed_c46 = target_values(actual[8:], predicted[8:])
    return {
        "normalized_rmse_pct": 100 * rmse, "normalized_mae_pct": 100 * mae,
        "signed_forecast_bias_pct": 100 * np.mean(predicted - actual) / anchor,
        "placebo_c46": c46, "signed_placebo_c46": signed_c46,
    }


def backtests(panel):
    """Step 01. Compare identical region/origin pairs on 54 later months."""
    series = panel.pivot(index="cbsa_code", columns="date", values="employment").reindex(columns=HISTORY_DATES)
    rows, exclusions = [], []
    for cbsa, values in series.iterrows():
        y = values.to_numpy(float)
        for period, origins, short_length in (
                ("recession_stress", RECESSION_ORIGINS, 60), ("post_recession", RECENT_ORIGINS, TRAIN_MONTHS)):
            for date in origins:
                origin = HISTORY_DATES.get_loc(date) + 1
                end = origin + OUTER_HORIZON
                if len(y) < end or origin <= short_length:
                    raise ValueError(f"Invalid backtest origin {date}")
                observed = y[:end]
                if not np.isfinite(observed).all() or (observed <= 0).any():
                    exclusions.append({"cbsa_code": cbsa, "origin": date, "period": period,
                                       "missing_training_months": int((~np.isfinite(y[:origin])).sum()),
                                       "missing_test_months": int((~np.isfinite(y[origin:end])).sum())})
                    continue
                history, truth = y[:origin], y[origin:end]
                anchor = history[-12:].mean()
                short, short_flags = guarded_paths(history[-short_length:], OUTER_HORIZON)
                long, long_flags = guarded_paths(history, OUTER_HORIZON)
                rules = [
                    (f"fixed_ensemble_{short_length}m", short[FIXED_ENSEMBLE], short_length,
                     FIXED_ENSEMBLE, short_flags),
                    ("fixed_ensemble_expanding", long[FIXED_ENSEMBLE], origin, FIXED_ENSEMBLE, long_flags),
                ]
                if period == "post_recession":
                    folds = tuple(origin - offset for offset in (78, 66, 54))
                    fits = {n: guarded_paths(history[:n], OUTER_HORIZON)[0] for n in folds}
                    fits[origin] = long
                    for horizon in (12, 54):
                        selected, _ = select_name(history, origin, folds, horizon, fits)
                        rules.append((f"nested_selected_{horizon}m", long[selected], origin, selected, long_flags))
                for rule, prediction, n_train, family, flags in rules:
                    rows.append({"cbsa_code": cbsa, "origin": date, "period": period, "rule": rule,
                                 "training_months": n_train, "family": family,
                                 "guardrail_rejections": "|".join(sorted(flags)),
                                 "test_start": HISTORY_DATES[origin], "test_end": HISTORY_DATES[end-1],
                                 "c46_start": HISTORY_DATES[origin+8],
                                 **score_path(truth, prediction, anchor)})
    return pd.DataFrame(rows), pd.DataFrame(exclusions, columns=[
        "cbsa_code", "origin", "period", "missing_training_months", "missing_test_months"])


def run_history_backtests(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / OUTPUT_DIR
    manifest_file = output / "historical_panel_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError("Run the historical acquisition cell in notebook 00 first.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if sha(output / PANEL_NAME) != manifest["files"][PANEL_NAME] or any(
            sha(root / path) != checksum for path, checksum in manifest["base_sources"].items()):
        raise ValueError("The historical or reference data changed. Rerun Step 00 historical acquisition.")
    results, exclusions = backtests(cbsa_csv(output / PANEL_NAME, parse_dates=["date"]))
    if results.empty:
        raise ValueError("No CBSA has sufficient complete history for matched backtests.")
    summary = results.groupby(["period", "origin", "rule"], as_index=False).agg(
        n_cbsa=("cbsa_code", "size"), mean_normalized_rmse_pct=("normalized_rmse_pct", "mean"),
        mean_normalized_mae_pct=("normalized_mae_pct", "mean"),
        mean_signed_bias_pct=("signed_forecast_bias_pct", "mean"),
        median_placebo_c46=("placebo_c46", "median"),
        median_signed_placebo_c46=("signed_placebo_c46", "median"),
        p90_placebo_c46=("placebo_c46", lambda x: x.quantile(.90)))
    matched = results.pivot(index=["period", "origin", "cbsa_code"], columns="rule", values="normalized_rmse_pct")
    paired = []
    for period, reference in (("recession_stress", "fixed_ensemble_60m"),
                              ("post_recession", "fixed_ensemble_114m")):
        pairs = matched.loc[matched.index.get_level_values("period") == period]
        if pairs.empty:
            continue
        for alternative in ("fixed_ensemble_expanding", *(("nested_selected_54m",) if period == "post_recession" else ())):
            baseline = reference if alternative == "fixed_ensemble_expanding" else "nested_selected_12m"
            diff = (pairs[alternative] - pairs[baseline]).rename("rmse_difference_pp").reset_index()
            paired.append(diff.assign(reference=baseline, alternative=alternative))
    differences = pd.concat(paired, ignore_index=True)
    paired_summary = differences.groupby(["period", "origin", "reference", "alternative"], as_index=False).agg(
        matched_cbsa=("rmse_difference_pp", "size"), mean_rmse_difference_pp=("rmse_difference_pp", "mean"),
        median_rmse_difference_pp=("rmse_difference_pp", "median"),
        share_lower_rmse=("rmse_difference_pp", lambda x: x.lt(0).mean()))
    files = {"historical_backtest_summary.csv": summary, "historical_backtest_detail.csv": results,
             "historical_paired_comparison.csv": paired_summary, "historical_backtest_exclusions.csv": exclusions}
    for name, data in files.items():
        data.to_csv(output / name, index=False)
    metadata = {"input_manifest_sha256": sha(manifest_file), "outer_horizon": OUTER_HORIZON,
                "last_observed_month": HISTORY_END, "placebo_leads": [9, 54],
                "files": {name: sha(output / name) for name in files}}
    (output / "historical_backtest_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary, paired_summary


def acquire_or_reuse_history(root, refresh=False):
    root = Path(root)
    output = root / OUTPUT_DIR
    manifest_path = output / "historical_panel_manifest.json"
    if manifest_path.is_file() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("geography_vintage") == "2018-09"
                and all((root / file).is_file() and sha(root / file) == digest
                        for file, digest in manifest["base_sources"].items())
                and all((output / file).is_file() and sha(output / file) == digest
                        for file, digest in manifest["files"].items())):
            return output
    return acquire_history(root)
