"""Long-horizon C46 forecast and geographic sensitivity used by notebooks 01 and 04."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (12, 24, 36, 48, 54)
ORIGINS = (54, 57, 60)  # June, September and December 2014
ORIGINS_36 = (66, 69, 72)  # June, September and December 2015
NAMES = ("seasonal_naive", "seasonal_drift", "log_seasonal_trend", "damped_log_seasonal_trend")
FIXED_ENSEMBLE = "fixed_ensemble"
RECENT_GROWTH = "qcew2018to2019q2_growth"
OLDER_GROWTH = "qcew2015to2018_annualized_log_growth"
PANEL_FILE = "data/processed/qcew_cbsa_monthly_2010_2023_omb2018.csv"
LEGACY_FILE = "data/processed/stage1_omb2018_forecasts_jul2019_dec2023.csv"
PREDICTOR_FILE = "data/processed/stage2_omb2018_pre2020_predictors.csv"
OMB_FILE = "data/raw/geography/list1_Sep_2018.xls"
DATES = pd.date_range("2010-01-01", "2023-12-01", freq="MS")
TRAIN_MONTHS = 114
FORECAST_DATES = DATES[TRAIN_MONTHS:]
OUTCOME_DATES = FORECAST_DATES[8:]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cbsa_csv(path, **kwargs):
    return pd.read_csv(path, dtype={"cbsa_code": "string"}, **kwargs)


def path_problem(history, forecast):
    """Check against same-season levels and year-over-year growth using pre-origin data."""
    history, forecast = np.asarray(history, float), np.asarray(forecast, float)
    if forecast.ndim != 1 or not len(forecast) or not np.isfinite(forecast).all() or (forecast <= 0).any():
        return "nonfinite, nonpositive or empty forecast"
    logs = np.log(history)
    # Analytic guardrails, chosen for this sensitivity. They are not probability bands.
    annual_limit = np.clip(np.quantile(abs(logs[12:] - logs[:-12]), .95),
                           np.log1p(.10), np.log1p(.25))
    levels = np.log(forecast)
    combined = np.r_[logs[-12:], levels]
    seasonal_anchor = np.resize(logs[-12:], len(levels))
    years = np.arange(len(levels)) // 12 + 1
    tolerance = 1e-10
    if (abs(levels - combined[:-12]) > annual_limit + tolerance).any():
        return "year-over-year growth exceeds the pre-origin limit"
    if (abs(levels - seasonal_anchor) > np.minimum(years * annual_limit, np.log(2)) + tolerance).any():
        return "cumulative change exceeds the seasonal-anchor limit"
    return None


def candidate_paths(history, h):
    """Three simple trend alternatives and a seasonal-naive benchmark."""
    history = np.asarray(history, float)
    baseline = np.resize(history[-12:], h)
    years = np.arange(h) // 12 + 1
    drift = (history[-12:].mean() - history[:12].mean()) / ((len(history) - 12) / 12)
    t = np.arange(len(history))
    season = np.eye(12)[t % 12]
    x = np.column_stack((t, season))
    logged = np.log(history)
    slope, *seasonal = np.linalg.lstsq(x, logged, rcond=None)[0]
    fitted = slope * t + np.asarray(seasonal)[t % 12]
    smearing = np.exp(logged - fitted).mean()
    future_t = np.arange(len(history), len(history) + h)
    trend = np.exp(slope * future_t + np.asarray(seasonal)[future_t % 12]) * smearing
    # Damp the departure from the last observed seasonal profile, with a 36-month e-folding time.
    damped = baseline + np.exp(-np.arange(1, h + 1) / 36) * (trend - baseline)
    return dict(zip(NAMES, (baseline, baseline + drift * years, trend, damped)))


def guarded_paths(history, h):
    raw = candidate_paths(history, h)
    rejected = {name: problem for name, path in raw.items()
                if (problem := path_problem(history, path)) is not None}
    allowed = {name: path for name, path in raw.items() if name not in rejected}
    # A rejected fixed member receives the seasonal-naive path at its original weight.
    members = (NAMES[0], NAMES[1], NAMES[3])
    ensemble = np.mean([allowed.get(name, raw[NAMES[0]]) for name in members], axis=0)
    problem = path_problem(history, ensemble)
    if problem:
        rejected[FIXED_ENSEMBLE] = problem
        ensemble = raw[NAMES[0]]
    return {**allowed, FIXED_ENSEMBLE: ensemble}, rejected


def target_values(actual, predicted):
    gap = 100 * (1 - actual / predicted)
    return float(np.maximum(gap, 0).sum()), float(gap.sum())


def accuracy(actual, pred, anchor):
    error = (pred - actual) / anchor
    return float(np.sqrt(np.mean(error ** 2))), float(np.mean(abs(error)))


def select_name(history, fitting_origin, folds, horizon, fit_paths):
    """Rank on common folds, then take the best-ranked available final path."""
    if max(folds) + horizon > fitting_origin:
        raise ValueError("Selection windows extend past the forecast origin")
    rankings = []
    for name in NAMES:
        if name not in fit_paths[fitting_origin] or any(name not in fit_paths[n] for n in folds):
            continue
        errors = [accuracy(history[n:n+horizon], fit_paths[n][name][:horizon], history[n-12:n].mean())
                  for n in folds]
        rmse, mae = np.mean(errors, axis=0)
        rankings.append((rmse, mae, name))
    return min(rankings)[2], rankings


def temporal_case(history):
    """For historical tests, each fit and inner selection stops at its own origin."""
    paths, rejected = {}, {}
    # Same inner origins for every policy compared at a given outer origin.
    inner = (24, 27, 30)
    for n in (*inner, *ORIGINS, *ORIGINS_36, TRAIN_MONTHS):
        # Fit at least as far as the largest lead evaluated at this origin.
        length = 36 if n in (*inner, *ORIGINS_36) else 54
        paths[n], rejected[n] = guarded_paths(history[:n], length)

    historical, scores, final = [], [], {}
    for n in (*ORIGINS, *ORIGINS_36):
        outer_horizon = 54 if n in ORIGINS else 36
        truth = history[n:n+outer_horizon]
        by_rule = {}
        for name in (*NAMES, FIXED_ENSEMBLE):
            by_rule[name] = (name, paths[n].get(name, paths[n][NAMES[0]]), name in rejected[n])
        selection_horizons = (12, 24) if outer_horizon == 54 else (12, 24, 36)
        for horizon in selection_horizons:
            selected, _ = select_name(history, n, inner, horizon, paths)
            by_rule[f"nested_selected_{horizon}"] = (selected, paths[n][selected], False)
        for rule, (family, prediction, fallback) in by_rule.items():
            positive, signed = (target_values(truth[8:], prediction[8:])
                                if outer_horizon == 54 else (np.nan, np.nan))
            recent_growth = history[n-3:n].mean() / history[n-15:n-12].mean() - 1
            projected_growth = (prediction[-9:-6].mean() / history[n-3:n].mean() - 1
                                if outer_horizon == 54 else np.nan)
            for horizon in (h for h in HORIZONS if h <= outer_horizon):
                rmse, mae = accuracy(truth[:horizon], prediction[:horizon], history[n-12:n].mean())
                historical.append((n, rule, family, horizon, rmse, mae, fallback,
                                   positive, signed, recent_growth, projected_growth))

    for horizon in HORIZONS:
        name, ranking = select_name(history, TRAIN_MONTHS, ORIGINS, horizon, paths)
        final[f"selected_{horizon}"] = (name, paths[TRAIN_MONTHS][name], False)
        scores.extend((horizon, family, rmse, mae, family == name) for rmse, mae, family in ranking)

    for name in (NAMES[0], NAMES[3], FIXED_ENSEMBLE):
        available = paths[TRAIN_MONTHS]
        final[name] = (name, available.get(name, available[NAMES[0]]), name in rejected[TRAIN_MONTHS])
    return final, historical, scores, rejected


def run_stage1(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "output/c46_forecast_validity"
    output.mkdir(parents=True, exist_ok=True)
    panel_path = root / PANEL_FILE
    panel_hash = sha(panel_path)
    geo = json.loads((root / "data/processed/omb2018_geography_manifest.json").read_text())
    if geo["geography_vintage"] != "2018-09" or geo["panel_sha256"] != panel_hash:
        raise ValueError("Employment panel differs from the September 2018 geography manifest")
    panel = cbsa_csv(panel_path, parse_dates=["date"])
    matrix = panel.pivot(index="cbsa_code", columns="date", values="employment").reindex(columns=DATES)
    if len(matrix) != geo["n_cbsa"] or not np.isfinite(matrix.to_numpy()).all() or (matrix <= 0).any().any():
        raise ValueError("Expected complete positive monthly employment for every CBSA")

    forecast_rows, history_rows, score_rows, flag_rows = [], [], [], []
    for cbsa, series in matrix.iterrows():
        final, historical, scores, rejected = temporal_case(series.to_numpy(float)[:TRAIN_MONTHS])
        history_rows.extend((cbsa, DATES[n-1], *row) for n, *row in historical)
        score_rows.extend((cbsa, *row) for row in scores)
        flag_rows.extend((cbsa, DATES[n-1], name, reason) for n, flags in rejected.items()
                         for name, reason in flags.items())
        for rule, (family, path, fallback) in final.items():
            forecast_rows.extend((cbsa, date, rule, family, value, fallback)
                                 for date, value in zip(FORECAST_DATES, path))

    files = {
        "forecasts.csv": (forecast_rows, ["cbsa_code", "date", "scenario", "family", "forecast", "naive_fallback"]),
        "historical_accuracy.csv": (history_rows, ["cbsa_code", "origin", "rule", "family", "horizon",
            "nrmse", "nmae", "naive_fallback", "placebo_c46", "placebo_signed46", "recent_growth", "projected_growth"]),
        "selection_scores.csv": (score_rows, ["cbsa_code", "selection_horizon", "family", "nrmse", "nmae", "selected"]),
        "forecast_flags.csv": (flag_rows, ["cbsa_code", "origin", "family", "reason"]),
    }
    for filename, (rows, fields) in files.items():
        pd.DataFrame(rows, columns=fields).to_csv(output / filename, index=False)
    historical = pd.read_csv(output / "historical_accuracy.csv")
    historical.groupby(["origin", "rule", "horizon"], as_index=False).agg(
        n=("nrmse", "size"), mean_nrmse=("nrmse", "mean"), mean_nmae=("nmae", "mean"),
        median_placebo_c46=("placebo_c46", "median"),
        p90_placebo_c46=("placebo_c46", lambda s: s.quantile(.90)),
        fallback_count=("naive_fallback", "sum"),
    ).to_csv(output / "historical_summary.csv", index=False)
    metadata = {"panel_sha256": panel_hash, "n_cbsa": len(matrix),
                "training_end": "2019-06", "forecast_start": "2019-07", "outcome_start": "2020-03",
                "horizons": HORIZONS, "reference_sensitivity": FIXED_ENSEMBLE,
                "files": {name: sha(output / name) for name in (*files, "historical_summary.csv")}}
    (output / "stage1_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def place_names(root):
    raw = pd.read_excel(root / OMB_FILE, header=2, usecols=["CBSA Code", "CBSA Title"])
    raw.columns = ["cbsa_code", "cbsa_name"]
    raw["cbsa_code"] = pd.to_numeric(raw.cbsa_code, errors="coerce").astype("Int64").astype("string").str.zfill(5)
    names = raw.dropna().drop_duplicates()
    if names.cbsa_code.duplicated().any():
        raise ValueError("Conflicting September 2018 OMB titles for a CBSA code")
    return names


def legacy_paths(root):
    path = root / LEGACY_FILE
    if not path.is_file():
        return None
    manifest = json.loads((root / "output/stage1_omb2018_manifest.json").read_text())
    if manifest["counterfactual_forecasts_sha256"] != sha(path):
        raise ValueError("Existing Stage 1 forecast differs from its manifest")
    source = cbsa_csv(path, parse_dates=["date"])
    columns = {
        "counterfactual_employment": "existing_stage1",
        "counterfactual_ensemble_employment": "existing_stage1_ensemble",
        "counterfactual_damped_employment": "existing_stage1_damped",
    }
    frames = []
    for column, scenario in columns.items():
        frame = source[["cbsa_code", "date", "selected_model", column]].rename(
            columns={"selected_model": "family", column: "forecast"})
        frames.append(frame.assign(scenario=scenario, naive_fallback=False))
    return pd.concat(frames, ignore_index=True)


def compute_outcomes(panel, forecasts):
    actual = panel[["date", "cbsa_code", "employment"]]
    paired = forecasts.merge(actual, on=["date", "cbsa_code"], validate="many_to_one")
    if len(paired) != len(forecasts) or paired.duplicated(["cbsa_code", "scenario", "date"]).any():
        raise ValueError("Forecasts and observations must pair uniquely for every CBSA-month")
    if not np.isfinite(paired.forecast).all() or (paired.forecast <= 0).any():
        raise ValueError("Every forecast must be finite and positive")
    keys = ["cbsa_code", "scenario", "family"]
    c46 = paired[paired.date.isin(OUTCOME_DATES)].copy()
    c46["signed_gap"] = 100 * (1 - c46.employment / c46.forecast)
    c46["shortfall"] = c46.signed_gap.clip(lower=0)
    results = c46.groupby(keys, as_index=False).agg(
        n=("shortfall", "size"), c46=("shortfall", "sum"), signed46=("signed_gap", "sum"),
        naive_fallback=("naive_fallback", "max"))
    expected = pd.MultiIndex.from_product([panel.cbsa_code.unique(), paired.scenario.unique()])
    actual_keys = pd.MultiIndex.from_frame(results[["cbsa_code", "scenario"]])
    if not results.n.eq(46).all() or not actual_keys.is_unique or not expected.difference(actual_keys).empty:
        raise ValueError("All forecast scenarios must cover the same CBSAs and 46 outcome months")
    anchor = panel.loc[panel.date.between("2019-04-01", "2019-06-01")].groupby("cbsa_code").employment.mean()
    endpoint = c46[c46.date.between("2023-04-01", "2023-06-01")].groupby(keys).agg(
        projected_endpoint=("forecast", "mean"), actual_endpoint=("employment", "mean")).reset_index()
    results = results.merge(endpoint, on=keys, validate="one_to_one")
    for name in ("projected", "actual"):
        results[name + "_growth"] = results[name + "_endpoint"] / results.cbsa_code.map(anchor) - 1
    results["mean_monthly_shortfall_pp"] = results.c46 / 46
    baseline = results.loc[results.scenario.eq(NAMES[0])].set_index("cbsa_code")
    for metric in ("c46", "signed46"):
        results[metric + "_minus_naive"] = results[metric] - results.cbsa_code.map(baseline[metric])
    return results


def geographic_checks(frame, features):
    from design_models import DesignPreprocessor, scores
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    no_growth = [name for name in features if name not in (RECENT_GROWTH, OLDER_GROWTH)]
    modes = {"no_growth": no_growth, "recent_growth": no_growth + [RECENT_GROWTH],
             "existing_primary": features, "primary_plus_recent": features + [RECENT_GROWTH]}
    rows, summary = [], []
    for scenario, data in frame.groupby("scenario"):
        data = data.reset_index(drop=True)
        for mode, columns in modes.items():
            for train, test in LeaveOneGroupOut().split(data, groups=data.geographic_group):
                tr, te = data.iloc[train], data.iloc[test]
                model = make_pipeline(DesignPreprocessor(), StandardScaler(), Ridge(alpha=10))
                model.fit(tr[columns], tr.c46)
                pred = np.maximum(0, model.predict(te[columns]))
                rows.append(te[["cbsa_code", "geographic_group", "c46"]].assign(
                    scenario=scenario, feature_mode=mode, predicted=pred,
                    baseline_mean=tr.c46.mean(), baseline_median=tr.c46.median()))
    predictions = pd.concat(rows, ignore_index=True)
    predictions["abs_error"] = abs(predictions.c46 - predictions.predicted)
    for (scenario, mode), data in predictions.groupby(["scenario", "feature_mode"]):
        summary.append({"scenario": scenario, "feature_mode": mode,
                        **scores(data.c46, data.predicted, data.baseline_mean, data.baseline_median)})
    return predictions, pd.DataFrame(summary)


def correlation_rows(frame, group_cols, growth, metrics):
    rows = []
    for group, data in frame.groupby(group_cols):
        group = group if isinstance(group, tuple) else (group,)
        for metric in metrics:
            paired = data[[growth, metric]].dropna()
            unique = paired.nunique().min()
            rows.append(dict(zip(group_cols, group), metric=metric, n=len(paired),
                             pearson=paired[growth].corr(paired[metric]) if unique > 1 else np.nan,
                             spearman=paired[growth].rank().corr(paired[metric].rank()) if unique > 1 else np.nan))
    return pd.DataFrame(rows)


def run_stage2(root, output=None):
    from design_models import feature_sets

    root = Path(root)
    output = Path(output) if output is not None else root / "output/c46_forecast_validity"
    manifest_path = output / "stage1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(root / PANEL_FILE) != manifest["panel_sha256"]:
        raise ValueError("Employment panel changed; rerun the long-horizon cell in notebook 01")
    for name, checksum in manifest["files"].items():
        if sha(output / name) != checksum:
            raise ValueError(f"Stage 1 sensitivity file changed: {name}")
    panel = cbsa_csv(root / PANEL_FILE, parse_dates=["date"])
    new = cbsa_csv(output / "forecasts.csv", parse_dates=["date"])
    existing = legacy_paths(root)
    forecasts = pd.concat([new, existing], ignore_index=True) if existing is not None else new
    results = compute_outcomes(panel, forecasts)
    inputs = cbsa_csv(root / PREDICTOR_FILE)
    features = feature_sets(inputs)["primary"]
    frame = results.merge(inputs, on="cbsa_code", how="left", validate="many_to_one")
    if frame.geographic_group.isna().any() or frame[RECENT_GROWTH].notna().sum() == 0:
        raise ValueError("Run notebook 02 to supply geographic groups and recent Q2 growth")
    names = place_names(root)
    frame = frame.merge(names, on="cbsa_code", how="left", validate="many_to_one")
    if frame.cbsa_name.isna().any():
        raise ValueError("A modeled CBSA is missing from the September 2018 OMB workbook")
    predictions, geo_scores = geographic_checks(frame, features)
    primary = predictions[predictions.feature_mode.eq("existing_primary")].copy()
    primary["residual_rank"] = primary.groupby("scenario").abs_error.rank(method="min", ascending=False).astype(int)
    audit = frame[["cbsa_code", "cbsa_name", "family", "scenario", "c46", "mean_monthly_shortfall_pp",
                   RECENT_GROWTH, "projected_growth", "actual_growth", "c46_minus_naive", "signed46_minus_naive",
                   "naive_fallback"]].merge(
        primary[["cbsa_code", "scenario", "predicted", "abs_error", "residual_rank"]],
        on=["cbsa_code", "scenario"], validate="one_to_one")
    audit["c46_rank"] = audit.groupby("scenario").c46.rank(method="min", ascending=False).astype(int)
    named = names[names.cbsa_name.str.match(r"^(Pecos,|Carlsbad-Artesia,|Williston,|Lake Charles,)")]
    if len(named) != 4:
        raise ValueError("Expected all four named places in September 2018 OMB file")
    named_audit = named.merge(audit.drop(columns="cbsa_name"), on="cbsa_code", how="left", validate="one_to_many")
    named_audit["in_analysis_sample"] = named_audit.scenario.notna()
    coverage = root / "output/omb2018_cbsa_sample_selection_audit.csv"
    if coverage.is_file():
        cols = ["cbsa_code", "pre_cutoff_complete_months", "pre_cutoff_required_months"]
        named_audit = named_audit.merge(cbsa_csv(coverage, usecols=cols), on="cbsa_code",
                                        how="left", validate="many_to_one")
    archived = root / "output/stage2_geographic04_largest_residuals.csv"
    prior = pd.DataFrame()
    if archived.is_file():
        prior = cbsa_csv(archived)
        prior = prior[prior.target.eq("c46")][["cbsa_code", "abs_error"]].copy()
        prior["archived_c46_residual_rank"] = prior.abs_error.rank(method="min", ascending=False).astype(int)
        named_audit = named_audit.merge(prior[["cbsa_code", "archived_c46_residual_rank"]],
                                        on="cbsa_code", how="left", validate="many_to_one")

    family = audit.groupby(["scenario", "family"], as_index=False).agg(
        n=("c46", "size"), mean_c46=("c46", "mean"), median_c46=("c46", "median"),
        p90_c46=("c46", lambda s: s.quantile(.90)), max_c46=("c46", "max"),
        median_projected_growth=("projected_growth", "median"), naive_fallbacks=("naive_fallback", "sum"))
    scenarios = audit.groupby("scenario", as_index=False).agg(
        n=("c46", "size"), median_c46=("c46", "median"), p90_c46=("c46", lambda s: s.quantile(.90)),
        max_c46=("c46", "max"), max_mean_monthly_shortfall_pp=("mean_monthly_shortfall_pp", "max"))
    history = cbsa_csv(output / "historical_accuracy.csv")
    placebo = correlation_rows(history[history.horizon.eq(54)], ["origin", "rule"], "recent_growth",
                               ["projected_growth", "placebo_c46", "placebo_signed46"])
    growth = correlation_rows(frame, ["scenario"], RECENT_GROWTH,
                              ["projected_growth", "actual_growth", "c46", "signed46",
                               "c46_minus_naive", "signed46_minus_naive"])
    files = {
        "c46_by_cbsa.csv": audit, "c46_by_model_family.csv": family, "c46_by_scenario.csv": scenarios,
        "c46_extremes.csv": audit[(audit.c46_rank.le(20)) | (audit.residual_rank.le(20))],
        "named_cbsa_verification.csv": named_audit, "growth_mechanical_association.csv": growth,
        "historical_growth_placebo.csv": placebo, "geographic_predictions.csv": predictions,
        "geographic_growth_ablation.csv": geo_scores,
    }
    if not prior.empty:
        files["archived_c46_residual_codes.csv"] = prior.merge(names, on="cbsa_code",
                                                               how="left", validate="one_to_one")
    for filename, table in files.items():
        table.to_csv(output / filename, index=False)
    metadata = {"stage1_manifest_sha256": sha(manifest_path), "predictor_sha256": sha(root / PREDICTOR_FILE),
                "omb_sha256": sha(root / OMB_FILE), "included_existing_forecasts": existing is not None,
                "existing_forecast_sha256": sha(root / LEGACY_FILE) if existing is not None else None,
                "geographic_model": "Ridge alpha=10, original-scale C46, leave-one-division-out",
                "files": {name: sha(output / name) for name in files}}
    (output / "stage2_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage1", "stage2", "both"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage in ("stage1", "both"):
        print(run_stage1(args.root, args.output), flush=True)
    if args.stage in ("stage2", "both"):
        print(run_stage2(args.root, args.output), flush=True)
