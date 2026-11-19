"""Publish and verify the fixed-ensemble counterfactual used by Stages 1 and 2."""
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from c46_forecast_validity import (
    DATES, FIXED_ENSEMBLE, FORECAST_DATES, LEGACY_FILE, NAMES, PANEL_FILE,
    TRAIN_MONTHS, cbsa_csv, guarded_paths, path_problem, sha,
)

DESIGN = "nested_54month_competition_v1"
MEMBERS = (NAMES[0], NAMES[1], NAMES[3])
MANIFEST = "output/stage1_omb2018_manifest.json"
GAPS = "data/processed/stage1_omb2018_observed_vs_counterfactual_2020_2023.csv"
MODEL_FILE = "output/stage1_omb2018_selected_models.csv"
STAGE2 = "output/stage2_omb2018_modeling_manifest.json"
STAGE2_OUTCOMES = "data/processed/stage2_omb2018_resilience_outcomes.csv"


def load_primary_inputs(root, diagnostic_dir):
    root, diagnostic_dir = Path(root), Path(diagnostic_dir)
    source = diagnostic_dir / "forecasts.csv"
    checks = json.loads((diagnostic_dir / "stage1_manifest.json").read_text(encoding="utf-8"))
    panel_path = root / PANEL_FILE
    if checks["files"]["forecasts.csv"] != sha(source) or checks["panel_sha256"] != sha(panel_path):
        raise ValueError("Forecast diagnostics changed. Rerun notebook 01 long-horizon validation.")
    panel = cbsa_csv(panel_path, parse_dates=["date"])
    matrix = panel.pivot(index="cbsa_code", columns="date", values="employment").reindex(columns=DATES)
    if len(matrix) != checks["n_cbsa"] or not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("Incomplete CBSA panel")
    diagnostics = cbsa_csv(source, parse_dates=["date"])
    index = pd.MultiIndex.from_product([matrix.index, FORECAST_DATES], names=["cbsa_code", "date"])
    wide = diagnostics.pivot(index=["cbsa_code", "date"], columns="scenario", values="forecast").reindex(index)
    columns = [FIXED_ENSEMBLE, NAMES[0], NAMES[3], "selected_54"]
    if not np.isfinite(wide[columns]).all().all() or (wide[columns] <= 0).any().any():
        raise ValueError("Forecast paths must be complete and positive for all CBSAs")

    forecast = pd.DataFrame({
        "counterfactual_employment": wide[FIXED_ENSEMBLE],
        # In this sensitivity, blend the restricted 54-month winner with seasonal naive.
        "counterfactual_ensemble_employment": 0.5 * wide["selected_54"] + 0.5 * wide[NAMES[0]],
        "counterfactual_damped_employment": wide[NAMES[3]],
        "selected_model": FIXED_ENSEMBLE,
        "selected_specification": DESIGN,
    })
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    common, short, model_rows = [], [], []
    for cbsa, row in matrix.iterrows():
        history = row.to_numpy(float)[:TRAIN_MONTHS]
        paths, flags = guarded_paths(history, len(FORECAST_DATES))
        actual = forecast.loc[cbsa, "counterfactual_employment"].to_numpy()
        if not np.allclose(actual, paths[FIXED_ENSEMBLE], rtol=1e-12, atol=0):
            raise ValueError(f"Published forecast differs from the fixed ensemble for CBSA {cbsa}")
        short.append(guarded_paths(history[-78:], len(FORECAST_DATES))[0][FIXED_ENSEMBLE])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                ets = ExponentialSmoothing(history, trend="add", damped_trend=True,
                                           seasonal="add", seasonal_periods=12).fit(use_brute=False)
            alternative = np.asarray(ets.forecast(len(FORECAST_DATES)))
            reason = path_problem(history, alternative)
            if reason:
                alternative = np.full(len(FORECAST_DATES), np.nan)
        except (ValueError, RuntimeError, FloatingPointError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            alternative = np.full(len(FORECAST_DATES), np.nan)
        common.append(alternative)
        model_rows.append({
            "cbsa_code": cbsa, "model": FIXED_ENSEMBLE, "specification": DESIGN,
            "rejected_members": "|".join(member for member in MEMBERS if member in flags),
            "full_naive_fallback": FIXED_ENSEMBLE in flags, "common_ets_failure": reason,
        })
    forecast["counterfactual_common_ets_employment"] = np.concatenate(common)
    forecast["counterfactual_short_window_employment"] = np.concatenate(short)
    training = matrix.iloc[:, :TRAIN_MONTHS]
    from stage1_nested_selection import publish_competition

    chosen_forecasts, assignments, competition = publish_competition(root, training)
    chosen_series = chosen_forecasts.set_index(["cbsa_code", "date"]).forecast.reindex(forecast.index)
    if not np.isfinite(chosen_series).all() or (chosen_series <= 0).any():
        raise ValueError("Selected forecast path is incomplete")
    old_reference = forecast["counterfactual_employment"].copy()
    forecast["counterfactual_employment"] = chosen_series
    if competition["primary_policy"] != FIXED_ENSEMBLE:
        current_sensitivity = forecast["counterfactual_ensemble_employment"].copy()
        forecast["counterfactual_ensemble_employment"] = old_reference
        forecast["counterfactual_fixed_ensemble_employment"] = old_reference
        forecast["counterfactual_selected54_blend_employment"] = current_sensitivity
    assigned = assignments.set_index("cbsa_code")
    forecast["selected_model"] = forecast.index.get_level_values("cbsa_code").map(assigned.candidate)
    audit = pd.DataFrame(model_rows).drop(columns=["model"]).merge(assignments, on="cbsa_code", validate="one_to_one")
    audit = audit.rename(columns={"candidate": "model"})
    return panel, training, forecast.reset_index(), audit


def finish_manifest(root, diagnostic_dir, n_cbsa, stress_origins, stress_monthly, stress_outcomes, backtest_path):
    root, diagnostic_dir = Path(root), Path(diagnostic_dir)
    forecast = root / LEGACY_FILE
    observed = root / GAPS
    competition_path = root / "output/stage1_nested_selection/manifest.json"
    competition = json.loads(competition_path.read_text(encoding="utf-8"))
    policy = competition["primary_policy"]
    manifest = {
        "design_schema": "released_by_20200229_v2",
        "counterfactual_design": DESIGN,
        "geography_vintage": "2018-09",
        "input_panel_sha256": sha(root / PANEL_FILE),
        "information_cutoff": "2020-02-29",
        "model_training_cutoff": "2019-06-01",
        "training_window_months": TRAIN_MONTHS,
        "forecast_window": "2019-07 through 2023-12",
        "outcome_window": "2020-03 through 2023-12",
        "forecast_horizon_months": len(FORECAST_DATES),
        "primary_policy": policy,
        "per_cbsa_specification_selection": policy in ("partially_pooled", "per_cbsa"),
        "selection_metric": "normalized RMSE, forecast leads 9 through 54",
        "validation_horizon_months": 54,
        "candidate_specification_count": len(competition["candidates"]),
        "competition_manifest_sha256": sha(competition_path),
        "historical_panel_sha256": competition["historical_panel_sha256"],
        "primary_assignment_counts": competition["assignment_counts"],
        "fixed_member_count": len(MEMBERS),
        "fixed_member_weights": {member: 1 / len(MEMBERS) for member in MEMBERS},
        "guardrail_fallback": "Replace rejected member with seasonal naive at the same weight",
        "diagnostic_comparison_horizons": [12, 24, 36, 48, 54],
        "diagnostic_manifest_sha256": sha(diagnostic_dir / "stage1_manifest.json"),
        "long_horizon_backtest_completed": True,
        "historical_stress_paths_completed": True,
        "historical_stress_origins": [x.strftime("%Y_%m") for x in stress_origins],
        "historical_stress_monthly_sha256": sha(stress_monthly),
        "historical_stress_outcomes_sha256": sha(stress_outcomes),
        "long_horizon_backtest_sha256": sha(backtest_path),
        "counterfactual_sensitivities": ["ensemble", "damped", "common_ets", "short_window"],
        "counterfactual_sensitivity_definitions": {
            "ensemble": ("Equal blend of the restricted 54-month-selection sensitivity and seasonal naive"
                         if policy == FIXED_ENSEMBLE else "Fixed equal-weight ensemble benchmark"),
            "damped": "Damped log seasonal trend with seasonal-naive guardrail fallback",
            "common_ets": "Fixed damped additive Holt-Winters, missing on rejection",
            "short_window": "Primary fixed ensemble using the last 78 training months",
        },
        "cbsa_count": n_cbsa,
        "all_regions_included": True,
        "selected_models_sha256": sha(root / MODEL_FILE),
        "outcomes_sha256": sha(observed),
        "counterfactual_forecasts_sha256": sha(forecast),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = root / MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def require_primary(root, require_stage2=False):
    root = Path(root)
    path = root / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("counterfactual_design") != DESIGN
            or manifest.get("competition_manifest_sha256") != sha(root / "output/stage1_nested_selection/manifest.json")
            or manifest.get("input_panel_sha256") != sha(root / PANEL_FILE)
            or manifest.get("counterfactual_forecasts_sha256") != sha(root / LEGACY_FILE)
            or manifest.get("outcomes_sha256") != sha(root / GAPS)):
        raise ValueError("Run revised Notebook 01 to publish the nested-selection primary forecast.")
    if require_stage2:
        stage2 = json.loads((root / STAGE2).read_text(encoding="utf-8"))
        if (stage2.get("stage1_manifest_sha256") != sha(path)
                or stage2.get("stage1_counterfactual_design") != DESIGN
                or stage2.get("resilience_outcomes_sha256") != sha(root / STAGE2_OUTCOMES)):
            raise ValueError("Run revised Notebook 03 to rebuild outcomes from the current primary forecast.")
    return manifest
