"""Nested 54-month selection of already guarded Stage 1 forecast specifications."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from c46_forecast_validity import FIXED_ENSEMBLE, NAMES, FORECAST_DATES, guarded_paths, sha, cbsa_csv, target_values
from qcew_historical_validation import HISTORY_DATES, HISTORY_END, OUTPUT_DIR, PANEL_NAME, RECENT_ORIGINS

CANDIDATES = (FIXED_ENSEMBLE, *NAMES)
RULES = ("fixed_ensemble", "pooled", "partially_pooled", "per_cbsa")
HORIZON = 54
FIRST_LEAD = 9
WINDOW = 114
MIN_TRAIN = 48
INNER_OFFSETS = (78, 66, 54)
OVERRIDE_MARGIN = 0.02


def paths_at(values, cutoff):
    """Fit only the most recent uninterrupted sample at or before cutoff."""
    history = np.asarray(values[max(0, cutoff - WINDOW):cutoff], float)
    unavailable = np.flatnonzero(~np.isfinite(history) | (history <= 0))
    if len(unavailable):
        history = history[unavailable[-1] + 1:]
    if len(history) < MIN_TRAIN:
        return None
    paths, rejected = guarded_paths(history, HORIZON)
    # Every candidate has a scored operational path, including if its guard fails.
    return (np.stack([paths.get(name, paths[NAMES[0]]) for name in CANDIDATES]),
            np.array([name in rejected for name in CANDIDATES]), history[-12:].mean())


def scaled_errors(paths, truth, scale):
    error = 100 * (paths - truth) / scale
    return np.sqrt(np.mean(error[..., FIRST_LEAD - 1:] ** 2, axis=-1))


def select_rules(scores):
    """scores has dimensions CBSA, inner fold and candidate, on a common sample."""
    pooled = int(scores.mean(axis=(0, 1)).argmin())
    local = scores.mean(axis=1).argmin(axis=1)
    index = np.arange(len(local))
    better = scores[index, :, local] < scores[:, :, pooled]
    local_mean = scores.mean(axis=1)[index, local]
    pooled_mean = scores[:, :, pooled].mean(axis=1)
    override = (local_mean < (1 - OVERRIDE_MARGIN) * pooled_mean) & (better.sum(axis=1) >= 2)
    return np.column_stack([np.zeros(len(local), dtype=int), np.full(len(local), pooled),
                            np.where(override, local, pooled), local])


def _index(date):
    return HISTORY_DATES.get_loc(pd.Timestamp(date)) + 1


def inner_scores(matrix, origin, cached):
    eligible, losses = [], []
    for cbsa, row in enumerate(matrix):
        candidate_fold_losses = []
        for offset in INNER_OFFSETS:
            split = origin - offset
            if split + HORIZON > origin:
                raise ValueError("Inner evaluation overlaps outer forecast")
            truth = row[split:split + HORIZON]
            forecast = cached(cbsa, split)
            if forecast is None or len(truth) != HORIZON or not (np.isfinite(truth) & (truth > 0)).all():
                break
            paths, _, scale = forecast
            candidate_fold_losses.append(scaled_errors(paths, truth, scale))
        if len(candidate_fold_losses) == len(INNER_OFFSETS):
            eligible.append(cbsa)
            losses.append(candidate_fold_losses)
    return eligible, np.asarray(losses)


def learn(history, deployment, output):
    """Compare forecasting policies on paired hindcasts and apply a declared promotion gate."""
    matrix = history.to_numpy(float)
    production = deployment.to_numpy(float)
    if not history.index.equals(deployment.index):
        raise ValueError("Historical and deployment CBSA order differs")
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def cached(cbsa, cutoff):
        return paths_at(matrix[cbsa], cutoff)

    rows = []
    for date in RECENT_ORIGINS:
        origin = _index(date)
        ids, scores = inner_scores(matrix, origin, cached)
        if not ids:
            raise ValueError(f"No inner folds with complete data at {date}")
        decisions = select_rules(scores)
        for cbsa, choices in zip(ids, decisions):
            truth = matrix[cbsa, origin:origin + HORIZON]
            forecast = cached(cbsa, origin)
            if forecast is None or len(truth) != HORIZON or not (np.isfinite(truth) & (truth > 0)).all():
                continue
            paths, flags, scale = forecast
            for rule, choice in zip(RULES, choices):
                path = paths[choice]
                signed_error = 100 * (path - truth) / scale
                c46, signed_c46 = target_values(truth[FIRST_LEAD - 1:], path[FIRST_LEAD - 1:])
                rows.append({
                    "origin": date, "cbsa_code": history.index[cbsa], "rule": rule,
                    "candidate": CANDIDATES[choice], "candidate_guard_fallback": bool(flags[choice]),
                    "rmse_1_54_pct": np.sqrt(np.mean(signed_error ** 2)),
                    "rmse_9_54_pct": np.sqrt(np.mean(signed_error[FIRST_LEAD - 1:] ** 2)),
                    "signed_bias_9_54_pct": signed_error[FIRST_LEAD - 1:].mean(),
                    "placebo_c46": c46, "signed_placebo_c46": signed_c46,
                })
    detail = pd.DataFrame(rows)
    if detail.empty or detail.origin.nunique() != len(RECENT_ORIGINS):
        raise ValueError("An outer forecast origin lacks held-out observations")
    summary = detail.groupby(["origin", "rule"], as_index=False).agg(
        n_cbsa=("cbsa_code", "size"), rmse_1_54_pct=("rmse_1_54_pct", "mean"),
        rmse_9_54_pct=("rmse_9_54_pct", "mean"), bias_9_54_pct=("signed_bias_9_54_pct", "mean"),
        median_placebo_c46=("placebo_c46", "median"), signed_placebo_c46=("signed_placebo_c46", "median"),
        p90_placebo_c46=("placebo_c46", lambda x: x.quantile(.9)))
    fixed = summary[summary.rule.eq(RULES[0])].set_index("origin")
    passed, gate = [], []
    for rule in RULES[1:]:
        candidate = summary[summary.rule.eq(rule)].set_index("origin").reindex(fixed.index)
        rmse_ok = bool(candidate.rmse_9_54_pct.lt(fixed.rmse_9_54_pct).all())
        tail_ok = bool(candidate.p90_placebo_c46.le(fixed.p90_placebo_c46 + 1e-9).all())
        gate.append({"rule": rule, "lower_c46_window_rmse_every_origin": rmse_ok,
                     "no_higher_p90_placebo_every_origin": tail_ok,
                     "mean_rmse_9_54_pct": candidate.rmse_9_54_pct.mean()})
        if rmse_ok and tail_ok:
            passed.append((candidate.rmse_9_54_pct.mean(), RULES.index(rule)))
    chosen_rule = min(passed)[1] if passed else 0

    origin = _index(HISTORY_END)
    ids, scores = inner_scores(matrix, origin, cached)
    if not ids:
        raise ValueError("No complete inner folds for the final forecasts")
    decisions = select_rules(scores)
    choices = dict(zip(ids, decisions[:, chosen_rule]))
    selected_rows, paths = [], []
    for cbsa, y in enumerate(production):
        forecasts, flags, _ = paths_at(y, len(y))
        choice = int(choices.get(cbsa, 0))
        paths.append(forecasts[choice])
        selected_rows.append({"cbsa_code": history.index[cbsa], "candidate": CANDIDATES[choice],
                              "candidate_guard_fallback": bool(flags[choice]), "complete_inner_folds": cbsa in choices,
                              "policy": RULES[chosen_rule]})
    selected = pd.DataFrame(selected_rows)
    forecasts = pd.DataFrame({
        "cbsa_code": np.repeat(history.index.to_numpy(), HORIZON),
        "date": np.tile(FORECAST_DATES.to_numpy(), len(history)),
        "forecast": np.concatenate(paths),
    })
    detail.to_csv(output / "outer_detail.csv", index=False)
    summary.to_csv(output / "outer_summary.csv", index=False)
    pd.DataFrame(gate).to_csv(output / "promotion_checks.csv", index=False)
    selected.to_csv(output / "deployment_assignments.csv", index=False)
    forecasts.to_csv(output / "deployment_forecasts.csv", index=False)
    cached.cache_clear()
    return forecasts, selected, RULES[chosen_rule]


def publish_competition(root, training):
    root = Path(root)
    hist = root / OUTPUT_DIR
    manifest_path = hist / "historical_panel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel_path = hist / PANEL_NAME
    if sha(panel_path) != manifest["files"][PANEL_NAME] or any(
            sha(root / relative) != checksum for relative, checksum in manifest["base_sources"].items()):
        raise ValueError("Historical data or its inputs changed; rerun Step 00.")
    panel = cbsa_csv(panel_path, parse_dates=["date"])
    matrix = panel.pivot(index="cbsa_code", columns="date", values="employment").reindex(
        index=training.index, columns=HISTORY_DATES)
    if set(panel.cbsa_code) != set(training.index):
        raise ValueError("Historical and primary CBSA samples differ")
    output = root / "output/stage1_nested_selection"
    output.mkdir(parents=True, exist_ok=True)
    forecasts, selected, rule = learn(matrix, training, output)
    meta = {"historical_panel_sha256": sha(panel_path), "rules": RULES, "candidates": CANDIDATES,
            "selection_error_leads": [FIRST_LEAD, HORIZON], "outer_origins": [d.date().isoformat() for d in RECENT_ORIGINS],
            "inner_offsets_months": INNER_OFFSETS, "local_override_min_improvement_fraction": OVERRIDE_MARGIN,
            "primary_policy": rule, "assignment_counts": selected.candidate.value_counts().to_dict(),
            "deployment_forecast_sha256": sha(output / "deployment_forecasts.csv")}
    (output / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return forecasts, selected, meta
