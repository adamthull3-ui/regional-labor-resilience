# Priority methodological follow-up for the practicum

## 1. Extend the implemented historical stress analysis to calibrated forecast uncertainty — **still open**

The Stage 2 dependent variable is itself constructed using estimated potential employment. The counterfactual `Y0_hat` is not observed. The revised code already **implements** three 54-month historical pseudo-origin forecast-error stress paths chosen using pre-origin-only baseline tuning, exports corresponding alternative C46 targets and, with `RL_STAGE2_STRESS_EVAL=1`, **repeats the entire nested geographic prediction experiment for each target**. This is a meaningful robustness check, but it is **not calibrated forecast uncertainty**: the three origins overlap, do not represent the complex 2019-selected model's full error distribution, and scenario multipliers are capped. To go further, implement the following in a separate, prespecified experiment:

1. **Extend the already-saved historical forecast paths beyond the 3 baseline-selected trajectories.** `stage1_omb2018_historical_backtest_monthly.csv` stores 54-month log actual/forecast ratios using pre-origin-only simple-model selection. To calibrate uncertainty specific to the final forecasting rule, extend the pseudo-origin experiment to the whole prespecified ETS/SARIMA selection process, with all candidate choice completed before each historical test block; avoid the optional `RL_STAGE1_RETRO_SPEC=1` evaluation for calibration because it selects the 2019 winner using later pre-pandemic data.
2. **Retain temporal and geographic dependence.** Resample coherent multi-month forecast-error paths, preferably with hierarchical pooling across comparable groups/selected families for CBSAs with limited historical origins. A 54-month joint path is more defensible than 54 independent monthly draws. Distinguish forecast innovations, parameter error and between-specification uncertainty, and state which sources the procedure actually captures.
3. **Generate candidate counterfactuals:** `Y0_i,t^(b) = Y0_hat_i,t * exp(e_i,t^(b))` using sampled log errors, or use predictive distributions directly from the selected state-space model if calibration checks show reliable intervals. Do not mistake historical realized error from the real world for a purely sampling-based confidence interval. Enforce strictly positive, physically plausible trajectories and record any rejections.
4. **Recompute** the gap and positive shortfall for each path: `G_i,t^(b) = 100 * (Y_i,t / Y0_i,t^(b) - 1)` and `C46_i^(b) = sum_{t=Mar2020}^{Dec2023} max(-G_i,t^(b),0)`. Recompute resistance, persistence, recovery and shorter horizons consistently.
5. **Extend the 3-path Stage 2 evaluation to a calibrated simulation distribution:** the existing `RL_STAGE2_STRESS_EVAL=1` already repeats model selection and outer-fold testing for each scenario, with division holdouts preserved. For formal interval interpretation, repeat it over a substantially larger, validated set of coherent forecast paths and summarize the *distribution* of held-out skill relative to each draw's training-median benchmark, selected models, and CBSA rankings. Do not use outer-fold scores for model selection.
6. **Validate the uncertainty procedure** on the known 2019 pre-pandemic window and other fully pre-2019 12–48 month historical holdouts. Assess empirical coverage of forecast bands and calibration. With limited independent regions and few historical origins, present intervals as sensitivity-based evidence rather than definitive confidence intervals.

This design quantifies how much apparent 46-month unpredictability arises from the generated target. It does not by itself identify the causal effect of the pandemic or remove structural-break uncertainty.

## 2. Address long-horizon model selection

- The optional **54-month** historical pseudo-origin backtest in notebook 01 is diagnostic, **not a change to the main Stage 1 winner-selection procedure**. Its simple baseline choice is made before the historical evaluation period; record its 12/24/36/48/54-month errors alongside original 12-month selection scores. The optional retrospective 2019-selected-specification comparison is not an unbiased historical selection test.
- **Implemented optional longer-horizon model-selection experiments:** set `RL_STAGE1_SELECTION_HORIZON=24`, `36`, or `48` and `RL_ALLOW_ALTERNATE_SELECTION=1` in a **separate project root**. These select among the full Stage 1 candidate grid using three training-period rolling origins whose evaluation windows all end by June 2019. Run the entire Stage 1–Stage 2 pipeline there, then compare family frequencies, forecast quality and C46 outcomes against the prespecified 12-month primary analysis. Do **not** pick a horizon using outer geographic results.
- Consider an externally calibrated ensemble of seasonal-naive, ETS, log-trend and SARIMA candidates, with weights learned only from training-origin historical errors. The supplied 50/50 ensemble is a transparent baseline, not optimized or guaranteed to improve forecasts.
- Compare each candidate under `RL_LOG_BIAS_ADJUST=0` and `1`, with bounded forecast plausibility rules held fixed. Record fit failures (especially SARIMA nonconvergence or implausible 54-month growth).

## 3. Sample selection audit and naming check

- After executing notebooks 00 and 02, inspect `output/omb2018_cbsa_sample_selection_audit_with_division.csv` and `..._by_division.csv`. The audit division is the county-count majority; the modeling division uses 2019 Q2 county employment, so multi-state CBSA assignments may differ. You should report the distinction.
- Summarize how excluded CBSAs differ in pre-2020 employment, metro/micro type and geographic distribution. If the number remains two, report two; do not assert the number before your rerun.
- Match any largest C46 outlier code against the **September 2018** OMB crosswalk and confirm whether its historical forecast errors reflect energy-cycle shifts, hurricanes or other plausible shocks. These are explanatory hypotheses, not identified effects. The place/code examples in `README.md` are not proof those CBSAs are still the largest residuals.

## 4. Sensitivities to report without increasing the model library

- Hold the outer geographic folds fixed and rerun without `qcew2018to2019q2_growth`, to test mechanically induced correlation with Stage 1 trend extrapolation. Contrast the original and damped/ensemble counterfactuals.
- Compare CBSA-weighted and 2019 Q2 employment-weighted model performance. The latter is a **test loss sensitivity**; it is *not* a new employment-weighted tuning strategy unless you explicitly change the fitting/selection objective using training-only weights.
- Report fold-level R² against the held-out mean and, separately, OOS R² relative to the training mean; emphasize the latter as a fairer no-feature benchmark. Retain the median MAE comparator.
- Compare the training median skill at 12, 24 and 46 months; avoid selecting the more favorable target after examining results. The 46-month target should remain the primary outcome if prespecified.
- If Lasso or Elastic Net coefficients are all zero, say they collapsed to a constant prediction and compare against the training median/mean directly.
- Treat the number of geographic divisions as the independent unit for bootstrap sensitivity; do not use the large number of CBSAs to justify tight division-cluster confidence bounds.

## 5. Reporting language

Suggested conclusion when results remain weak: *“Pre-pandemic regional characteristics offer limited reliable out-of-division predictive skill for cumulative employment deficits through December 2023. Results for nearer-term windows are compared as prespecified sensitivities. Interpretation is further limited by the long-horizon uncertainty and specification dependence of the counterfactual employment trajectories.”*

This is appropriate only if the rerun supports that conclusion. The reviewer’s previously cited `c46 R² ≈ 0.004` and `skill vs median ≈ 2.4%` are **historical reported values**, not results produced by these corrected notebooks.


The deliverable is organized for **individual notebook execution in VS Code**, without a launcher. See `docs/FULL_RUN_GUIDE.md`.
