# Static-only review log

**Verification limit:** At the user's instruction, no tests and no notebook cells were executed. Only file inspection, notebook JSON handling, targeted edits, and Python syntax parsing of code-cell source were performed.

## Updated in this package

- Notebook 00: no change to validated acquisition and sample-audit algorithm; documented full-run assumptions.
- Notebook 01: forces all eligible CBSAs, all ETS/SARIMA standard candidates, 12-month primary selection, 54-month historical pre-pandemic backtests, log bias adjustment, and the retrospective model-family check. Historical stress analysis now has a clearly labeled fallback (`flat_last12_fallback`) if none of its prespecified benchmark candidates produces an admissible 54-month path; a record is retained in the historical audit.
- Notebook 02: retains corrected structural-zero vs suppression and key-safe retrieval; documented Census key prerequisite.
- Notebook 03: continues to refuse partial Stage 1 provenance. Individual **exploratory** model fit errors are now recorded in `stage2_omb2018_exploratory_fit_failures.csv` rather than silently hiding them or preventing the primary Stage 2 target files. Outcome and provenance checks remain strict.
- Notebook 04: forces all Census division outer folds, all base and historical stress outcomes, HGB, signed-outcome sensitivity, and 1,000 division-bootstrap resamples; adds training-fold-only removal of constant features and `stage2_geographic04_candidate_fit_failures.csv` for failed optional candidate models. The median-only benchmark remains available and every regional prediction is still required for a completed run.

## What was not established

- No BLS, Census, network, or API requests were made.
- No regressions, forecasts, bootstrap replications, synthetic scripts, or notebook cells were run.
- Runtime results (sample coverage, numerical errors, convergence, installed dependency compatibility) cannot be verified from static inspection alone.
- All methodological limitations in the prior `NEXT_STEPS.md` continue to apply.


## VS Code manual-run layout update

- The PowerShell launcher was **removed** at the user’s request.
- The five `.ipynb` files are in `code/`; the six synthetic `.py` scripts are in `tests/`.
- Test notebook paths are resolved relative to each test script (`../code/`) rather than the shell working directory.
- No tests or notebook cells were executed for this packaging change; test scripts were syntax-parsed only.
