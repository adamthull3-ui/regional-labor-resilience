# Manual full-run guide (VS Code — no launcher)

**Execution status:** These files are configured and reorganized; no notebook or test script has been run for this request. The original full-run settings have not been changed. Runtime performance, downloads, and full empirical results remain unverified.

## 1. Install dependencies and open the project

Open `Regional-Labor-Resilience/` as a folder in VS Code. The notebooks are all under `code/`, optional standalone synthetic tests under `tests/`, documentation under `docs/`, and `requirements.txt` at the repository root.

From VS Code PowerShell terminal at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select the `.venv` Python interpreter and Jupyter kernel in VS Code for each notebook.

**Root detection:** The notebooks search up from the current working directory for their project files. Open the repository root in VS Code. If this is not sufficient, insert a temporary cell at the top of **each notebook** and execute it before the existing first code cell:

```python
import os
os.environ['RL_PROJECT_ROOT'] = r'C:\Users\YOUR_USER\Documents\Regional-Labor-Resilience'
```

Use your actual project path. Keep the variable constant across notebooks. Never store local username paths as committed shared defaults.

**ACS API key (only if required for a fresh download in notebook 02):** Set the key in the environment of the running kernel, without adding the secret to source control. Example from PowerShell **before opening a new VS Code kernel**:

```powershell
$env:CENSUS_API_KEY = '<your Census API key>'
```

Alternatively configure a private environment for the Python process. The key should not appear in committed `.ipynb` files.

## 2. Run notebooks manually, in order

For each, choose **Restart Kernel and Run All** and review warnings and generated audit tables before advancing.

| Notebook | Full-run settings / expected role |
|---|---|
| `code/00_Acquire_QCEW_OMB2018.ipynb` | Download/reuse 2018 geography and QCEW, create the balanced CBSA panel and manifest. All eligible CBSAs; sample-selection audit. |
| `code/01_Stage1_Counterfactual_OMB2018.ipynb` | `VALIDATION_MONTHS=12`, `RUN_ETS=True`, `RUN_SARIMA=True`, standard candidate grids, `MAX_CBSAS=None`, `LONG_CV=True`, `RETRO_SPEC=True`, `LOG_BIAS_ADJUST=True`. Includes 54-month historical stress paths and descriptive retrospective fit. |
| `code/02_Download_PreFeb2020_Predictors.ipynb` | QCEW sector-share treatment and 2018 ACS predictor construction, no post-Feb-2020 features. |
| `code/03_Stage2_Resilience_and_Initial_Prediction.ipynb` | Derived resilience outcomes, model diagnostics, cross-dataset provenance checks. Requires notebook 01's complete Stage 1 manifest. |
| `code/04_Cumulative_Shortfall_Geographic_Validation.ipynb` | `MAX_OUTER_DIVISIONS=0`, `INCLUDE_HISTORICAL_STRESS=True`, `ENABLE_HGB=True`, `INCLUDE_SIGNED_46=True`, `BOOTSTRAP_REPS=1000`; nested CV for all Census divisions. |

Note that enabling every model and historical stress path can be computationally expensive. Notebook 01 defaults to 4 workers and notebook 04 to 2 workers. If you encounter genuine resource constraints, reduce `N_JOBS` in those notebooks; do not reduce regional sample size or geographic folds and call the result the full model.

## 3. Optional 24-, 36-, and 48-month selection designs

The primary notebook 01 keeps the **prespecified 12-month selection horizon**. Longer selection-horizon variants are separate scientific experiments; they are **not launched automatically**. When ready, copy the repository or isolate its `data/processed` and `output` results in a separate project root. Notebook 01 computes its selection horizon from `RL_STAGE1_SELECTION_HORIZON` and requires `RL_ALLOW_ALTERNATE_SELECTION=1` for alternate designs. Set both variables in a setup cell **before** the first code cell:

```python
import os
os.environ['RL_STAGE1_SELECTION_HORIZON'] = '24'
os.environ['RL_ALLOW_ALTERNATE_SELECTION'] = '1'
```

Repeat with 36 and 48 **only in isolated project copies**, and rerun Stage 1 and downstream notebooks there. Do not select a horizon by peeking at its held-out geographic test performance.

## 4. Synthetic tests kept separately (never run automatically)

The six scripts in `tests/` reference their notebooks using `Path(__file__).resolve().parents[1] / 'code'`. As a result, their imports work regardless of whether the VS Code terminal is rooted at the repository or `tests/`.

From the repository root, optional standalone commands are:

```powershell
.\.venv\Scripts\python.exe tests\test_acquisition_smoke.py
.\.venv\Scripts\python.exe tests\test_corrections.py
.\.venv\Scripts\python.exe tests\test_stage1_selection_config.py
.\.venv\Scripts\python.exe tests\test_historical_uncertainty_smoke.py
.\.venv\Scripts\python.exe tests\test_stage2_feature_smoke.py
.\.venv\Scripts\python.exe tests\test_geographic_smoke.py
```

These use synthetic input data and **do not verify** the empirical national model. They are optional and are not referenced by any notebook as mandatory execution steps.

## 5. Outputs and limitations

Generated raw/reusable inputs reside under `data/raw/`, processed panel/predictor/outcome files under `data/processed/`, and summary diagnostics, model scores, manifests, and figures under `output/`. The notebooks check hashes/provenance between stages; if an upstream file changes, rerun affected downstream notebooks in order.

The stress paths are historical error sensitivities, **not calibrated confidence intervals**. With no raw national datasets executed here, actual availability, coverage, model convergence, CBSA counts, and final metrics are unverified.
