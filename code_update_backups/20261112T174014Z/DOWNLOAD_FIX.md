# Recover notebook 02 source downloads

The screenshot shows failure to acquire the 2018 LAUS county workbook. The former downloader suppressed the original error, so that traceback does not establish whether BLS blocked the request, a TLS certificate failed, or the connection failed.

The revised downloader reports HTTP/TLS/network causes, retries temporary failures, and automatically adopts a valid exact-name file from your Downloads directory or `RL_SOURCE_DOWNLOAD_DIR`. It rejects HTML error pages disguised as Excel and does not skip the predictor or substitute a newer year. Notebook 02 reloads its helper module before the final cell to avoid retaining the old function in memory.

1. Open https://www.bls.gov/lau/laucnty18.xlsx in your normal browser. This is the 2018 annual county source linked by https://www.bls.gov/lau/tables.htm .
2. Save it as **laucnty18.xlsx** in your Downloads folder. Alternatively, save it directly to `data/raw/laus/laucnty18.xlsx` within the project. If your browser added `(1)`, rename the downloaded file to its exact original filename.
3. Rerun the failed final cell in notebook 02. The new helper detects and copies the browser download into the project cache. If you overwrite the helper without replacing the notebook, restart the kernel and run notebook 02 again so Python imports the new code.

For a different Downloads location, set this in the notebook before the final cell:

```python
import os
os.environ['RL_SOURCE_DOWNLOAD_DIR'] = r'D:\Your Download Folder'
```

If OES retrieval is also blocked, the same recovery supports the official filenames `oesm18ma.zip` and `area_definitions_m2018.xlsx`. O*NET uses `db_24_1_text.zip`. Links and required cache locations are in `RERUN_DESIGN_2020.md`.

No Stage 1 rerun is needed for this downloader-only fix. A successful notebook 02 must still precede notebooks 03--05. Four offline downloader checks and the nine existing design checks passed. The statistical specification and cutoff are unchanged; no empirical model was run. This patch does not establish that live BLS downloads will work automatically on your network.
