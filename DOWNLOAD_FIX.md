# Automatic source acquisition for notebook 02

The pipeline now downloads all required LAUS, OES, OES-geography and O*NET files itself. No manual file download or browser step is required.

The downloader uses compatibility HTTP headers and urllib with verified platform TLS, including Windows certificate trust. It keeps certificate validation enabled, streams downloads into temporary files, checks the response length and ZIP/Excel structure, commits valid files atomically and records retrieval URLs, timestamps and SHA256 values. Existing valid project caches are reused.

Restart the notebook 02 kernel and choose Run All. Notebook 01 does not need to be rerun for this acquisition-only repair. You can also use the notebook's existing module-reload final cell when its prior cells remain in memory. Restarting avoids accidentally using an older imported downloader from a stale VS Code buffer.

The actual official sources were downloaded automatically into an initially empty validation cache during this repair:

| Source | Bytes | Parsing check |
|---|---:|---|
| LAUS 2018 county annual workbook | 232800 | 3220 county records |
| May 2018 OES metro/nonmetro archive | 31193526 | Correct employment workbooks selected; descriptive workbooks excluded |
| May 2018 OES area definitions | 230785 | Official MSA/FIPS headers recognized; town-based partial counties rejected |
| O*NET 24.1 text archive | 13477457 | Task rules parsed for 774 SOC categories |

The occupational acquisition functions were then exercised against the existing 936-CBSA geography without fitting a model. They produced 22 occupational-share columns, 297 exact OES county-set matches and 76 nonmissing task-feasibility scores under the existing coverage requirement. These are acquisition/coverage checks, not new prediction results.

The statistical specification, September 2018 delineations and February 29, 2020 release cutoff remain unchanged. No mirrors, later-year substitutes or later published telework-score files are used. Current historical BLS files can contain later revisions; this pre-existing vintage limitation remains documented.

The earlier DOWNLOAD_FIX.md manual procedure is superseded by this automatic transport. As with any online acquisition, a continuing service outage or network restriction can still stop execution, but it is reported explicitly rather than bypassed or treated as a missing predictor.
