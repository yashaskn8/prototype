# Operation 25 — candidate recall diagnostics

Historical standalone diagnostics for candidate recall and ranking reconciliation. These scripts are not part of the application request path or current CI workflow. They rely on the local evaluation benchmark and private company data configured for the original investigation.

Run explicitly from the repository root with the project virtual environment, for example:

```powershell
.\.venv\Scripts\python.exe -B tools\op25\op25_diagnose_candidates.py
```

Private outputs remain under the ignored project evaluation directory. These results are historical diagnostics, not a current production benchmark.
