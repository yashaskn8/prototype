# Live 100-question benchmark

The scripts run the real existing Django `ask_rag()` and customer formatter,
sequentially, without agents, model changes, index rebuilding, mock responses,
customer POST requests or company-record writes.

This run was explicitly authorized only for skydream / SKY-AST-00002, derived
Customer 13 / Asset 60. Do not expand scope automatically.

`Invoke-ServyLive100.ps1` invokes the installed `.venv` from PowerShell.
`servy_live100_benchmark.py` prepares independent source quotations and question
requirements, freezes their hashes, then executes/checkpoints exactly 100 IDs.
`servy_live100_audit.py` revalidates source bindings, records separate raw and
formatted assessments, and leaves unresolved semantics as NEEDS EXPERT REVIEW.

Company-derived files are stored beneath LOCALAPPDATA/Servy/private_evaluations,
outside the repository and OneDrive. The directory grants only the current user
and SYSTEM access. Windows packaged-app path redirection can produce a longer
canonical path; use the path printed by the prepare phase.

```powershell
# After explicit authorization for a new run:
.\tools\Invoke-ServyLive100.ps1 -Phase prepare -DataDirectory "$env:LOCALAPPDATA\Servy\private_evaluations\approved_run_name"
.\tools\Invoke-ServyLive100.ps1 -Phase run -DataDirectory "$env:LOCALAPPDATA\Servy\private_evaluations\approved_run_name"
.\.venv\Scripts\python.exe tools\servy_live100_audit.py --data-dir "$env:LOCALAPPDATA\Servy\private_evaluations\approved_run_name"
```

Do not overwrite a frozen run or mix changed sources/configuration/serving code.
Resume an existing run with `-Phase run`; completed IDs, including error records,
are not submitted again. Review failed records separately before any retry.

The benchmark enforces MySQL session read-only mode and rejects non-read SQL.
Operational counts are checked after every query. The formatter receives an
unsaved interaction with no fabricated customer login. Its signed source URLs
therefore do not constitute an authenticated portal/source-viewer test.

Source-derived targets are provisional. Neither exact-source matching nor a
retrieval hit establishes complete technical correctness or citation relevance.
Null scores are unresolved, not passes. Observable section omissions and repeated
generic uncertainty are reported separately. The detailed private method note
explains which judgments still need independent review. No model or grounding
correction is included in these scripts.
