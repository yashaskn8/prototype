# Servy 100-question evaluation: database access request

**Status: BLOCKED.** No benchmark query may run until a separate read-only account is provisioned and its effective permissions are verified. The configured `servy_app` account must remain unchanged and must not be used for evaluation.

## Database administrator action only

On the approved local MySQL instance, the administrator may provision a dedicated account restricted to loopback and the `servy` schema. The example below is a request for review, **not a command to be run by the evaluation process**. The administrator must choose and deliver the password through an approved secret channel.

```sql
CREATE USER 'servy_eval_ro'@'127.0.0.1' IDENTIFIED BY '<administrator-generated-secret>';
GRANT SELECT ON `servy`.* TO 'servy_eval_ro'@'127.0.0.1';
```

The schema-wide SELECT grant is a functional read-only starting point. If company policy requires narrower table access, the administrator should grant SELECT only on the tables used by the authorized RAG query path and confirm that every required read succeeds. Do not grant roles or privileges on other schemas. Do not change `servy_app`, application `.env`, company data, or production services.

## Evaluation-shell configuration

Use process-level variables so the shared `.env` remains unchanged. Never place the password in source files, command arguments, reports, or chat.

```powershell
$env:MYSQL_HOST = '127.0.0.1'
$env:MYSQL_DATABASE = 'servy'
$env:MYSQL_USER = 'servy_eval_ro'
$secret = Read-Host 'Read-only MySQL password' -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try { $env:MYSQL_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
Remove-Variable secret, bstr
```

`servy_rag/settings.py` loads `.env` with `override=False`, so these process variables take precedence only in that shell. Close the shell after testing.

## Required verification before any chatbot request

1. Connect using the evaluation-shell variables and verify `DATABASE()` is `servy` and `CURRENT_USER()` is `servy_eval_ro@127.0.0.1`. Stop on any fallback to `servy_app`.
2. Inspect `SHOW GRANTS FOR CURRENT_USER()` and `CURRENT_ROLE()`. Effective grants must contain only `USAGE` plus the approved `SELECT` scope. Reject global privileges, inherited write-capable roles, and any `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`, `INDEX`, `TRIGGER`, `EVENT`, `EXECUTE`, `FILE`, `CREATE USER`, or `GRANT OPTION` privilege. Do not test by attempting a write.
3. Under a read-only transaction and a SQL read-only guard, verify the required tenant, customer, asset, document, provenance, and chunk reads succeed.
4. Verify the frozen original Chroma hash manifest, activate a fresh private copy, and verify the original manifest again after the run. Never open the original index through Chroma during testing.
5. Require local BGE-small, SHA-pinned local V5 reranking, loopback-only Ollama or local extractive generation, model-download blocking, and remote LLM disabled. Paid API budget is ₹0.
6. Save raw and formatted responses only in a new ACL-protected private evaluation directory. Verify no operational database record counts changed.

The existing 100-question bank and previous 500-question results must remain unchanged. No benchmark requests have been made under this access request.
