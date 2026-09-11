"""Dependency-free static red-team checks for the Servy RAG prototype.

Run with:  python red_team_static.py
This does not replace Django's dynamic test suite; it catches dangerous regressions
that can be checked without installing Django.
"""
from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).resolve().parent
FAILURES = []
PASSES = []


def check(name, ok, detail=""):
    (PASSES if ok else FAILURES).append((name, detail))


# 1) Every Python source must parse.
for path in ROOT.rglob("*.py"):
    if ".venv" in path.parts:
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        FAILURES.append((f"syntax:{path.relative_to(ROOT)}", str(exc)))
check("python-sources-parse", not any(n.startswith("syntax:") for n, _ in FAILURES))

# 2) No known Django security bypasses / unsafe evaluators in application source.
application_text = "\n".join(
    p.read_text(encoding="utf-8", errors="ignore")
    for p in list((ROOT / "core").rglob("*.py")) + list((ROOT / "servy_rag").rglob("*.py"))
)
for forbidden in ["csrf_exempt", "mark_safe(", "eval(", "exec(", "shell=True", "os.system("]:
    check(f"forbidden:{forbidden}", forbidden not in application_text, "unsafe construct present")

# 3) Media must not be exposed directly by Django development URLs.
urls_text = (ROOT / "servy_rag" / "urls.py").read_text()
check("no-public-media-route", "static(settings.MEDIA_URL" not in urls_text and "document_root=settings.MEDIA_ROOT" not in urls_text)

# 4) Frontend runtime assets are local; no CDN dependency in templates.
template_text = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "templates").rglob("*.html"))
check("no-runtime-cdn", not re.search(r"https?://(?:cdn\.|cdn.jsdelivr|unpkg|cdnjs)", template_text, re.I))

# 5) Startup scripts must not destroy data implicitly.
startup_names = ["run_windows.bat", "run_linux.sh", "setup_windows.bat", "setup_linux.sh"]
for name in startup_names:
    path = ROOT / name
    check(f"startup-no-reset:{name}", path.exists() and "seed_demo --reset" not in path.read_text(errors="ignore"))

# 6) RAG confidentiality and tenant filters should exist.
retriever = (ROOT / "core" / "services" / "retriever.py").read_text()
check("rag-tenant-filter", "tenant=tenant" in retriever)
check("rag-confidential-filter", "document__is_confidential=False" in retriever)
check("rag-quarantine-filter", "is_quarantined=False" in retriever)
check("rag-customer-filter", "document__customer" in retriever)
check("rag-past-resolution-scope", "staff_mode" in retriever)

# 7) Model relationship integrity and protected download must be present.
models = (ROOT / "core" / "models.py").read_text()
views = (ROOT / "core" / "views.py").read_text()
check("tenant-relation-validation", "_validate_tenant_relations" in models)
check("customer-relation-validation", "Call asset must belong to the selected customer" in models)
check("protected-kb-download", "def knowledge_download" in views and "FileResponse" in views)

# 8) Local LLM must default-deny remote endpoints and redirects.
llm = (ROOT / "core" / "services" / "llm.py").read_text()
check("local-llm-loopback-guard", "SERVY_ALLOW_REMOTE_LLM" in llm and "127.0.0.1" in llm and "allow_redirects=False" in llm)
check("local-llm-proxy-bypass", "trust_env = False" in llm)

# 9) Upload/import bounds and formula neutralisation.
forms = (ROOT / "core" / "forms.py").read_text()
excel = (ROOT / "core" / "services" / "excel_io.py").read_text()
check("kb-upload-limit", "MAX_KB_UPLOAD_BYTES" in forms)
check("kb-signature-check", "head.startswith(b\"%PDF-\")" in forms and "head.startswith(b\"PK\")" in forms)
check("xlsx-zip-bomb-limit", "SERVY_MAX_XLSX_UNCOMPRESSED_BYTES" in excel)
check("excel-formula-neutralisation", "_safe_excel" in excel and "{\"=\", \"+\", \"-\", \"@\"}" in excel)

# 10) Audit trail and AI rate limiting.
check("audit-events", (ROOT / "core" / "services" / "audit.py").exists() and "audit_event(" in views)
security = (ROOT / "core" / "security.py").read_text()
check("rag-rate-limit", "rate_limit_exceeded" in security and "status=429" in views)

# 11) Prompt-injection containment and server-side references.
indexing = (ROOT / "core" / "services" / "indexing.py").read_text()
rag = (ROOT / "core" / "services" / "rag.py").read_text()
check("prompt-injection-quarantine", "detect_document_prompt_injection" in indexing and "is_quarantined=bool(flags)" in indexing)
check("server-verified-references", '"references": refs' in rag)

# 12) No Docker dependency.
docker_files = [p for p in ROOT.rglob("*") if p.is_file() and (p.name.lower().startswith("docker") or p.name.lower() == "compose.yml")]
check("no-docker-required", not docker_files, ", ".join(str(p.relative_to(ROOT)) for p in docker_files))


# 13) Authorization boundaries for high-value routes.
for function_name in ["knowledge_base", "knowledge_download", "knowledge_upload", "call_register", "call_detail", "call_export", "call_import", "rag_assistant", "operations", "users"]:
    # Decorators appear immediately above the function; require one of the role guards.
    pattern = rf"@(roles_required|any_member)[^\n]*\n(?:@[^\n]+\n)*def {function_name}\("
    check(f"authz-route:{function_name}", bool(re.search(pattern, views)))
check("customer-identity-server-derived", 'customer = membership.customer' in views and 'CustomerRAGQueryForm(request.POST or None, tenant=tenant, customer=customer)' in views)
check("customer-no-confidential-rag", 'include_confidential=False, staff_mode=False' in views)

# 14) Project/asset many-to-many scope enforcement.
signals = (ROOT / "core" / "signals.py").read_text()
check("project-asset-m2m-tenant-guard", "enforce_project_asset_scope" in signals and "same tenant" in signals)
check("project-asset-m2m-customer-guard", "project customer" in signals)

# 15) Demo data lifecycle must be safe and explicit.
seed = (ROOT / "core" / "management" / "commands" / "seed_demo.py").read_text()
check("seed-idempotency-guard", "Demo data already exists; seed_demo made no changes" in seed)
for name in ["run_windows.bat", "run_linux.sh"]:
    text = (ROOT / name).read_text(errors="ignore")
    check(f"run-does-not-seed:{name}", "seed_demo" not in text)
check("explicit-reset-script", (ROOT / "reset_demo_windows.bat").exists() and "seed_demo --reset" in (ROOT / "reset_demo_windows.bat").read_text(errors="ignore"))

# 16) Remote knowledge ingestion is intentionally disabled (SSRF containment).
doc_loader = (ROOT / "core" / "services" / "document_loader.py").read_text()
check("no-remote-kb-fetch", "requests.get" not in doc_loader and "urlopen" not in doc_loader and "httpx" not in doc_loader)

# 17) Review SQLite must be relational and tenant-aware.
review_builder = ROOT / "sample_data" / "build_review_sqlite.py"
check("review-db-builder-present", review_builder.exists())
if review_builder.exists():
    review_text = review_builder.read_text(errors="ignore")
    check("review-db-foreign-keys", "PRAGMA foreign_keys=ON" in review_text and "PRAGMA foreign_key_check" in review_text)
    check("review-db-tenant-columns", review_text.count("tenant_id INTEGER NOT NULL") >= 20)

print(f"Static red-team checks: {len(PASSES)} passed, {len(FAILURES)} failed")
for name, detail in PASSES:
    print(f"[PASS] {name}")
for name, detail in FAILURES:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))

sys.exit(1 if FAILURES else 0)
