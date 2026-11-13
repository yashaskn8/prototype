"""Dependency-free static red-team checks for the current Servy RAG/API application.

Run with: python red_team_static.py
This complements, but does not replace, Django's dynamic test suite.
"""
from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[2]   # repository root (this file is tools/security/)
FAILURES = []
PASSES = []


def check(name, ok, detail=""):
    (PASSES if ok else FAILURES).append((name, detail))


def class_block(source, class_name):
    match = re.search(
        rf"^class {re.escape(class_name)}\b.*?(?=^class |\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


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
application_paths = list((ROOT / "core").rglob("*.py")) + list((ROOT / "servy_rag").rglob("*.py"))
application_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in application_paths)
for forbidden in ["csrf_exempt", "mark_safe(", "exec(", "shell=True", "os.system("]:
    check(f"forbidden:{forbidden}", forbidden not in application_text, "unsafe construct present")

unsafe_eval_calls = []
for source_path in application_paths:
    try:
        source_tree = ast.parse(source_path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        continue
    for node in ast.walk(source_tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if (isinstance(called, ast.Name) and called.id == "eval") or (
            isinstance(called, ast.Attribute) and called.attr == "eval"
        ):
            unsafe_eval_calls.append(f"{source_path.relative_to(ROOT)}:{node.lineno}")
check("forbidden:eval(", not unsafe_eval_calls, ", ".join(unsafe_eval_calls) or "unsafe eval call not found")

# 3) Django exposes the React app and API, not the retired server-rendered prototype.
urls_text = (ROOT / "servy_rag" / "urls.py").read_text(encoding="utf-8")
check("no-public-media-route", "static(settings.MEDIA_URL" not in urls_text and "document_root=settings.MEDIA_ROOT" not in urls_text)
check("root-redirects-to-react", 'RedirectView.as_view(url="/app/"' in urls_text)
check("legacy-django-ui-removed", not (ROOT / "core" / "views.py").exists() and not (ROOT / "core" / "urls.py").exists() and not (ROOT / "templates" / "core").exists())

# React runtime source must not rely on a CDN.
frontend_sources = [
    p for p in (ROOT / "frontend" / "src").rglob("*")
    if p.is_file() and p.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".css"}
]
frontend_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in frontend_sources)
check("no-runtime-cdn", not re.search(r"https?://(?:cdn\.|cdn.jsdelivr|unpkg|cdnjs)", frontend_text, re.I))

# 4) Startup scripts must not seed or reset synthetic business records implicitly.
for name in ["run_linux.sh", "setup_windows.bat", "setup_linux.sh"]:
    path = ROOT / "tools" / "launchers" / name
    check(f"startup-no-demo-seed:{name}", path.exists() and "seed_demo" not in path.read_text(errors="ignore"))

# 5) RAG confidentiality and tenant filters should exist.
retriever = (ROOT / "core" / "services" / "retriever.py").read_text(encoding="utf-8")
check("rag-tenant-filter", "tenant=tenant" in retriever)
check("rag-confidential-filter", "document__is_confidential=False" in retriever)
check("rag-quarantine-filter", "is_quarantined=False" in retriever)
check("rag-customer-filter", "document__customer" in retriever)
check("rag-past-resolution-scope", "staff_mode" in retriever)

# 6) Model relationship integrity and protected knowledge downloads must be present.
models = (ROOT / "core" / "models.py").read_text(encoding="utf-8")
knowledge_download = class_block((ROOT / "core" / "api_views.py").read_text(encoding="utf-8"), "KnowledgeDownloadView")
check("tenant-relation-validation", "_validate_tenant_relations" in models)
check("customer-relation-validation", "Call asset must belong to the selected customer" in models)
check("protected-kb-download", "require_api_context(request)" in knowledge_download and "authorized_knowledge_queryset(ctx)" in knowledge_download)

# 7) Upload/import bounds and formula neutralisation are enforced by API/service code.
api_views = (ROOT / "core" / "api_views.py").read_text(encoding="utf-8")
knowledge_upload = class_block(api_views, "KnowledgeUploadView")
asset_upload = class_block(api_views, "AssetDocumentUploadView")
check("kb-upload-limit", "uploaded_file.size > MAX_KB_UPLOAD_BYTES" in knowledge_upload and "uploaded_file.size > MAX_KB_UPLOAD_BYTES" in asset_upload)
check("kb-signature-check", 'head.startswith(b"%PDF-")' in knowledge_upload and 'head.startswith(b"PK")' in knowledge_upload)
excel = (ROOT / "core" / "services" / "excel_io.py").read_text(encoding="utf-8")
check("xlsx-zip-bomb-limit", "SERVY_MAX_XLSX_UNCOMPRESSED_BYTES" in excel)
check("excel-formula-neutralisation", "_safe_excel" in excel and '{"=", "+", "-", "@"}' in excel)

# 8) Customer RAG API enforces identity, asset scope and throttling.
customer_query = class_block(api_views, "CustomerSupportQueryView")
check("customer-rag-api-authz", "require_api_context(request)" in customer_query)
check("customer-context-server-derived", 'customer = ctx["customer"]' in customer_query and "membership.customer_id != customer.id" in customer_query)
check("customer-asset-scope-revalidated", "validate_customer_serving_policy" in customer_query and "Asset.objects.get(" in customer_query)
check("customer-rag-throttled", "throttle_classes = [CustomerRAGThrottle]" in customer_query)

# 9) Conversation-to-service-call and technician inbox APIs preserve the handoff.
ticket_view = class_block(api_views, "CustomerConversationTicketView")
check("ticket-requires-explicit-confirmation", "confirm_ticket_creation" in ticket_view and "is not True" in ticket_view)
check("ticket-creation-is-idempotent-and-locked", "select_for_update" in ticket_view and "linked_service_call" in ticket_view)
check("ticket-creates-servy-service-call", "create_call_from_interaction" in ticket_view)
technician_api = (ROOT / "core" / "technician_inbox.py").read_text(encoding="utf-8")
check("technician-role-boundary", "TECH_ROLES" in technician_api and "require_api_context(request)" in technician_api)
check("technician-audit-trail", "audit_event(" in technician_api)

# 10) Local model serving remains explicit and remote generation remains guarded.
llm = (ROOT / "core" / "services" / "llm.py").read_text(encoding="utf-8")
check("local-llm-loopback-guard", "SERVY_ALLOW_REMOTE_LLM" in llm and "127.0.0.1" in llm and "allow_redirects=False" in llm)
check("local-llm-proxy-bypass", "trust_env = False" in llm)

# 11) Prompt-injection containment and server-side references.
indexing = (ROOT / "core" / "services" / "indexing.py").read_text(encoding="utf-8")
rag = (ROOT / "core" / "services" / "rag.py").read_text(encoding="utf-8")
check("prompt-injection-quarantine", "detect_document_prompt_injection" in indexing and "is_quarantined=bool(flags)" in indexing)
check("server-verified-references", '"references": refs' in rag)

# 12) No Docker dependency.
docker_files = [
    p for p in ROOT.rglob("*")
    if p.is_file() and (p.name.lower().startswith("docker") or p.name.lower() == "compose.yml")
]
check("no-docker-required", not docker_files, ", ".join(str(p.relative_to(ROOT)) for p in docker_files))

# 13) The previous screenshot-driven dummy UI and seed dataset must not return.
check("no-synthetic-business-seeder", not (ROOT / "core" / "management" / "commands" / "seed_demo.py").exists())
check("no-dummy-sqlite-review-db", not (ROOT / "sample_data" / "servy_dummy_schema.sqlite3").exists())
check("no-dummy-xlsx-dataset", not (ROOT / "sample_data" / "servy_dummy_data.xlsx").exists())
check("no-legacy-bootstrap-assets", not (ROOT / "static" / "vendor" / "bootstrap").exists())

# 14) Remote knowledge ingestion is intentionally disabled (SSRF containment).
doc_loader = (ROOT / "core" / "services" / "document_loader.py").read_text(encoding="utf-8")
check("no-remote-kb-fetch", "requests.get" not in doc_loader and "urlopen" not in doc_loader and "httpx" not in doc_loader)

# 15) Core project/asset relationship guards remain.
signals = (ROOT / "core" / "signals.py").read_text(encoding="utf-8")
check("project-asset-m2m-tenant-guard", "enforce_project_asset_scope" in signals and "same tenant" in signals)
check("project-asset-m2m-customer-guard", "project customer" in signals)

print(f"Static red-team checks: {len(PASSES)} passed, {len(FAILURES)} failed")
for name, detail in PASSES:
    print(f"[PASS] {name}")
for name, detail in FAILURES:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))

sys.exit(1 if FAILURES else 0)
