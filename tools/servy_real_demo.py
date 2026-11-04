"""Loopback-only, SELECT-only real company RAG showcase.

This process has its own credential and identity, separate from the customer
portal and the synthetic tester. It never writes operational records.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
from tools.servy_500_benchmark import (activate_index_copy, file_manifest,
    load_json, operational_counts, prepare_index_copy, private_folder, read_only_database,
    serving, write_json)

PRIVATE_BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
PRIVATE = PRIVATE_BASE / "servy_real_demo_20261031_01"
ASSET_SCOPE = {59: 14, 60: 13, 61: 14}
DEMO_USER_ID = -27360
MAX_BODY = 5000
LOGGED_ROUTES = frozenset({"/demo-api/inventory/", "/demo-api/source/",
    "/demo-api/query/", "/demo-api/tickets/preview/"})


def _original_state():
    from django.conf import settings
    from core.models import KnowledgeDocument
    from core.services.customer_isolation import is_verified_real_document
    docs = {}
    for doc in KnowledgeDocument.objects.filter(tenant__slug="skydream").select_related(
            "tenant", "asset", "customer", "product", "category", "domain"):
        if not is_verified_real_document(doc) or not doc.file:
            continue
        path = Path(doc.file.path).resolve()
        media = Path(settings.MEDIA_ROOT).resolve()
        if not path.is_relative_to(media) or not path.is_file():
            raise RuntimeError("Verified original document is missing or outside media root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != (doc.checksum_sha256 or "").lower():
            raise RuntimeError("Verified original document hash mismatch")
        docs[str(doc.id)] = digest
    return {"operational_counts": operational_counts(),
            "documents": docs,
            "index": file_manifest(settings.CHROMA_PERSIST_DIR)}


def audit():
    """Compare company state with both this demo's before image and the prior frozen baseline."""
    configure_process()
    verify_readonly_identity()
    from django.conf import settings
    from core.models import (User, Tenant, Customer, Asset, KnowledgeDocument,
        KnowledgeChunk, CustomerInteraction, DiagnosticSession, ServiceCall)
    previous = load_json(PRIVATE / "before.json")
    frozen = load_json(PRIVATE_BASE / "servy_ui_test_isolated_20261031_01" /
                       "original_integrity_before.json")
    with read_only_database():
        current = _original_state()
        counts = {model.__name__: model.objects.count() for model in (
            User, Tenant, Customer, Asset, KnowledgeDocument, KnowledgeChunk,
            CustomerInteraction, DiagnosticSession, ServiceCall)}
        records = {}
        for doc in KnowledgeDocument.objects.filter(tenant__slug="skydream").order_by("id"):
            row = {"content_sha256": hashlib.sha256((doc.content_text or "").encode()).hexdigest(),
                   "checksum_sha256": doc.checksum_sha256, "index_version": doc.index_version}
            if doc.file:
                path = Path(doc.file.path).resolve()
                if not path.is_relative_to(Path(settings.MEDIA_ROOT).resolve()) or not path.is_file():
                    raise RuntimeError("Original attachment missing or outside media root")
                row["attachment_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            records[str(doc.id)] = row
    model = Path(settings.SERVY_RERANKER_MODEL_PATH).resolve()
    checks = {
        "demo_before_state": current == previous,
        "original_database_counts": counts == frozen["database"]["counts"],
        "original_company_documents": records == frozen["database"]["company_document_records"],
        "original_chroma": current["index"] == frozen["chroma_files"],
        "original_v5_weights": model.is_dir() and file_manifest(model) == frozen["reranker_files"],
    }
    return checks


def _asset(asset_id):
    from core.models import Asset, Tenant
    from core.services.customer_isolation import (is_verified_real_asset,
        is_verified_real_customer, validate_customer_serving_policy)
    tenant = Tenant.objects.get(slug="skydream", is_active=True)
    asset = Asset.objects.select_related("customer", "product", "site", "tenant").get(
        pk=asset_id, tenant=tenant)
    if not is_verified_real_asset(asset) or not asset.customer or not is_verified_real_customer(asset.customer):
        raise ValueError("Asset/customer binding is not verified")
    validate_customer_serving_policy(tenant, asset.customer, asset)
    return tenant, asset.customer, asset


def _inventory():
    from core.models import Asset, KnowledgeDocument
    from core.services.customer_isolation import is_verified_real_asset, is_verified_real_document
    from core.services.retriever import _knowledge_candidates
    rows = []
    for asset in Asset.objects.filter(tenant__slug="skydream").select_related(
            "tenant", "customer", "product", "site").order_by("id"):
        if not is_verified_real_asset(asset):
            continue
        approved_scope = asset.id in ASSET_SCOPE and ASSET_SCOPE[asset.id] == asset.customer_id
        documents = []
        if approved_scope:
            tenant, customer, checked_asset = _asset(asset.id)
            seen = set()
            for chunk in _knowledge_candidates(tenant, checked_asset, customer, False, backend="all"):
                doc = chunk.document
                if doc.id in seen or not is_verified_real_document(doc) or doc.is_confidential or doc.disable_sharing:
                    continue
                seen.add(doc.id)
                documents.append({"id": doc.id, "title": doc.title})
        rows.append({"id": asset.id, "code": asset.asset_code, "name": asset.name,
            "model": asset.model_number, "product": asset.product.name if asset.product else "",
            "customer_id": asset.customer_id, "site": asset.site.name if asset.site else "",
            "can_query": approved_scope, "documents": documents})
    return rows


def _answer(asset_id, question):
    if ASSET_SCOPE.get(asset_id) is None:
        return {"response_type": "INSUFFICIENT_EVIDENCE", "message":
            "This equipment is visible in the inventory, but technical-document access is not approved for this demo identity.",
            "presentation": None, "citations": [], "reranker_applied": False}
    from core.services.customer_answer import build_customer_answer
    from core.services.rag import ask_rag
    tenant, customer, asset = _asset(asset_id)
    if customer.id != ASSET_SCOPE[asset_id]:
        raise ValueError("Customer scope changed")
    rag = ask_rag(tenant=tenant, customer=customer, asset=asset, question=question,
                  staff_mode=False, include_confidential=False, service_call=None, top_k=5)
    interaction = SimpleNamespace(id=None, created_by_id=DEMO_USER_ID, tenant=tenant,
                                  customer=customer, asset=asset, question=question)
    contract = build_customer_answer(interaction, None, rag.get("references", []))
    presentation = contract["presentation"]
    for citation in presentation.get("citations", []):
        citation["url"] = citation["url"].replace("/api/customer-support/source/", "/demo-api/source/", 1)
    from django.core import signing
    ticket_context = signing.dumps({"asset": asset_id, "customer": customer.id,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()},
        salt="servy-real-demo-ticket-preview-v1")
    return {"response_type": contract["response_type"], "reason_code": contract.get("reason_code"),
            "message": presentation.get("message", ""), "presentation": presentation,
            "ticket_context": ticket_context,
            "citations": [{"document_id": c["document_id"], "chunk_id": c["chunk_id"]}
                          for c in presentation.get("citations", [])],
            "retrieval_backend": rag.get("retrieval_backend"),
            "reranker_applied": rag.get("reranker_applied"),
            "reranker_model": rag.get("reranker_model"), "engine": contract.get("engine")}


def _ticket_fields(body):
    """Validate a previously answered, scoped question and return ServiceCall fields only."""
    from django.core import signing
    token = body.get("ticket_context")
    question = body.get("question")
    complaint_type = body.get("complaint_type")
    complaint_text = body.get("complaint_text")
    if not all(isinstance(value, str) for value in
               (token, question, complaint_type, complaint_text)):
        raise ValueError("Invalid ticket preview")
    question = question.strip()
    complaint_type = complaint_type.strip()
    complaint_text = complaint_text.strip()
    if not question or len(question) > 4000 or not complaint_type or len(complaint_type) > 180 \
            or not complaint_text or len(complaint_text) > 4000:
        raise ValueError("Invalid ticket preview")
    context = signing.loads(token, salt="servy-real-demo-ticket-preview-v1", max_age=3600)
    asset_id = context.get("asset")
    if not isinstance(asset_id, int) or asset_id not in ASSET_SCOPE \
            or context.get("customer") != ASSET_SCOPE[asset_id] \
            or context.get("question_sha256") != hashlib.sha256(question.encode("utf-8")).hexdigest():
        raise ValueError("Ticket context is outside the approved question scope")
    _, customer, asset = _asset(asset_id)
    if customer.id != context["customer"]:
        raise ValueError("Customer scope changed")
    if asset.site_id and (asset.site.customer_id != customer.id or
                          asset.site.tenant_id != asset.tenant_id):
        raise ValueError("Site is outside the approved customer scope")
    return {"call_type": "Service", "complaint_type": complaint_type,
            "complaint_text": complaint_text, "customer_id": customer.id,
            "site_id": asset.site_id, "asset_id": asset.id,
            "priority": "normal", "status": "open"}


class Handler(BaseHTTPRequestHandler):
    token = ""

    def log_message(self, *_args):
        # Request URLs can contain signed citation tokens; never log them.
        pass

    def _send(self, status, value):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(raw)
        # Never log question text, signed source URLs, credentials, or response bodies.
        route = urlsplit(self.path).path
        route = route if route in LOGGED_ROUTES else "other"
        started = getattr(self, "_request_started", None)
        elapsed = f" {int((time.perf_counter() - started) * 1000)}ms" if started is not None else ""
        print(f"{time.strftime('%H:%M:%S')} {self.command} {route} {status}{elapsed}", flush=True)

    def _authorized(self):
        host = self.headers.get("Host", "").split(":", 1)[0]
        supplied = self.headers.get("Authorization", "")
        return (self.client_address[0] in ("127.0.0.1", "::1")
                and host in ("127.0.0.1", "localhost")
                and secrets.compare_digest(supplied, "Demo " + self.token))

    def do_GET(self):
        self._request_started = time.perf_counter()
        if not self._authorized():
            return self._send(401, {"detail": "Demo identity required"})
        path = urlsplit(self.path)
        try:
            with read_only_database():
                if path.path == "/demo-api/inventory/":
                    return self._send(200, {"assets": _inventory(), "scope": "skydream"})
                if path.path == "/demo-api/source/":
                    from core.services.customer_answer import open_citation
                    from django.core import signing
                    from core.services.customer_answer import SALT
                    payload = signing.loads(parse_qs(path.query).get("citation", [""])[0], salt=SALT, max_age=3600)
                    asset_id = payload.get("asset")
                    if ASSET_SCOPE.get(asset_id) != payload.get("customer") or payload.get("user") != DEMO_USER_ID:
                        raise ValueError("Source outside approved scope")
                    tenant, customer, asset = _asset(asset_id)
                    ctx = {"user": SimpleNamespace(id=DEMO_USER_ID), "tenant": tenant, "customer": customer,
                           "can_view_confidential": False}
                    return self._send(200, open_citation(parse_qs(path.query)["citation"][0], ctx))
        except Exception:
            return self._send(404, {"detail": "Source unavailable or access changed"})
        return self._send(404, {"detail": "Not found"})

    def do_POST(self):
        self._request_started = time.perf_counter()
        if not self._authorized():
            return self._send(401, {"detail": "Demo identity required"})
        path = urlsplit(self.path).path
        if path not in ("/demo-api/query/", "/demo-api/tickets/preview/"):
            return self._send(404, {"detail": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > (MAX_BODY if path == "/demo-api/query/" else 9000):
                return self._send(400, {"detail": "Invalid request size"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                return self._send(400, {"detail": "Invalid request"})
            if path != "/demo-api/query/":
                try:
                    with read_only_database():
                        fields = _ticket_fields(body)
                except Exception:
                    return self._send(400, {"detail": "Ticket preview is invalid or authorization changed"})
                return self._send(200, {"mode": "PREVIEW_ONLY", "ticket": fields})
            asset_id = body.get("asset_id")
            question = body.get("question")
            if not isinstance(asset_id, int) or not isinstance(question, str) or not question.strip() or len(question) > 4000:
                return self._send(400, {"detail": "Select equipment and enter a question (maximum 4000 characters)"})
            with read_only_database():
                result = _answer(asset_id, question.strip())
            return self._send(200, result)
        except Exception:
            return self._send(503, {"detail": "Read-only demo request failed"})


def main():
    if os.environ.get("SERVY_REAL_DEMO_MODE") != "1" or os.environ.get("DJANGO_DEBUG") != "1":
        raise RuntimeError("Explicit localhost debug demo mode is required")
    configure_process()
    from django.conf import settings
    if settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only" or settings.SERVY_LLM_PROVIDER != "extractive":
        raise RuntimeError("Verified-real, local extractive serving is required")
    identity = verify_readonly_identity()
    configuration = serving()
    private_folder(PRIVATE)
    access_file = PRIVATE / "real_demo_access.txt"
    if access_file.exists():
        token = access_file.read_text(encoding="ascii").strip()
    else:
        token = secrets.token_urlsafe(40)
        access_file.write_text(token, encoding="ascii")
    with read_only_database():
        original = _original_state()
        inventory = _inventory()
    if not any(row["id"] == 60 and row["can_query"] and row["documents"] for row in inventory):
        raise RuntimeError("Approved Customer 13 / Asset 60 documents are not ready")
    before = PRIVATE / "before.json"
    if before.exists():
        if load_json(before) != original:
            raise RuntimeError("Original data differs from the frozen demo baseline")
    else:
        write_json(before, original)
    copy = PRIVATE / "chroma_index_copy"
    manifest = PRIVATE / "production_index_manifest.json"
    if copy.exists() or manifest.exists():
        if not copy.is_dir() or not manifest.is_file():
            raise RuntimeError("Private Chroma copy is incomplete")
    else:
        prepare_index_copy(PRIVATE)
    activate_index_copy(PRIVATE)
    Handler.token = token
    print(json.dumps({"ready": True, "database": identity["database"],
        "database_account": identity["account"], "asset_count": len(inventory),
        "query_asset_scope": sorted(ASSET_SCOPE), "retrieval": configuration["retrieval"],
        "embedding": configuration["embedding"], "reranker_enabled": configuration["reranker_enabled"],
        "index": "private-copy", "access_file": str(access_file), "url": "http://127.0.0.1:8002/"}))
    HTTPServer(("127.0.0.1", 8002), Handler).serve_forever()


if __name__ == "__main__":
    if sys.argv[1:] == ["--audit"]:
        print(json.dumps(audit()))
    else:
        main()
