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
from tools.servy_500_benchmark import (
    file_manifest, load_json, operational_counts, prepare_index_copy,
    private_folder, read_only_database, serving, write_json,
)
from tools.servy_real_demo_index_review import (
    activate_runtime_snapshot, review_request, verify_signed_approval,
)
from tools.servy_real_demo_source_rebuild import rebuild_isolated_bge_index
from tools.servy_real_demo_local_planner import plan_turn
from tools.servy_real_demo_query_plan import targeted_retry_query
from tools.servy_real_demo_dialogue import (
    TOPICS, TECHNICAL_SYMPTOMS, conversational_turn, conversation_recap,
    followup_clarification, vague_problem_clarification, focus_answer, is_component_followup,
    is_contextual_followup, focused_causal_query, requested_topics,
    out_of_scope_response,
)

PRIVATE_BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
PRIVATE = PRIVATE_BASE / "servy_real_demo_20261031_01"
ASSET_SCOPE = {59: 14, 60: 13, 61: 14}
DEMO_USER_ID = -27360
MAX_BODY = 12000
# A signed, bounded continuation contains only customer-reported text. It
# lives in the browser; the real-data demo never writes conversation records.
DEMO_CONTEXT_SALT = "servy-real-demo-conversation-v1"
DEMO_CONTEXT_MAX_AGE = 7200
DEMO_CONTEXT_MAX_TURNS = 6
DEMO_CONTEXT_TEXT_LIMIT = 650
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


def _load_demo_context(token, asset_id, customer_id, thread_id=None):
    """Read only a server-signed continuation for this exact asset/customer."""
    if token is None or token == "":
        return "", [], False
    if not isinstance(token, str) or len(token) > 10000:
        raise ValueError("Invalid conversation continuation")
    from django.core import signing
    try:
        state = signing.loads(token, salt=DEMO_CONTEXT_SALT, max_age=DEMO_CONTEXT_MAX_AGE)
    except signing.BadSignature as exc:
        raise ValueError("Invalid or expired conversation continuation") from exc
    if not isinstance(state, dict) or state.get("asset") != asset_id or state.get("customer") != customer_id:
        raise ValueError("Conversation continuation is outside the selected equipment scope")
    # Thread binding is signed. The continuation of one tab conversation
    # must not be accepted in another thread, even for the same asset.
    bound_thread = state.get("thread_id")
    if bound_thread is not None or thread_id is not None:
        if (not isinstance(bound_thread, str)
                or not isinstance(thread_id, str)
                or bound_thread != thread_id):
            raise ValueError("Conversation continuation belongs to another chat")
    first, recent = state.get("first"), state.get("recent")
    if (not isinstance(first, str) or not first.strip()
        or len(first) > DEMO_CONTEXT_TEXT_LIMIT or not isinstance(recent, list)
        or not 1 <= len(recent) <= DEMO_CONTEXT_MAX_TURNS
        or any(not isinstance(t, str) or not t.strip()
               or len(t) > DEMO_CONTEXT_TEXT_LIMIT for t in recent)):
        raise ValueError("Invalid conversation continuation")
    return first, recent, state.get("hazard") is True


def _load_demo_dialogue_state(token):
    """Restore only server-signed, bounded UI dialogue preferences."""
    if not token:
        return None, []
    from django.core import signing
    state = signing.loads(token, salt=DEMO_CONTEXT_SALT, max_age=DEMO_CONTEXT_MAX_AGE)
    topic = state.get("last_topic")
    facts = state.get("shown_facts", [])
    if (topic not in TOPICS and topic is not None
            or not isinstance(facts, list)
            or len(facts) > 24
            or any(not isinstance(item, str)
                   or len(item) != 64
                   or any(ch not in "0123456789abcdef" for ch in item)
                   for item in facts)):
        raise ValueError("Invalid signed dialogue state")
    return topic, facts


def _sign_demo_context(asset_id, customer_id, question, first, history,
                       hazard=False, last_topic=None, shown_facts=None,
                       thread_id=None):
    from django.core import signing
    recent = (history + [question[:DEMO_CONTEXT_TEXT_LIMIT]])[-DEMO_CONTEXT_MAX_TURNS:]
    return signing.dumps({
        "asset": asset_id,
        "customer": customer_id,
        **({"thread_id": thread_id} if thread_id is not None else {}),
        "first": (first or question)[:DEMO_CONTEXT_TEXT_LIMIT],
        "recent": recent,
        "hazard": bool(hazard),
        "last_topic": last_topic if last_topic in TOPICS else None,
        "shown_facts": list(shown_facts or [])[-24:],
    }, salt=DEMO_CONTEXT_SALT, compress=True)


def _demo_query_context(asset, question, first, history):
    """Build retrieval context from customer reports, never from AI claims."""
    from core.services.conversation_context import (
        ConversationContext, _confirmed_checks,
        contextualize_customer_query, extract_symptoms_from_text,
    )
    context = ConversationContext(
        conversation_uuid="real-demo-ephemeral",
        asset_id=asset.id,
        asset_name=asset.name or "equipment",
        asset_model=asset.model_number or "",
        asset_code=asset.asset_code or "",
        product_name=asset.product.name if asset.product else "",
    )
    from tools.servy_real_demo_dialogue import TECHNICAL_SYMPTOMS
    # Preserve the signed chat for recap, but search the latest independently
    # reported technical issue instead of mixing every past issue into it.
    active_question = bool(TECHNICAL_SYMPTOMS.search(question)) and not is_contextual_followup(question)
    active_index = 0
    if not active_question:
        for index in range(len(history) - 1, -1, -1):
            if TECHNICAL_SYMPTOMS.search(history[index]) and not out_of_scope_response(history[index]):
                active_index = index
                break
    active_history = [] if active_question else history[active_index:]
    context.turn_count = len(active_history)
    context.original_complaint = (question if active_question else
                                  (active_history[0] if active_history else first or question))
    for utterance in [context.original_complaint, *active_history, question]:
        for symptom in extract_symptoms_from_text(utterance):
            if symptom not in context.reported_symptoms:
                context.reported_symptoms.append(symptom)
        # The user must name a completed action explicitly. "I checked that"
        # cannot identify which suggestion was performed in this read-only demo.
        for completed in _confirmed_checks(utterance, []):
            if completed not in context.completed_checks:
                context.completed_checks.append(completed)
    context.history_turns = [{"role": "customer", "text": t} for t in active_history]
    # Never append prior equipment symptoms to an unrelated question just
    # because it is short, or to a new self-contained symptom description.
    context.is_followup = bool(active_history) and is_contextual_followup(question)
    contextual_question = contextualize_customer_query(question, context, asset)
    return context, contextual_question or question


def _answer(asset_id, question, continuation=None, thread_id=None):
    if ASSET_SCOPE.get(asset_id) is None:
        return {"response_type": "INSUFFICIENT_EVIDENCE", "message":
            "This equipment is visible in the inventory, but technical-document access is not approved for this demo identity.",
            "presentation": None, "citations": [], "reranker_applied": False}
    from core.services.customer_answer import _question_guard, build_customer_answer, exception_presentation
    from core.services.customer_diagnostic import check_for_safety_hazard, check_for_vague_complaint
    from core.services.rag import ask_rag
    tenant, customer, asset = _asset(asset_id)
    if customer.id != ASSET_SCOPE[asset_id]:
        raise ValueError("Customer scope changed")
    if (thread_id is not None
            and (not isinstance(thread_id, str)
                 or not re.fullmatch(r"thread-[a-z0-9-]{8,90}", thread_id))):
        raise ValueError("Invalid conversation thread identity")
    original, history, earlier_hazard = _load_demo_context(
        continuation, asset_id, customer.id, thread_id=thread_id)
    last_topic, previous_facts = _load_demo_dialogue_state(continuation)
    context, retrieval_query = _demo_query_context(asset, question, original, history)
    # A new, explicit equipment symptom starts a new evidence thread; retain
    # the signed customer history, but don't hide relevant facts just because
    # they were shown for an earlier and different problem.
    if (history and not context.is_followup
            and TECHNICAL_SYMPTOMS.search(question)
            and not out_of_scope_response(question)):
        previous_facts = []
        last_topic = None
    current_hazard, hazard_warning = check_for_safety_hazard(question)
    small_talk = conversational_turn(question)
    ambiguous_reply = (followup_clarification(question, history)
                       or vague_problem_clarification(question, history))
    recap = conversation_recap(question, history)
    off_topic = out_of_scope_response(question)
    suggestions = []
    local_planner_used = False
    retrieval_attempts = 0
    is_hazard = current_hazard or earlier_hazard
    if is_hazard:
        warning = hazard_warning or (
            "A previously reported safety hazard requires authorized service. "
            "Do not operate or open the equipment."
        )
        presentation = exception_presentation({
            "response_mode": "IMMEDIATE_SAFETY_ESCALATION",
            "response_type": "ESCALATION_REQUIRED",
            "asset": {"id": asset.id, "name": asset.name, "model_number": asset.model_number},
            "summary": warning, "stop_conditions": [warning], "can_escalate": True,
        })
        contract = {"response_type": "ESCALATION_REQUIRED", "presentation": presentation,
                    "reason_code": "SAFETY_HAZARD", "engine": "safety-boundary"}
        rag = {"retrieval_backend": "none", "reranker_applied": False, "reranker_model": None}
    elif off_topic is not None:
        # An out-of-scope question must not perform RAG, claim an AC citation,
        # change active equipment context, or become a service-ticket draft.
        presentation = exception_presentation({
            "response_type": "INSUFFICIENT_EVIDENCE",
            "summary": off_topic,
            "asset": {"id": asset.id, "name": asset.name,
                      "model_number": asset.model_number},
            "can_escalate": False,
        })
        contract = {"response_type": "INSUFFICIENT_EVIDENCE",
                    "presentation": presentation,
                    "reason_code": "OUT_OF_SCOPE_QUERY",
                    "engine": "dialogue-scope-guard"}
        rag = {"retrieval_backend": "not-required",
               "reranker_applied": False, "reranker_model": None}
    elif small_talk is not None or ambiguous_reply is not None or recap is not None:
        presentation = exception_presentation({
            "response_type": "CLARIFICATION",
            "clarification_question": small_talk or ambiguous_reply or recap,
            "asset": {"id": asset.id, "name": asset.name, "model_number": asset.model_number},
            "can_escalate": True,
        })
        contract = {"response_type": "CLARIFICATION", "presentation": presentation,
                    "reason_code": "CONVERSATION_ACKNOWLEDGEMENT" if small_talk
                                   else "AMBIGUOUS_CUSTOMER_REPORT" if ambiguous_reply
                                   else "CUSTOMER_REPORT_RECAP",
                    "engine": "local-dialogue"}
        rag = {"retrieval_backend": "not-required", "reranker_applied": False,
               "reranker_model": None}
    else:
        # Targeted component follow-ups benefit from a slightly wider
        # authorized candidate set, without adding an unbounded agent loop.
        # A multi-intent question draws causes and checks from different
        # passages, so it gets the same bounded wider set.
        multi_intent = len(requested_topics(question)) > 1
        retrieval_top_k = 8 if (is_component_followup(question) and history) or multi_intent else 5
        rag = ask_rag(
            tenant=tenant, customer=customer, asset=asset, question=retrieval_query,
            original_question=question, conversation_context=context if history else None,
            staff_mode=False, include_confidential=False, service_call=None,
            top_k=retrieval_top_k,
        )
        retrieval_attempts = 1
        if rag.get("engine") == "policy-guard":
            presentation = exception_presentation({
                "response_type": "POLICY_VIOLATION",
                "summary": rag.get("answer", "Request cannot be answered."),
                "asset": {"id": asset.id, "name": asset.name, "model_number": asset.model_number},
                "can_escalate": False,
            })
            contract = {"response_type": "POLICY_VIOLATION", "presentation": presentation,
                        "engine": "policy-guard", "reason_code": "POLICY_BLOCKED"}
        else:
            interaction = SimpleNamespace(
                id=None, created_by_id=DEMO_USER_ID, tenant=tenant, customer=customer,
                asset=asset, question=retrieval_query,
            )
            # A short continuation ("what next?") is clarified by the signed
            # history; do not reject it as a standalone vague first question.
            answer_question = (
                retrieval_query if history and context.is_followup
                and _question_guard(question) == "clarify" else question
            )
            contract = build_customer_answer(
                interaction, None, rag.get("references", []),
                conversation_context=context if history else None,
                original_question=answer_question,
            )
            presentation = contract["presentation"]
            for citation in presentation.get("citations", []):
                citation["url"] = citation["url"].replace(
                    "/api/customer-support/source/", "/demo-api/source/", 1
                )
            # Local Ollama may plan intent/evidence order only; it cannot
            # create, edit or authorize a technical claim or a citation.
            plan = plan_turn(
                question, history=history, last_topic=last_topic, contract=contract,
            )
            local_planner_used = plan is not None
            contract, dialogue = focus_answer(
                contract, question, last_topic=last_topic,
                previous_facts=previous_facts, plan=plan,
                history=history,
            )
            last_topic = dialogue["topic"]
            previous_facts = dialogue["facts"]
            suggestions = dialogue["suggestions"]

            # Conditional agentic retrieval: at most ONE additional query.
            # The first pass must be merely non-answerable, never a refusal,
            # denied source, restricted procedure or safety escalation.
            if (contract.get("response_type") == "INSUFFICIENT_EVIDENCE"
                    and contract.get("reason_code") == "NO_RELEVANT_APPROVED_EVIDENCE"):
                second_query = targeted_retry_query(question, asset, context)
                if second_query and second_query != retrieval_query:
                    retried = ask_rag(
                        tenant=tenant, customer=customer, asset=asset,
                        question=second_query, original_question=question,
                        conversation_context=context if history else None,
                        staff_mode=False, include_confidential=False,
                        service_call=None, top_k=8,
                    )
                    retrieval_attempts = 2
                    if retried.get("engine") != "policy-guard":
                        # Build the citation contract afresh. The second pass
                        # does not trust or reuse a claim from the first pass.
                        second_interaction = SimpleNamespace(
                            id=None, created_by_id=DEMO_USER_ID, tenant=tenant,
                            customer=customer, asset=asset, question=second_query,
                        )
                        next_contract = build_customer_answer(
                            second_interaction, None,
                            retried.get("references", []),
                            conversation_context=context if history else None,
                            original_question=question,
                        )
                        if next_contract.get("response_type") == "DIAGNOSTIC_GUIDANCE":
                            for citation in next_contract["presentation"].get("citations", []):
                                citation["url"] = citation["url"].replace(
                                    "/api/customer-support/source/",
                                    "/demo-api/source/", 1,
                                )
                            next_plan = plan_turn(
                                question, history=history,
                                last_topic=last_topic, contract=next_contract,
                            )
                            next_contract, next_dialogue = focus_answer(
                                next_contract, question, last_topic=last_topic,
                                previous_facts=previous_facts,
                                plan=next_plan, history=history,
                            )
                            if next_contract.get("response_type") == "DIAGNOSTIC_GUIDANCE":
                                contract = next_contract
                                rag = retried
                                local_planner_used = next_plan is not None
                                last_topic = next_dialogue["topic"]
                                previous_facts = next_dialogue["facts"]
                                suggestions = next_dialogue["suggestions"]

    from django.core import signing
    ticket_context = signing.dumps({
        "asset": asset_id, "customer": customer.id,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
    }, salt="servy-real-demo-ticket-preview-v1")
    presentation = contract["presentation"]
    response_type = contract["response_type"]
    message = presentation.get("message", "")
    clarification_question = None
    clarification_options = []
    if response_type == "INSUFFICIENT_EVIDENCE" and not history:
        vague, clarifying, options = check_for_vague_complaint(question, asset)
        if vague or _question_guard(question) == "clarify":
            clarification_question = clarifying or (
                "Which specific symptom, warning, or displayed fault code are you observing?"
            )
            clarification_options = options or [
                "The equipment does not power on",
                "A fault code or warning is displayed",
                "Performance is reduced",
                "There is an unusual noise or vibration",
            ]
            response_type = "CLARIFICATION"
            message = clarification_question
            # No documentation is implied by these generic observation choices.
            presentation = exception_presentation({
                "response_type": "CLARIFICATION", "clarification_question": message,
                "asset": {"id": asset.id, "name": asset.name, "model_number": asset.model_number},
                "can_escalate": True,
            })
    return {
        "response_type": response_type, "reason_code": contract.get("reason_code"),
        "message": message, "presentation": presentation,
        "clarification_question": clarification_question,
        "clarification_options": clarification_options,
        "suggested_follow_ups": suggestions if response_type == "DIAGNOSTIC_GUIDANCE" else [],
        # Ignore unrelated turns in RAG memory. Keep the previous signed
        # continuation exactly unchanged, including its customer/asset scope.
        "continuation": continuation if off_topic is not None else _sign_demo_context(
            asset_id, customer.id, question, original, history, hazard=is_hazard,
            last_topic=last_topic, shown_facts=previous_facts, thread_id=thread_id,
        ),
        "conversation_turn": len(history) + 1,
        "dialogue_planner": "local-ollama-structured" if local_planner_used else "deterministic-grounded",
        "retrieval_attempts": retrieval_attempts,
        "ticket_context": None if off_topic is not None else ticket_context,
        "citations": [
            {"document_id": c["document_id"], "chunk_id": c["chunk_id"]}
            for c in presentation.get("citations", [])
        ],
        "retrieval_backend": rag.get("retrieval_backend"),
        "reranker_applied": rag.get("reranker_applied"),
        "reranker_model": rag.get("reranker_model"),
        "engine": contract.get("engine"),
    }


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
            continuation = body.get("continuation")
            thread_id = body.get("thread_id")
            if (not isinstance(asset_id, int) or isinstance(asset_id, bool)
                or not isinstance(question, str) or not question.strip()
                or len(question) > 4000
                or continuation is not None and not isinstance(continuation, str)
                or thread_id is not None and
                   (not isinstance(thread_id, str)
                    or not re.fullmatch(r"thread-[a-z0-9-]{8,90}", thread_id))):
                return self._send(400, {"detail": "Select equipment and enter a question (maximum 4000 characters)"})
            with read_only_database():
                if thread_id is None:
                    result = _answer(asset_id, question.strip(), continuation=continuation)
                else:
                    result = _answer(asset_id, question.strip(), continuation=continuation,
                                     thread_id=thread_id)
            return self._send(200, result)
        except ValueError:
            return self._send(400, {"detail": "Conversation context is invalid or equipment access changed. Start a new chat."})
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
    copy = PRIVATE / "chroma_index_copy"
    manifest = PRIVATE / "production_index_manifest.json"
    source = Path(settings.CHROMA_PERSIST_DIR).resolve()
    if before.is_file():
        frozen = load_json(before)
        # Only the explicitly reviewed Chroma storage bytes may differ.
        # Original company document records, database counts and V5 model
        # weights must still match the earlier independent frozen baseline.
        baseline_checks = audit()
        required = ("original_database_counts", "original_company_documents",
                    "original_v5_weights")
        if any(baseline_checks.get(key) is not True for key in required):
            raise RuntimeError("A non-Chroma company integrity control differs from the original frozen baseline.")
        # Keep both the frozen company data and original attachments immutable.
        if (frozen.get("operational_counts") != original["operational_counts"]
                or frozen.get("documents") != original["documents"]):
            raise RuntimeError("Company data or original documents differ from the frozen baseline.")
    else:
        # Initial bootstrap remains strict; never reconstruct a missing baseline
        # when earlier private evidence already exists.
        if copy.exists() or manifest.exists():
            raise RuntimeError("Frozen demo baseline missing but private Chroma evidence exists.")
        write_json(before, original)
        frozen = original

    if not manifest.is_file() and not copy.exists():
        prepare_index_copy(PRIVATE)
    if not manifest.is_file() or not copy.is_dir():
        raise RuntimeError("Frozen private Chroma snapshot or manifest is missing.")

    expected = frozen["index"]
    if load_json(manifest) != expected:
        raise RuntimeError("Frozen original Chroma manifests disagree.")
    if original["index"] != expected and os.environ.get("SERVY_REAL_DEMO_REBUILD_VERIFIED_SOURCES") == "1":
        # Distinct explicit recovery mode: rebuild a NEW local vector index
        # from unchanged, independently audited source documents. Never trust
        # or query the three byte-drifted Chroma storage files.
        with read_only_database():
            rebuilt = rebuild_isolated_bge_index(
                PRIVATE, original["index"], ASSET_SCOPE, _asset
            )
        review_mode = rebuilt["mode"]
    elif original["index"] == expected:
        # Frozen original image matches: run only from a verified throwaway copy.
        activate_runtime_snapshot(source, expected, PRIVATE)
        review_mode = "original-frozen"
    else:
        # Existing signed review pathway for a separately approved snapshot.
        private_manifest = file_manifest(copy)
        request = review_request(
            PRIVATE, expected=expected, production=original["index"],
            private_copy=private_manifest,
        )
        verify_signed_approval(PRIVATE, request)
        activate_runtime_snapshot(copy, private_manifest, PRIVATE)
        review_mode = "independently-reviewed-versioned"
    Handler.token = token
    print(json.dumps({"ready": True, "database": identity["database"],
        "database_account": identity["account"], "asset_count": len(inventory),
        "query_asset_scope": sorted(ASSET_SCOPE), "retrieval": configuration["retrieval"],
        "embedding": configuration["embedding"], "reranker_enabled": configuration["reranker_enabled"],
        "index": "ephemeral-private-runtime", "integrity_review": review_mode,
        "access_file": str(access_file), "url": "http://127.0.0.1:8002/"}))
    HTTPServer(("127.0.0.1", 8002), Handler).serve_forever()


def local_index_review_request():
    """Read-only, scoped request for an independent technical reviewer."""
    configure_process()
    verify_readonly_identity()
    from django.conf import settings
    private_folder(PRIVATE)
    with read_only_database():
        observed = _original_state()
    baseline = load_json(PRIVATE / "before.json")
    if (baseline.get("operational_counts") != observed["operational_counts"]
            or baseline.get("documents") != observed["documents"]):
        raise RuntimeError("Frozen company data or original document binding differs.")
    source = Path(settings.CHROMA_PERSIST_DIR).resolve()
    if file_manifest(source) != observed["index"]:
        raise RuntimeError("Production Chroma changed during read-only review.")
    return review_request(
        PRIVATE, expected=baseline["index"], production=observed["index"],
        private_copy=file_manifest(PRIVATE / "chroma_index_copy"),
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--audit"]:
        print(json.dumps(audit()))
    elif sys.argv[1:] == ["--index-review-request"]:
        print(json.dumps(local_index_review_request(), indent=2))
    else:
        main()
