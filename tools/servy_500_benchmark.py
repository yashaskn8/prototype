"""Read-only, resumable evaluation of the authorized Servy 500-question bank.

The runner uses the real ask_rag service and an unsaved customer-answer contract.
It never invokes the customer POST API or writes company models, indexes or files.
Results and source snapshots must live beneath LOCALAPPDATA/Servy/private_evaluations.
"""
import argparse
import contextlib
import csv
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
TENANT = "skydream"
AUTHORIZED_ASSETS = frozenset({59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70})
AUTHORIZATION = (
    "User confirmed all 11 listed skydream assets for the new question bank's "
    "baseline and post-change retest on 2026-10-30; company data and indexes read-only."
)
SERVING_FILES = (
    "core/services/rag.py", "core/services/retriever.py", "core/services/llm.py",
    "core/services/customer_answer.py", "core/services/customer_diagnostic.py",
    "core/services/customer_isolation.py", "core/services/vectors.py",
    "core/services/content_safety.py",
    "core/ml/conditional_rerank.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_manifest(directory):
    output = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            output[str(path.relative_to(directory))] = sha(path)
    return output


def prepare_index_copy(folder):
    """Use a byte-identical private copy because Chroma rewrites files on read."""
    from django.conf import settings
    source = Path(settings.CHROMA_PERSIST_DIR).resolve()
    target = folder / "chroma_index_copy"
    if not source.is_dir() or target.exists():
        raise RuntimeError("Expected existing production Chroma index and a fresh private copy path.")
    original = file_manifest(source)
    shutil.copytree(source, target)
    if file_manifest(source) != original or file_manifest(target) != original:
        raise RuntimeError("Index changed during copy or private copy differs; no queries run.")
    write_json(folder / "production_index_manifest.json", original)
    return source, target


def activate_index_copy(folder):
    from django.conf import settings
    from core.services import chroma_store
    if chroma_store._chroma_client is not None:
        raise RuntimeError("Chroma client was opened before the private index copy was selected.")
    source = Path(settings.CHROMA_PERSIST_DIR).resolve()
    target = (folder / "chroma_index_copy").resolve()
    if not target.is_dir() or target.is_relative_to(source):
        raise RuntimeError("Private evaluation index copy is unavailable.")
    if file_manifest(source) != load_json(folder / "production_index_manifest.json"):
        raise RuntimeError("Original company index differs from frozen manifest; no queries run.")
    settings.CHROMA_PERSIST_DIR = target  # Process-local only; normal serving is unchanged.


def private_folder(path):
    if os.name != "nt":
        raise RuntimeError("Run in the authorized local Windows environment.")
    base = (Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations").resolve()
    path = path.resolve()
    if not path.is_relative_to(base) or path == base:
        raise RuntimeError("Results must be in a new private directory below LOCALAPPDATA/Servy/private_evaluations.")
    path.mkdir(parents=True, exist_ok=True)
    output = subprocess.check_output(
        ["whoami", "/user", "/fo", "csv", "/nh"], text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    sid = next(csv.reader(io.StringIO(output.strip())))[1]
    subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"],
        check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return path


def sql_guard(execute, sql, params, many, context):
    statement = re.sub(r"/\*.*?\*/", "", sql, flags=re.S).lstrip().upper()
    if context["connection"].alias != "default" or not re.match(
        r"^(SELECT|SHOW|EXPLAIN|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE SAVEPOINT)\b", statement
    ):
        raise RuntimeError("Non-read-only SQL blocked by benchmark.")
    if re.search(r"\b(?:INTO\s+(?:OUTFILE|DUMPFILE)|FOR\s+UPDATE|GET_LOCK|RELEASE_LOCK|LOAD_FILE)\b", statement):
        raise RuntimeError("Side-effect SQL blocked by benchmark.")
    return execute(sql, params, many, context)


@contextlib.contextmanager
def read_only_database():
    from django.db import connection, connections
    connection.ensure_connection()
    if connection.vendor != "mysql":
        raise RuntimeError("Existing MySQL connection required.")
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
    with contextlib.ExitStack() as stack:
        for conn in connections.all():
            stack.enter_context(conn.execute_wrapper(sql_guard))
        yield
    connection.close()


def serving():
    from django.conf import settings
    from core.services.retriever import get_serving_config
    from core.management.commands.check_rag_serving import serving_configuration_issues
    config = get_serving_config()
    problems = serving_configuration_issues(config, settings)
    if problems or settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only":
        raise RuntimeError("Required existing verified-real BGE-small/hybrid/V5 serving is not ready.")
    if settings.SERVY_ALLOW_REMOTE_LLM or urlparse(settings.SERVY_OLLAMA_URL).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError("Remote LLM is outside this authorization.")
    if config.embedding_profile != "bge-small" or config.retrieval_mode != "hybrid" or not config.reranker_enabled or settings.SERVY_RERANKER_POOL_SIZE != 5:
        raise RuntimeError("Expected BGE-small hybrid retrieval with V5 pool size 5.")
    return {"embedding": config.embedding_profile, "retrieval": config.retrieval_mode,
            "reranker_enabled": config.reranker_enabled, "reranker_pool_size": settings.SERVY_RERANKER_POOL_SIZE,
            "expected_v5_sha256": settings.SERVY_RERANKER_EXPECTED_SHA256,
            "data_mode": settings.SERVY_CUSTOMER_DATA_MODE, "llm_provider": settings.SERVY_LLM_PROVIDER}


def validate_bank(path):
    bank = load_json(path)
    rows = bank.get("questions")
    if not isinstance(rows, list) or len(rows) != 500:
        raise RuntimeError("Expected exactly 500 original question rows.")
    required = {"id", "asset_id", "product", "module", "intent", "question", "kb_coverage",
                "ground_truth_status", "expected_handling", "run_permission"}
    ids = set()
    quality = []
    for row in rows:
        missing = sorted(required - row.keys())
        issues = []
        if missing:
            issues.append("missing_fields:" + ",".join(missing))
        if row.get("id") in ids:
            issues.append("duplicate_id")
        ids.add(row.get("id"))
        question = row.get("question", "")
        if not isinstance(question, str) or not question.strip():
            issues.append("empty_question")
        elif len(question) > 1000:
            issues.append("exceeds_customer_api_length")
        if row.get("question_char_count") != len(question):
            issues.append("length_metadata_mismatch")
        if row.get("question_sha256") != hashlib.sha256(question.encode("utf-8")).hexdigest():
            issues.append("hash_metadata_mismatch")
        if row.get("asset_id") not in AUTHORIZED_ASSETS:
            issues.append("outside_explicit_asset_scope")
        quality.append({"id": row.get("id"), "issues": issues, "review_status": "PENDING_REVIEW"})
    if any(item["issues"] for item in quality):
        raise RuntimeError("Question bank integrity/scope failed; inspect the private quality report.")
    return bank, quality


def assets_and_sources(rows):
    from core.models import Tenant, Asset
    from core.services.customer_isolation import is_verified_real_asset, is_verified_real_customer, validate_customer_serving_policy
    from core.services.retriever import _knowledge_candidates
    from core.services.customer_answer import source_chunk
    tenant = Tenant.objects.get(slug=TENANT)
    assets = {}
    coverage = {}
    sources = {}
    for asset_id in sorted({row["asset_id"] for row in rows}):
        try:
            asset = Asset.objects.select_related("tenant", "customer", "product", "site").get(id=asset_id, tenant=tenant)
        except Asset.DoesNotExist:
            coverage[str(asset_id)] = {"status": "NOT_RUN_UNAUTHORIZED", "reason": "asset_not_in_authorized_tenant"}
            continue
        if not asset.customer:
            coverage[str(asset_id)] = {"status": "NOT_RUN_UNAUTHORIZED", "reason": "no_customer_binding"}
            continue
        if not is_verified_real_asset(asset) or not is_verified_real_customer(asset.customer):
            coverage[str(asset_id)] = {"status": "NOT_RUN_UNAUTHORIZED", "reason": "unverified_asset_or_customer"}
            continue
        try:
            validate_customer_serving_policy(tenant, asset.customer, asset)
        except Exception as exc:
            coverage[str(asset_id)] = {"status": "NOT_RUN_UNAUTHORIZED", "reason": "customer_serving_policy_rejected",
                                       "exception_type": type(exc).__name__}
            continue
        assets[asset_id] = asset
        eligible = []
        for candidate in _knowledge_candidates(tenant, asset, asset.customer, False, backend="all"):
            try:
                chunk = source_chunk(tenant=tenant, customer=asset.customer, asset=asset,
                                     document_id=candidate.document_id, chunk_id=candidate.id)
            except Exception:
                continue
            eligible.append({"document_id": chunk.document_id, "chunk_id": chunk.id,
                             "title": chunk.document.title, "heading": chunk.heading,
                             "text": chunk.text, "index_version": chunk.document.index_version,
                             "sha256": hashlib.sha256(chunk.text.encode()).hexdigest()})
        eligible.sort(key=lambda c: (c["document_id"], c["chunk_id"]))
        sources[str(asset_id)] = eligible
        from core.services.rag import has_authorized_original_documentation
        original_documentation = bool(eligible) and has_authorized_original_documentation(
            tenant=tenant, customer=asset.customer, asset=asset)
        coverage[str(asset_id)] = {"status": "AUTHORIZED_VERIFIED_BINDING",
                                    "customer_id": asset.customer_id, "asset_code": asset.asset_code,
                                    "site_id": asset.site_id, "document_count": len({c["document_id"] for c in eligible}),
                                    "chunk_count": len(eligible),
                                    "original_documentation_eligible": original_documentation}
    return tenant, assets, coverage, sources


def split_rows(rows):
    # Frozen before inference. The holdout is never inspected for model changes.
    groups = defaultdict(list)
    for row in rows:
        groups[row["asset_id"]].append(row)
    holdout = set()
    for key, group in groups.items():
        ranked = sorted(group, key=lambda r: hashlib.sha256(("servy500-holdout-v1:" + r["id"]).encode()).hexdigest())
        holdout.update(r["id"] for r in ranked[:round(len(group) * .2)])
    return {"development": [r["id"] for r in rows if r["id"] not in holdout],
            "holdout": [r["id"] for r in rows if r["id"] in holdout]}


def admission_plan(rows, coverage):
    """Exclude source gaps from answerable evaluation; sample negative cases."""
    documented = []
    gaps = defaultdict(list)
    unauthorized = []
    for row in rows:
        state = coverage[str(row["asset_id"])]
        if state["status"] != "AUTHORIZED_VERIFIED_BINDING":
            unauthorized.append(row["id"])
        elif state.get("original_documentation_eligible"):
            documented.append(row["id"])
        else:
            gaps[row["asset_id"]].append(row["id"])
    negatives = []
    for ids in gaps.values():
        ranked = sorted(ids, key=lambda id_: hashlib.sha256(("servy-negative-v1:" + id_).encode()).hexdigest())
        negatives.extend(ranked[:2])
    return {"documented_candidate_ids": documented,
            "coverage_gap_ids": [id_ for ids in gaps.values() for id_ in ids],
            "negative_suite_ids": negatives, "unauthorized_ids": unauthorized,
            "expert_approved_answerable_ids": []}


def question_audit(folder, dataset):
    bank, quality = validate_bank(dataset)
    write_json(folder / "question_quality_review.json", quality)
    rows = bank["questions"]
    split = split_rows(rows)
    # The corrected version is identical until a human-supported question fix exists.
    write_json(folder / "servy_500_corrected_questions_v1.json", bank)
    write_json(folder / "question_change_log.json", {"changes": [], "reason": "No verified corrections; original question text preserved."})
    write_json(folder / "split.json", split)
    return bank, split


def prepare(folder, dataset):
    if (folder / "freeze.json").exists():
        raise RuntimeError("Run directory already frozen; use a new directory or resume.")
    bank, split = question_audit(folder, dataset)
    rows = bank["questions"]
    _, assets, coverage, sources = assets_and_sources(rows)
    source_index, copied_index = prepare_index_copy(folder)
    admission = admission_plan(rows, coverage)
    write_json(folder / "admission.json", admission)
    write_json(folder / "authorized_source_snapshot.json", sources)
    write_json(folder / "source_review_candidates.json", [
        {"id": r["id"], "asset_id": r["asset_id"], "customer_id": coverage[str(r["asset_id"])].get("customer_id"),
         "expected_response": None, "supporting_document_ids": [], "supporting_chunk_ids": [],
         "alternative_chunk_ids": [], "evidence_sufficiency": "NOT_VERIFIED", "expert_review_status": "PENDING_REVIEW"}
        for r in rows])
    inputs = ["servy_500_corrected_questions_v1.json", "question_change_log.json", "authorized_source_snapshot.json",
              "source_review_candidates.json", "split.json", "question_quality_review.json", "admission.json"]
    freeze = {"authorization": AUTHORIZATION, "original_dataset_sha256": sha(dataset),
              "original_dataset_path": str(dataset), "tenant": TENANT, "scope": coverage,
              "serving": serving(), "input_sha256": {name: sha(folder / name) for name in inputs},
              "production_index_path": str(source_index), "evaluation_index_path": str(copied_index),
              "production_index_manifest_sha256": sha(folder / "production_index_manifest.json"),
              "baseline_code_sha256": {name: sha(ROOT / name) for name in SERVING_FILES},
              "created_at": time.time()}
    write_json(folder / "freeze.json", freeze)
    eligible = sum(r["asset_id"] in assets for r in rows)
    print(json.dumps({"phase": "dry-run", "questions": len(rows), "development": len(split["development"]),
                      "holdout": len(split["holdout"]), "assets": coverage, "eligible": eligible,
                      "documented_candidates": len(admission["documented_candidate_ids"]),
                      "coverage_gaps": len(admission["coverage_gap_ids"]),
                      "future_negative_suite": len(admission["negative_suite_ids"]),
                      "human_ground_truth": "PENDING_REVIEW", "company_writes": False}), flush=True)


def check_freeze(folder, phase):
    frozen = load_json(folder / "freeze.json")
    for name, expected in frozen["input_sha256"].items():
        if sha(folder / name) != expected:
            raise RuntimeError("Frozen evaluation input changed: " + name)
    if serving() != frozen["serving"]:
        raise RuntimeError("Serving configuration changed since dry-run.")
    if phase == "baseline":
        for name, expected in frozen["baseline_code_sha256"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Serving code changed before baseline ended: " + name)
    return frozen


def check_retest_sources(folder, frozen, coverage, sources, phase):
    """Permit only newly eligible chunks of unchanged baseline documents."""
    original = load_json(folder / "authorized_source_snapshot.json")
    for asset_id, old_scope in frozen["scope"].items():
        new_scope = coverage[asset_id]
        for field in ("status", "reason", "customer_id", "asset_code", "site_id", "document_count"):
            if new_scope.get(field) != old_scope.get(field):
                raise RuntimeError("Customer, asset, or document scope changed before retest.")
        old_rows = {(r["document_id"], r["chunk_id"]): r for r in original.get(asset_id, [])}
        new_rows = {(r["document_id"], r["chunk_id"]): r for r in sources.get(asset_id, [])}
        if any(new_rows.get(key) != row for key, row in old_rows.items()):
            raise RuntimeError("A baseline source chunk changed or disappeared before retest.")
        old_docs = {doc_id for doc_id, _ in old_rows}
        if any(doc_id not in old_docs for doc_id, _ in new_rows):
            raise RuntimeError("Retest discovered a new document outside frozen baseline scope.")
    tag = "final" if phase.endswith("final") else "retest2" if phase.endswith("retest2") else "retest"
    snapshot_file = folder / f"{tag}_source_snapshot.json"
    if snapshot_file.exists():
        if load_json(snapshot_file) != sources:
            raise RuntimeError("Retest source eligibility changed during evaluation.")
    else:
        write_json(snapshot_file, sources)
    code = {name: sha(ROOT / name) for name in SERVING_FILES}
    code_file = folder / f"{tag}_code_sha256.json"
    if code_file.exists():
        if load_json(code_file) != code:
            raise RuntimeError("Serving code changed during retest.")
    else:
        write_json(code_file, code)


def serial(value):
    if dataclasses.is_dataclass(value):
        return serial(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unexpected result type: " + type(value).__name__)


def classify(record):
    if record["status"] != "COMPLETED":
        return "ERROR", "Query or response construction failed."
    response = record.get("formatted") or {}
    response_type = response.get("response_type")
    mode = response.get("response_mode")
    answer = (response.get("answer") or "").strip()
    if not answer:
        return "ERROR", "No usable customer-facing answer."
    if response_type == "CLARIFICATION" or mode == "CLARIFICATION_REQUIRED":
        return "CLARIFICATION", "Customer contract requested information before answering."
    if mode == "IMMEDIATE_SAFETY_ESCALATION":
        return "REFUSED", "Customer contract withheld ordinary guidance for a safety hazard."
    if response_type in ("INSUFFICIENT_EVIDENCE", "POLICY_BLOCKED") or mode in ("INSUFFICIENT_EVIDENCE", "POLICY_BLOCKED"):
        return "REFUSED", "Customer contract withheld unsupported or policy-blocked guidance."
    if (response.get("presentation") or {}).get("sections"):
        # A four-point structure does not establish full or partial factual correctness.
        return "ANSWERED", "Customer contract provided substantive sections; correctness is NOT_VERIFIED."
    return "NEEDS_HUMAN_REVIEW", "Response shape does not reliably identify answer, refusal, or partial answer."


def run_one(row, asset):
    from core.models import CustomerInteraction
    from core.services.rag import ask_rag
    from core.services.customer_diagnostic import (build_structured_diagnostic_contract,
                                                   check_for_safety_hazard, check_for_vague_complaint)
    question = row["question"]
    hazard, warning = check_for_safety_hazard(question)
    vague, clarification, options = check_for_vague_complaint(question, asset)
    record = {"id": row["id"], "asset_id": asset.id, "customer_id": asset.customer_id,
              "question": question, "status": "ERROR", "authorization": "AUTHORIZED_VERIFIED_BINDING",
              "raw": None, "formatted": None, "evaluation_route": "direct service and unsaved customer formatter; no HTTP login"}
    start = time.monotonic()
    try:
        if not hazard and not vague:
            raw = ask_rag(tenant=asset.tenant, customer=asset.customer, asset=asset,
                          question=question, staff_mode=False, include_confidential=False,
                          service_call=None, top_k=5)
            record["raw"] = serial(raw)
        else:
            raw = {"answer": "", "references": [], "engine": "pre_rag_customer_guard"}
            record["raw"] = raw
        interaction = CustomerInteraction(tenant=asset.tenant, customer=asset.customer,
                                          asset=asset, site=asset.site, question=question,
                                          answer=raw.get("answer", ""), source_refs=raw.get("references", []))
        formatted = build_structured_diagnostic_contract(
            interaction=interaction, session=None, raw_answer=raw.get("answer", ""),
            retrieved_references=raw.get("references", []),
            clarification_question=clarification if vague and not hazard else None,
            clarification_options=options if vague and not hazard else None,
            is_hazard=hazard, hazard_warning=warning,
        )
        record["formatted"] = serial(formatted)
        record["status"] = "COMPLETED"
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)  # Private directory only.
    record["latency_seconds"] = round(time.monotonic() - start, 3)
    record["classification"], record["classification_reason"] = classify(record)
    record["quality"] = {"technical_correctness": "NOT_VERIFIED", "citation_semantics": "NOT_VERIFIED",
                         "appropriate_refusal": "NOT_VERIFIED", "unsupported_answer": "NOT_VERIFIED"}
    return record


def summarize(folder, phase, rows):
    counts = Counter(r["classification"] for r in rows)
    by_asset = defaultdict(Counter)
    by_module = defaultdict(Counter)
    by_intent = defaultdict(Counter)
    for row in rows:
        by_asset[str(row["asset_id"])][row["classification"]] += 1
        by_module[row["module"]][row["classification"]] += 1
        by_intent[row["intent"]][row["classification"]] += 1
    executed = sum(not r["classification"].startswith("NOT_RUN") for r in rows)
    summary = {"phase": phase, "total_questions": 500, "selected_questions": len(rows),
               "executed_questions": executed, "not_run_questions": len(rows) - executed,
               "classification_counts": dict(counts), "denominator_for_rates": executed,
               "answer_rate": counts["ANSWERED"] / executed if executed else None,
               "refusal_rate": counts["REFUSED"] / executed if executed else None,
               "by_asset": {k: dict(v) for k, v in by_asset.items()},
               "by_module": {k: dict(v) for k, v in by_module.items()},
               "by_intent": {k: dict(v) for k, v in by_intent.items()},
               "outcome_ids": {key: [r["id"] for r in rows if r["classification"] == key] for key in counts},
               "verified_answer_accuracy": None, "verified_unsupported_answer_count": None,
               "verified_inappropriate_refusal_count": None,
               "source_review_status": "PENDING_REVIEW", "http_customer_portal_tested": False}
    write_json(folder / f"servy_500_{phase}_summary.json", summary)
    return summary


def execute(folder, dataset, phase, max_queries=None):
    frozen = check_freeze(folder, phase)
    activate_index_copy(folder)
    if sha(dataset) != frozen["original_dataset_sha256"]:
        raise RuntimeError("Original dataset bytes changed since dry-run.")
    bank, _ = validate_bank(dataset)
    _, assets, coverage, sources = assets_and_sources(bank["questions"])
    if phase in ("retest", "holdout-retest", "retest2", "holdout-retest2", "final", "holdout-final"):
        check_retest_sources(folder, frozen, coverage, sources, phase)
    elif coverage != frozen["scope"] or sources != load_json(folder / "authorized_source_snapshot.json"):
        raise RuntimeError("Customer/asset/document scope changed since dry-run; stop before queries.")
    split = load_json(folder / "split.json")
    admission = load_json(folder / "admission.json") if "admission.json" in frozen["input_sha256"] else None
    coverage_gap_ids = set(admission["coverage_gap_ids"]) if admission else set()
    negative_ids = set(admission["negative_suite_ids"]) if admission else set()
    selected = set(split["holdout" if phase.startswith("holdout") else "development"])
    target = folder / f"servy_500_{phase}_actual_test_results.json"
    results = load_json(target) if target.exists() else []
    finished = {r["id"] for r in results}
    if len(finished) != len(results) or not finished.issubset(selected):
        raise RuntimeError("Existing results have duplicate or out-of-split IDs.")
    before = operational_counts()
    queries_this_invocation = 0
    for row in bank["questions"]:
        if row["id"] not in selected or row["id"] in finished:
            continue
        check_freeze(folder, phase)
        if row["asset_id"] not in assets:
            outcome = {"id": row["id"], "asset_id": row["asset_id"], "customer_id": None,
                       "question": row["question"], "status": "NOT_RUN_UNAUTHORIZED",
                       "authorization": frozen["scope"][str(row["asset_id"])]["reason"],
                       "classification": "NOT_RUN_UNAUTHORIZED",
                       "classification_reason": "Verified customer/asset binding failed; no chatbot call.",
                       "raw": None, "formatted": None, "latency_seconds": 0,
                       "quality": {"technical_correctness": "NOT_VERIFIED", "citation_semantics": "NOT_VERIFIED"}}
            results.append(outcome)
            write_json(target, results)
            finished.add(row["id"])
            continue
        if row["id"] in coverage_gap_ids and row["id"] not in negative_ids:
            results.append({"id": row["id"], "asset_id": row["asset_id"],
                            "customer_id": assets[row["asset_id"]].customer_id,
                            "question": row["question"], "status": "NOT_RUN_COVERAGE_GAP",
                            "authorization": "AUTHORIZED_NO_ORIGINAL_DOCUMENTATION",
                            "classification": "NOT_RUN_COVERAGE_GAP",
                            "classification_reason": "Excluded from answerable evaluation; only sampled negatives run.",
                            "raw": None, "formatted": None, "latency_seconds": 0})
            write_json(target, results)
            finished.add(row["id"])
            continue
        if max_queries is not None and queries_this_invocation >= max_queries:
            break
        # An intent file prevents an automatic replay after an interrupted in-flight call.
        intent = folder / "inflight.json"
        if intent.exists():
            raise RuntimeError("An in-flight query needs manual review before resume; no automatic duplicate calls.")
        write_json(intent, {"id": row["id"], "phase": phase, "started_at": time.time()})
        outcome = run_one(row, assets[row["asset_id"]])
        results.append(outcome)
        write_json(target, results)
        intent.unlink()
        finished.add(row["id"])
        queries_this_invocation += 1
        print(json.dumps({"completed": len(results), "selected": len(selected), "id": row["id"],
                          "classification": outcome["classification"], "seconds": outcome["latency_seconds"]}), flush=True)
        if operational_counts() != before:
            raise RuntimeError("Operational record counts changed; inspect concurrent activity before resuming.")
    result_rows = [{**next(r for r in bank["questions"] if r["id"] == x["id"]), **x} for x in results]
    summary = summarize(folder, phase, result_rows)
    print(json.dumps({"phase": phase, "completed": len(results), "counts": summary["classification_counts"]}), flush=True)


def operational_counts():
    from core.models import (ServiceCall, CustomerInteraction, CustomerConversation,
                             DiagnosticSession, CustomerMessage, RecoveryPassport)
    return {model.__name__: model.objects.count() for model in (
        ServiceCall, CustomerInteraction, CustomerConversation, DiagnosticSession,
        CustomerMessage, RecoveryPassport)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("question-audit", "dry-run", "baseline", "retest", "holdout", "holdout-retest", "retest2", "holdout-retest2", "final", "holdout-final"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Stop this invocation after N new live queries; results remain resumable.")
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    folder = private_folder(args.data_dir)
    if args.phase == "question-audit":
        bank, split = question_audit(folder, args.dataset)
        print(json.dumps({"phase": "question-audit", "questions": len(bank["questions"]),
                          "development": len(split["development"]), "holdout": len(split["holdout"]),
                          "ground_truth": "PENDING_REVIEW", "live_queries": 0}), flush=True)
        return
    import django
    django.setup()
    serving()
    with read_only_database():
        if args.phase == "dry-run":
            prepare(folder, args.dataset)
        else:
            execute(folder, args.dataset, args.phase, args.max_queries)


if __name__ == "__main__":
    main()
