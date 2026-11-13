"""Live local-pipeline evaluation of the customer RAG for assets 59, 60 and 61.

Reproduces the demo backend's own serving setup in one process, then asks the
question set in ``servy_three_asset_customer_eval`` through ``_answer()``:

* the SELECT-only database identity inside ``read_only_database()``;
* the same isolated BGE index the demo builds at every startup, created in a
  fresh private folder from the verified approved documents (the original
  Chroma index and all company records are only read);
* the strict serving gate: BGE dimensions, the company collection and real
  SHA-pinned V5 inference;
* real hybrid retrieval, V5 reranking, exhaustive scan, claim filtering,
  local Qwen generation through Ollama, source validation and citations.

Retrieval is observed through a pass-through recorder around ``ask_rag``; it is
never replaced. Nothing is written except the private report.

Usage (authorized Windows environment, Ollama running):
    set LOCALAPPDATA=<private demo LOCALAPPDATA>
    .\\.venv\\Scripts\\python.exe -B tools\\servy_three_asset_live_eval.py [--label before]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")


def _positives(chunks_by_doc, expected):
    from tools.servy_three_asset_rag_audit import _norm
    if not expected.get("doc"):
        return set()
    return {chunk.id for chunk in chunks_by_doc.get(expected["doc"], [])
            if any(_norm(p) in _norm(f"{chunk.heading or ''} {chunk.text}")
                   for p in expected["include"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    os.environ.update({
        "SERVY_LLM_PROVIDER": "extractive", "SERVY_ALLOW_REMOTE_LLM": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        # Same customer answer generation the launcher enables.
        "SERVY_CUSTOMER_ANSWER_GENERATION": "ollama",
        "SERVY_CUSTOMER_GENERATION_MODEL": os.environ.get(
            "SERVY_CUSTOMER_GENERATION_MODEL", "qwen3.5:2b-q4_K_M"),
        "SERVY_CUSTOMER_GENERATION_TIMEOUT": os.environ.get(
            "SERVY_CUSTOMER_GENERATION_TIMEOUT", "120"),
    })
    from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
    configure_process()
    identity = verify_readonly_identity()
    import django
    django.setup()
    import logging
    logging.disable(logging.WARNING)

    from django.conf import settings
    from core.models import KnowledgeChunk, Tenant
    from core.management.commands.check_rag_serving import verify_local_serving_artifacts
    from core.services import rag as rag_module
    from tools import servy_real_demo as demo
    from tools import servy_three_asset_customer_eval as evaluation
    from tools.servy_500_benchmark import (
        file_manifest, operational_counts, private_folder, read_only_database, serving,
    )
    from tools.servy_real_demo_source_rebuild import rebuild_isolated_bge_index
    from tools.servy_three_asset_rag_audit import _open_source_in_process
    from tools.servy_three_asset_scope import REQUIRED_MANUALS, readiness_failures

    if settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only":
        raise RuntimeError("Requires verified_real_only serving mode.")
    configuration = serving()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
    folder = private_folder(base / f"servy_three_asset_live_eval_{args.label}_{stamp}")
    original_index = Path(settings.CHROMA_PERSIST_DIR).resolve()
    original_manifest = file_manifest(original_index)

    rows = []
    with read_only_database():
        counts_before = operational_counts()
        readiness = demo._three_asset_readiness()
        failures = readiness_failures(readiness)
        if failures:
            raise RuntimeError("Three-asset manuals not ready: " + "; ".join(failures))
        rebuilt = rebuild_isolated_bge_index(folder, original_manifest, demo.ASSET_SCOPE, demo._asset)
        verified = verify_local_serving_artifacts(
            required_collections=("company-bge-small",), optional_collections=("bge-small",))
        demo.MANUAL_READINESS = readiness
        tenant = Tenant.objects.get(slug="skydream", is_active=True)

        chunks_by_doc, sources = {}, {}
        for asset in REQUIRED_MANUALS.values():
            for doc_id in asset["documents"]:
                chunks = list(KnowledgeChunk.objects.filter(
                    document_id=doc_id, tenant=tenant, is_quarantined=False,
                ).order_by("chunk_index", "id"))
                chunks_by_doc[doc_id] = chunks
                sources[doc_id] = " ".join(f"{c.heading or ''} {c.text or ''}" for c in chunks)
        doc_of_chunk = {c.id: d for d, cs in chunks_by_doc.items() for c in cs}
        heading_of_chunk = {c.id: c.heading or "" for cs in chunks_by_doc.values() for c in cs}

        # Pass-through recorder: the real ask_rag runs unchanged.
        real_ask_rag = rag_module.ask_rag
        calls = []

        def recording_ask_rag(*a, **kw):
            result = real_ask_rag(*a, **kw)
            retrieved = [item for item in (result.get("retrieved") or [])
                         if getattr(item, "source_kind", "knowledge_document") == "knowledge_document"]
            calls.append({
                "query_chars": len(str(kw.get("question") or "")),
                "top_k": kw.get("top_k"),
                "backend": result.get("retrieval_backend"),
                "reranker_applied": bool(result.get("reranker_applied")),
                "reranker_model": result.get("reranker_model"),
                "pre_rerank": [int(c) for c in result.get("pre_rerank_candidate_chunk_ids") or []],
                "final": [{"chunk": int(i.chunk_id), "doc": int(i.document_id),
                           "score": round(float(getattr(i, "score", 0) or 0), 4),
                           "rerank_logit": (round(float(i.rerank_logit), 3)
                                            if getattr(i, "rerank_logit", None) is not None else None)}
                          for i in retrieved if i.chunk_id is not None],
                "references": [int(r["chunk_id"]) for r in result.get("references") or []
                               if r.get("chunk_id") is not None],
            })
            return result

        rag_module.ask_rag = recording_ask_rag
        opener = SimpleNamespace(source=_open_source_in_process)
        try:
            for case_id, asset_id, manual, turns in evaluation.CASES:
                continuation, thread_id = None, "thread-" + uuid.uuid4().hex
                allowed_docs = set(REQUIRED_MANUALS[asset_id]["documents"])
                for index, expected in enumerate(turns, start=1):
                    calls.clear()
                    started = time.monotonic()
                    try:
                        data = demo._answer(asset_id, expected["question"],
                                            continuation=continuation, thread_id=thread_id)
                        status = 200
                    except ValueError:
                        data, status = {}, 400
                    elapsed = round(time.monotonic() - started, 2)
                    judged = evaluation.evaluate_turn(expected, status, data, asset_id=asset_id,
                                                      sources=sources, client=opener)
                    continuation = data.get("continuation") or continuation
                    positives = _positives(chunks_by_doc, expected)
                    pre = [c for call in calls for c in call["pre_rerank"]]
                    final = [f["chunk"] for call in calls for f in call["final"]]
                    refs = [c for call in calls for c in call["references"]]
                    retrieved_docs = {doc_of_chunk.get(c) for c in pre + final if c in doc_of_chunk}
                    cited = judged["cited_chunks"]
                    stage = None
                    if not judged["passed"]:
                        if expected["kind"] != evaluation.ANSWER:
                            stage = "WRONG_ANSWERABILITY_OR_SAFETY"
                        elif not positives:
                            stage = "EXPECTED_PASSAGE_NOT_INDEXED"
                        elif not set(pre) & positives and not set(final) & positives:
                            stage = "CANDIDATE_RECALL"
                        elif not set(refs) & positives:
                            stage = "RERANK_OR_REFERENCE_SELECTION"
                        elif not set(cited) & positives:
                            stage = "ANSWER_CONSTRUCTION_EVIDENCE_FILTER"
                        else:
                            stage = "ANSWER_CONTENT_OR_FOCUS"
                    generation = data.get("answer_generation") or {}
                    rows.append({
                        "case": case_id, "turn": index, "asset_id": asset_id, "manual": manual,
                        "question": expected["question"], "expected": expected["kind"],
                        "expected_section": expected.get("heading", ""),
                        "positive_chunks": sorted(positives),
                        "positive_sections": sorted({heading_of_chunk[c] for c in positives}),
                        "retrieval_calls": calls[:],
                        "first_positive_rank_pre_rerank": next(
                            (n + 1 for n, c in enumerate(pre) if c in positives), None),
                        "first_positive_rank_final": next(
                            (n + 1 for n, c in enumerate(final) if c in positives), None),
                        "retrieved_other_asset_documents": sorted(
                            d for d in retrieved_docs if d is not None and d not in allowed_docs),
                        "qwen": {k: generation.get(k) for k in (
                            "status", "eligible_claims", "accepted_claims", "rejected_claims",
                            "rejection_reasons", "applied_claims")},
                        "answer": judged, "failure_stage": stage, "seconds": elapsed,
                    })
                    mark = "PASS" if judged["passed"] else "FAIL " + str(stage)
                    print(f"[{asset_id}] {case_id}.{index} {mark:42s} qwen={generation.get('status')}"
                          f" {generation.get('accepted_claims', 0)}/{generation.get('eligible_claims', 0)}"
                          f" | {expected['question'][:60]}", flush=True)
        finally:
            rag_module.ask_rag = real_ask_rag
        counts_after = operational_counts()

    if counts_after != counts_before:
        raise RuntimeError("Operational record counts changed during the evaluation.")
    if file_manifest(original_index) != original_manifest:
        raise RuntimeError("Original Chroma index bytes changed during the evaluation.")

    def per_asset(asset_id):
        mine = [r for r in rows if r["asset_id"] == asset_id]
        documented = [r for r in mine if r["expected"] == evaluation.ANSWER]
        unsupported = [r for r in mine if r["expected"] in (evaluation.ABSTAIN,
                                                            evaluation.RESTRICTED_ABSTAIN)]
        with_claims = [r for r in mine if r["answer"]["claim_count"]]
        return {
            "documented_turns": len(documented),
            "documented_answered_correctly": sum(r["answer"]["passed"] for r in documented),
            "documented_failures": [
                {"case": r["case"], "turn": r["turn"], "stage": r["failure_stage"],
                 "failed_checks": r["answer"]["failed_checks"], "question": r["question"]}
                for r in documented if not r["answer"]["passed"]],
            "unsupported_turns": len(unsupported),
            "unsupported_correctly_refused": sum(r["answer"]["passed"] for r in unsupported),
            "safe_turns_passed": f"{sum(r['answer']['passed'] for r in mine if r['expected'] == evaluation.SAFE)}"
                                 f"/{sum(r['expected'] == evaluation.SAFE for r in mine)}",
            "turns_with_citations": len(with_claims),
            "citations_correct": sum(
                r["answer"]["checks"]["claims_verbatim_in_cited_manual"]
                and r["answer"]["checks"].get("citation_opens_to_the_claim", True)
                and r["answer"]["checks"]["no_cross_asset_citation"] for r in with_claims),
            "qwen_accepted": sum(r["qwen"]["accepted_claims"] or 0 for r in mine),
            "qwen_rejected": sum(r["qwen"]["rejected_claims"] or 0 for r in mine),
            "qwen_eligible": sum(r["qwen"]["eligible_claims"] or 0 for r in mine),
            "qwen_rejection_reasons": dict(sum((Counter(r["qwen"]["rejection_reasons"] or {})
                                                for r in mine), Counter())),
            "retrieval_other_asset_leaks": sum(bool(r["retrieved_other_asset_documents"]) for r in mine),
            "restricted_content_turns": sum(
                not r["answer"]["checks"]["no_restricted_or_procedural_content"] for r in mine),
        }

    summary = {
        "evaluation": "three-asset-live-local-pipeline-v1", "label": args.label,
        "index": rebuilt["mode"], "indexed_chunks": rebuilt["verified_chunks"],
        "serving": {"configuration": {k: v for k, v in configuration.items()
                                      if k != "expected_v5_sha256"}, "verified": verified},
        "database_account": identity.get("account"),
        "cases": len(evaluation.CASES), "turns": len(rows),
        "per_asset": {str(a): per_asset(a) for a in REQUIRED_MANUALS},
        "integrity": {"original_index_unchanged": True, "operational_counts_unchanged": True},
        "tickets_created": 0,
    }
    (folder / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                        encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("REPORT", folder / "report.json", flush=True)


if __name__ == "__main__":
    main()
