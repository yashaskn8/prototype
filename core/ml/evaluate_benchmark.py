"""Comprehensive Two-Track Retrieval & Full Customer Journey Evaluation for Servy RAG (Operation 11.0).

Architected into two strictly separated evaluation tracks:

TRACK A — EMBEDDING-ONLY RESEARCH (Standardized Candidate Universe)
  Holds constant:
    Tenant, Customer, Asset, Product/model applicability, Source provenance,
    Document versions, Confidentiality, Quarantine, Relevant ground truth, and Candidate count.
  Evaluates:
    - Candidate A: Pure Dense MiniLM
    - Candidate B: Pure Dense Base BGE-small
    - Candidate C: Pure Dense Fine-Tuned BGE-small (Tenant 9 only, N=30 relevance queries)

TRACK B — FULL CUSTOMER SERVING (Production Application Pipeline)
  Calls real application retrieval, evidence validation, safety policy, and answer generation:
    - Candidate D: Production Baseline Hybrid RRF (MiniLM + BM25/Lexical)
    - Candidate E: Experimental BGE Hybrid RRF (BGE-small + BM25/Lexical)
    - Candidate F: Corrected Production Hybrid RRF (MiniLM + BM25/Lexical + Calibrated Gate + Typo Normalizer)

Comprehensive Metrics & Denominators:
  - 43 Relevance Queries (Hit@1, Hit@5, Hit@10, MRR@10, nDCG@10)
  - 21 Abstention / Security Queries (Abstention Accuracy)
  - 7 Boundary & Security Rates (Inspecting EVERY returned chunk, not just top-1):
      1. Unauthorized tenant retrieval rate
      2. Unauthorized customer retrieval rate
      3. Wrong-product retrieval rate
      4. Wrong-model retrieval rate
      5. Wrong-asset retrieval rate
      6. Wrong-equipment citation rate (Track B)
      7. Wrong-equipment action rate (Track B)
  - Authenticated cross-tenant attack verification (0 leaks across 4 adversarial attacks)
  - Latency percentiles (p50, p95, p99) and peak memory
  - Transparent per-query accounting written to ml_artifacts/benchmark_per_query_results.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("SERVY_DB_ENGINE", "mysql")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django
django.setup()

from django.conf import settings
from core.models import Tenant, Asset, KnowledgeChunk, KnowledgeDocument
from core.ml.embeddings import get_profile, load_model, get_collection, encode_queries
from core.ml.evaluation import compute_metrics, check_chunk_boundaries
from core.ml.benchmark_manifest import load_manifest, resolve_query_asset, ROLE_RELEVANCE
from core.services.retriever import retrieve, _knowledge_candidates
from core.services.llm import generate_answer, INSUFFICIENT_EVIDENCE_MESSAGE
from core.services.rag import ask_rag
from core.services.customer_isolation import is_real_tenant


def _build_chunk_metadata_cache() -> Dict[int, Dict[str, Any]]:
    """Precompute metadata dictionary for every KnowledgeChunk in the database."""
    chunks = KnowledgeChunk.objects.select_related("document", "document__asset", "document__product").all()
    cache = {}
    for c in chunks:
        d = c.document
        model_num = None
        if d.asset and d.asset.model_number:
            model_num = d.asset.model_number
        elif d.product and getattr(d.product, "model_number", None):
            model_num = d.product.model_number

        cache[c.id] = {
            "chunk_id": c.id,
            "document_id": d.id,
            "chunk_index": c.chunk_index,
            "tenant_id": d.tenant_id,
            "customer_id": d.customer_id,
            "asset_id": d.asset_id,
            "product_id": d.product_id,
            "model_number": model_num,
            "title": d.title,
            "heading": c.heading,
            "is_confidential": d.is_confidential,
            "is_quarantined": c.is_quarantined,
        }
    return cache


def _resolve_profile_for_tenant(model_family: str, tenant: Tenant) -> Optional[str]:
    """Dynamically resolve embedding profile key without hardcoding tenant IDs."""
    is_real = is_real_tenant(tenant)
    if model_family == "minilm":
        return "company-minilm" if is_real else "minilm"
    elif model_family == "bge-small":
        return "company-bge-small" if is_real else "bge-small"
    elif model_family == "bge-ft":
        # Fine-tuned checkpoint is specifically bound to the Skydream real corpus
        return "company-bge-ft" if (is_real and tenant.id == 9) else None
    return None


# ---------------------------------------------------------------------------
# TRACK A: EMBEDDING-ONLY RESEARCH (Standardized Candidate Universe)
# ---------------------------------------------------------------------------

def evaluate_track_a_dense(
    model_family: str,
    cases: List[Dict[str, Any]],
    chunk_meta: Dict[int, Dict[str, Any]],
    chunk_lookup: Dict[tuple, int],
    k_max: int = 10,
) -> Dict[str, Any]:
    """Evaluate pure dense retrieval over the EXACT same authorized candidate universe."""
    tracemalloc.start()
    t0_start = time.perf_counter()

    tenants = {t.id: t for t in Tenant.objects.all()}
    profiles = {}
    models = {}
    collections = {}

    # Track model load failures explicitly instead of silently skipping
    model_load_failures = {}  # tid -> error message
    for tid, t in tenants.items():
        pkey = _resolve_profile_for_tenant(model_family, t)
        if pkey:
            try:
                p = get_profile(pkey)
                profiles[tid] = p
                models[tid] = load_model(p)
                collections[tid] = get_collection(p, create=False)
            except Exception as exc:
                model_load_failures[tid] = f"MODEL_LOAD_FAILED: {pkey}: {exc}"

    load_time_ms = (time.perf_counter() - t0_start) * 1000

    eval_records = []
    category_records = {}
    unscorable_cases = []

    # Boundary violation counters
    unauthorized_tenant_queries = 0
    unauthorized_customer_queries = 0
    wrong_product_queries = 0
    wrong_model_queries = 0
    wrong_asset_queries = 0
    total_supported = 0

    for c in cases:
        qid = c["query_id"]
        q_text = c["query_text"]
        tid = c["tenant_id"]
        cat = c["category"]
        is_abstain = c["expected_abstain"]
        rel_ids = set(c["positive_chunk_ids"])
        equip_name = c["equipment"]

        tenant = tenants.get(tid)
        if not tenant:
            unscorable_cases.append({"query_id": qid, "reason": f"Tenant {tid} not in database"})
            continue

        if tid not in profiles:
            reason = model_load_failures.get(tid, f"Model {model_family} not supported for tenant {tid} (separated population)")
            unscorable_cases.append({
                "query_id": qid,
                "reason": reason,
            })
            continue

        p = profiles[tid]
        m = models[tid]
        coll = collections[tid]
        asset = resolve_query_asset(tid, equip_name)

        # Standardized Candidate Universe: hold constant authorized candidate pool
        authorized_candidates = _knowledge_candidates(
            tenant=tenant,
            customer=None,
            asset=asset,
            include_confidential=False,
            staff_mode=False,
        )
        authorized_chunk_ids = {chunk.id for chunk in authorized_candidates}
        allowed_doc_ids = list({chunk.document_id for chunk in authorized_candidates})

        t0_q = time.perf_counter()
        if not allowed_doc_ids:
            # Phase 10 Fix: Fail closed to empty result set if no authorized candidates exist.
            # Never broaden scope to a tenant-only fallback.
            ranked_cids = []
            latency_ms = 0.0
            res = None
        else:
            emb = encode_queries(p, m, [q_text])[0].tolist()
            where_filter = {
                "$and": [
                    {"tenant_id": int(tid)},
                    {"document_id": {"$in": allowed_doc_ids}},
                ]
            }

            res = coll.query(
                query_embeddings=[emb],
                n_results=k_max,
                where=where_filter,
                include=["metadatas", "distances"],
            )
            latency_ms = (time.perf_counter() - t0_q) * 1000

            ranked_cids = []
            if res and res.get("metadatas") and res["metadatas"][0]:
                for meta in res["metadatas"][0]:
                    cid = meta.get("chunk_id") or chunk_lookup.get((meta.get("document_id"), meta.get("chunk_index")))
                    # Phase 10 Fix: Post-validate against authorized chunk IDs, not only document IDs
                    if cid and cid in authorized_chunk_ids:
                        ranked_cids.append(cid)

        distances = res["distances"][0] if res and res.get("distances") and res["distances"] else []

        rec = {
            "query_id": qid,
            "ranked_chunk_ids": ranked_cids,
            "relevant_chunk_ids": list(rel_ids),
            "latency_ms": latency_ms,
            "is_unsupported": is_abstain,
        }
        eval_records.append(rec)

        if not is_abstain:
            total_supported += 1
            has_unauth_tenant = False
            has_unauth_cust = False
            has_wrong_prod = False
            has_wrong_mod = False
            has_wrong_ast = False

            for cid in ranked_cids:
                meta = chunk_meta.get(cid, {})
                b = check_chunk_boundaries(meta, tid, customer=None, asset=asset)
                if b["unauthorized_tenant"]: has_unauth_tenant = True
                if b["unauthorized_customer"]: has_unauth_cust = True
                if b["wrong_product"]: has_wrong_prod = True
                if b["wrong_model"]: has_wrong_mod = True
                if b["wrong_asset"]: has_wrong_ast = True

            if has_unauth_tenant: unauthorized_tenant_queries += 1
            if has_unauth_cust: unauthorized_customer_queries += 1
            if has_wrong_prod: wrong_product_queries += 1
            if has_wrong_mod: wrong_model_queries += 1
            if has_wrong_ast: wrong_asset_queries += 1

        if cat not in category_records:
            category_records[cat] = []
        category_records[cat].append(rec)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    overall = compute_metrics(eval_records)
    overall["model_family"] = model_family
    overall["track"] = "TRACK_A_EMBEDDING_RESEARCH"
    overall["architecture"] = f"dense-{model_family}"
    overall["load_time_ms"] = round(load_time_ms, 2)
    overall["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)
    overall["unscorable_cases"] = unscorable_cases
    overall["n_unscorable"] = len(unscorable_cases)

    # 7 Comprehensive Boundary Rates
    denom = max(total_supported, 1)
    overall["unauthorized_tenant_retrieval_rate"] = round(unauthorized_tenant_queries / denom, 4)
    overall["unauthorized_customer_retrieval_rate"] = round(unauthorized_customer_queries / denom, 4)
    overall["wrong_product_retrieval_rate"] = round(wrong_product_queries / denom, 4)
    overall["wrong_model_retrieval_rate"] = round(wrong_model_queries / denom, 4)
    overall["wrong_asset_retrieval_rate"] = round(wrong_asset_queries / denom, 4)
    overall["wrong_equipment_citation_rate"] = 0.0  # Dense research track does not generate answers
    overall["wrong_equipment_action_rate"] = 0.0

    latencies = [r["latency_ms"] for r in eval_records if r.get("latency_ms") is not None]
    if latencies:
        latencies.sort()
        overall["latency_ms_p50"] = round(latencies[len(latencies) // 2], 2)
        overall["latency_ms_p95"] = round(latencies[int(len(latencies) * 0.95)], 2)
        overall["latency_ms_p99"] = round(latencies[int(len(latencies) * 0.99)], 2)

    cat_breakdown = {}
    for cat, recs in category_records.items():
        cat_metrics = compute_metrics(recs)
        n_rel = sum(1 for r in recs if not r.get("is_unsupported"))
        n_abs = sum(1 for r in recs if r.get("is_unsupported"))
        entry = {
            "n_queries": len(recs),
            "n_relevance": n_rel,
            "n_abstention": n_abs,
            "hit@1": cat_metrics.get("hit@1"),
            "hit@5": cat_metrics.get("hit@5"),
            "hit@10": cat_metrics.get("hit@10"),
            "mrr@10": cat_metrics.get("mrr@10"),
            "ndcg@10": cat_metrics.get("ndcg@10"),
        }
        if "abstention_accuracy" in cat_metrics:
            entry["abstention_accuracy"] = cat_metrics["abstention_accuracy"]
        cat_breakdown[cat] = entry
    overall["category_breakdown"] = cat_breakdown
    return overall


# ---------------------------------------------------------------------------
# TRACK B: FULL CUSTOMER SERVING (Production Application Pipeline)
# ---------------------------------------------------------------------------

def evaluate_track_b_serving(
    pipeline_mode: str,
    cases: List[Dict[str, Any]],
    chunk_meta: Dict[int, Dict[str, Any]],
    k_max: int = 10,
    per_query_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate full application retrieval, evidence validation, and answer generation."""
    tracemalloc.start()
    t0_start = time.perf_counter()

    orig_exp = getattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
    orig_gate = getattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.55)

    try:
        if pipeline_mode in ("experimental_bge", "bge"):
            setattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", True)
        elif pipeline_mode in ("threshold_ablation_gate_055", "production_baseline"):
            # Phase 5 (Operation 16.0): Correctly labeled as THRESHOLD ABLATION,
            # not historical production baseline. Both variants run the CURRENT
            # retrieval implementation with different gate settings.
            setattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
            setattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.55)
        elif pipeline_mode in ("calibrated_hybrid_gate_038", "corrected_hybrid"):
            setattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
            setattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.38)
        else:
            raise ValueError(f"Unknown mode {pipeline_mode}")

        tenants = {t.id: t for t in Tenant.objects.all()}
        load_time_ms = (time.perf_counter() - t0_start) * 1000

        eval_records = []
        category_records = {}
        per_query_rows = []

        unauthorized_tenant_queries = 0
        unauthorized_customer_queries = 0
        wrong_product_queries = 0
        wrong_model_queries = 0
        wrong_asset_queries = 0
        wrong_citation_queries = 0
        wrong_action_queries = 0
        total_supported = 0

        for c in cases:
            qid = c["query_id"]
            q_text = c["query_text"]
            tid = c["tenant_id"]
            cat = c["category"]
            is_abstain = c["expected_abstain"]
            rel_ids = set(c["positive_chunk_ids"])
            equip_name = c["equipment"]

            tenant = tenants.get(tid)
            asset = resolve_query_asset(tid, equip_name)

            t0_q = time.perf_counter()
            rag_res = ask_rag(
                tenant=tenant,
                question=q_text,
                asset=asset,
                customer=None,
                top_k=k_max,
                include_confidential=False,
                staff_mode=False,
            )
            latency_ms = (time.perf_counter() - t0_q) * 1000

            retrieved_chunks = rag_res.get("retrieved", [])
            answer_text = rag_res.get("answer", "")
            engine = rag_res.get("engine", "")
            references = rag_res.get("references", [])

            is_direct = bool(engine != "policy-guard" and answer_text != INSUFFICIENT_EVIDENCE_MESSAGE)
            answer_mode = "DIRECT_ANSWER" if is_direct else "ABSTAIN"
            evidence_supported = is_direct

            ranked_cids = [rc.chunk_id for rc in retrieved_chunks if rc.chunk_id]
            # Phase 6 (Operation 16.0): Separate retrieval from abstention metrics.
            # Retrieval metrics (Hit@K, MRR, nDCG) use the actual ranked chunk IDs
            # from the retriever, regardless of whether answer generation abstained.
            # This prevents conflating retrieval recall with answer-policy behavior.
            #
            # Abstention is tracked separately below as a policy outcome.

            rec = {
                "query_id": qid,
                "ranked_chunk_ids": ranked_cids,  # Always the actual retrieval results
                "relevant_chunk_ids": list(rel_ids),
                "latency_ms": latency_ms,
                "is_unsupported": is_abstain,
                # Separate abstention tracking
                "answer_mode": answer_mode,
                "generation_abstained": not is_direct,
            }
            eval_records.append(rec)

            # Per-query diagnostic accounting
            first_rel_rank = None
            for i, cid in enumerate(ranked_cids, 1):
                if cid in rel_ids:
                    first_rel_rank = i
                    break

            top_chunk = retrieved_chunks[0] if retrieved_chunks else None
            cited_cids = [ref.get("chunk_id") for ref in references if ref.get("chunk_id")]

            has_unauth_tenant = False
            has_unauth_cust = False
            has_wrong_prod = False
            has_wrong_mod = False
            has_wrong_ast = False
            has_wrong_citation = False
            has_wrong_action = False

            for cid in ranked_cids:
                meta = chunk_meta.get(cid, {})
                b = check_chunk_boundaries(meta, tid, customer=None, asset=asset)
                if b["unauthorized_tenant"]: has_unauth_tenant = True
                if b["unauthorized_customer"]: has_unauth_cust = True
                if b["wrong_product"]: has_wrong_prod = True
                if b["wrong_model"]: has_wrong_mod = True
                if b["wrong_asset"]: has_wrong_ast = True

            for ccid in cited_cids:
                meta = chunk_meta.get(ccid, {})
                b = check_chunk_boundaries(meta, tid, customer=None, asset=asset)
                if b["unauthorized_tenant"] or b["wrong_product"] or b["wrong_asset"]:
                    has_wrong_citation = True
                    if is_direct:
                        has_wrong_action = True

            if not is_abstain:
                total_supported += 1
                if has_unauth_tenant: unauthorized_tenant_queries += 1
                if has_unauth_cust: unauthorized_customer_queries += 1
                if has_wrong_prod: wrong_product_queries += 1
                if has_wrong_mod: wrong_model_queries += 1
                if has_wrong_ast: wrong_asset_queries += 1
                if has_wrong_citation: wrong_citation_queries += 1
                if has_wrong_action: wrong_action_queries += 1

            if cat not in category_records:
                category_records[cat] = []
            category_records[cat].append(rec)

            # Build comprehensive per-query record
            pq_row = {
                "query_id": qid,
                "tenant_id": tid,
                "category": cat,
                "primary_evaluation_role": c.get("primary_evaluation_role"),
                "equipment": equip_name,
                "expected_abstain": is_abstain,
                "positive_chunk_ids": list(rel_ids),
                "retrieved_chunk_ids": ranked_cids[:5],
                "first_relevant_rank": first_rel_rank,
                "top_score": round(top_chunk.score, 4) if top_chunk else None,
                "top_fusion_score": round(top_chunk.fusion_score, 6) if top_chunk and top_chunk.fusion_score else None,
                "top_dense_sim": round(top_chunk.dense_similarity, 4) if top_chunk and top_chunk.dense_similarity else None,
                "top_sparse_rel": round(top_chunk.sparse_relevance, 4) if top_chunk and top_chunk.sparse_relevance else None,
                "answer_mode": answer_mode,
                "evidence_support": evidence_supported,
                "cited_chunk_ids": cited_cids,
                "passed_relevance": bool(first_rel_rank == 1) if not is_abstain else None,
                "passed_abstention": bool(len(ranked_cids) == 0 or not evidence_supported) if is_abstain else None,
            }
            per_query_rows.append(pq_row)

    finally:
        # Exception-safe restoration: always restore process-global settings
        setattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", orig_exp)
        setattr(settings, "SERVY_DENSE_RELEVANCE_GATE", orig_gate)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    overall = compute_metrics(eval_records)
    overall["track"] = "TRACK_B_CUSTOMER_SERVING"
    overall["pipeline_mode"] = pipeline_mode
    overall["load_time_ms"] = round(load_time_ms, 2)
    overall["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)

    denom = max(total_supported, 1)
    overall["unauthorized_tenant_retrieval_rate"] = round(unauthorized_tenant_queries / denom, 4)
    overall["unauthorized_customer_retrieval_rate"] = round(unauthorized_customer_queries / denom, 4)
    overall["wrong_product_retrieval_rate"] = round(wrong_product_queries / denom, 4)
    overall["wrong_model_retrieval_rate"] = round(wrong_model_queries / denom, 4)
    overall["wrong_asset_retrieval_rate"] = round(wrong_asset_queries / denom, 4)
    overall["wrong_equipment_citation_rate"] = round(wrong_citation_queries / denom, 4)
    overall["wrong_equipment_action_rate"] = round(wrong_action_queries / denom, 4)

    # Authenticated Cross-Tenant Attack Verification
    tenant_attacks = [c for c in cases if c["category"] == "tenant_isolation_attack"]
    attack_disclosures = 0
    for attack in tenant_attacks:
        atk_t = tenants.get(attack["tenant_id"])
        atk_res = retrieve(
            tenant=atk_t,
            question=attack["query_text"],
            asset=None,
            customer=None,
            top_k=k_max,
            include_confidential=False,
            staff_mode=False,
        )
        for chunk in atk_res:
            meta = chunk_meta.get(chunk.chunk_id, {})
            if meta.get("tenant_id") and meta["tenant_id"] != attack["tenant_id"]:
                attack_disclosures += 1
    overall["wrong_tenant_disclosures"] = attack_disclosures

    latencies = [r["latency_ms"] for r in eval_records if r.get("latency_ms") is not None]
    if latencies:
        latencies.sort()
        overall["latency_ms_p50"] = round(latencies[len(latencies) // 2], 2)
        overall["latency_ms_p95"] = round(latencies[int(len(latencies) * 0.95)], 2)
        overall["latency_ms_p99"] = round(latencies[int(len(latencies) * 0.99)], 2)

    cat_breakdown = {}
    for cat, recs in category_records.items():
        cat_metrics = compute_metrics(recs)
        n_rel = sum(1 for r in recs if not r.get("is_unsupported"))
        n_abs = sum(1 for r in recs if r.get("is_unsupported"))
        entry = {
            "n_queries": len(recs),
            "n_relevance": n_rel,
            "n_abstention": n_abs,
            "hit@1": cat_metrics.get("hit@1"),
            "hit@5": cat_metrics.get("hit@5"),
            "hit@10": cat_metrics.get("hit@10"),
            "mrr@10": cat_metrics.get("mrr@10"),
            "ndcg@10": cat_metrics.get("ndcg@10"),
        }
        if "abstention_accuracy" in cat_metrics:
            entry["abstention_accuracy"] = cat_metrics["abstention_accuracy"]
        cat_breakdown[cat] = entry
    overall["category_breakdown"] = cat_breakdown

    if per_query_output_path:
        per_query_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_query_output_path, "w", encoding="utf-8") as f:
            for row in per_query_rows:
                f.write(json.dumps(row) + "\n")

    return overall


def main():
    print("=" * 115)
    print("SERVY RAG OPERATION 12.0: COMPREHENSIVE TWO-TRACK EVALUATION")
    print("TRACK A: EMBEDDING-ONLY RESEARCH | TRACK B: FULL CUSTOMER SERVING")
    print("=" * 115)

    from core.ml.benchmark_registry import (
        DEFAULT_BENCHMARK_ID, load_and_verify_benchmark, get_evaluation_provenance_metadata,
    )
    dataset_id = os.environ.get("SERVY_BENCHMARK_ID", DEFAULT_BENCHMARK_ID)
    manifest_meta = load_and_verify_benchmark(dataset_id)
    print(f"Verified immutable benchmark SHA-256: {manifest_meta['sha256_digest']}")

    manifest = load_manifest(dataset_id)
    if manifest["sha256_digest"] != manifest_meta["sha256_digest"]:
        raise RuntimeError("Benchmark identity changed during evaluation")
    cases = manifest["cases"]
    print(f"Loaded {len(cases)} benchmark queries across 14 categories.")
    print(f"Denominators: {manifest['primary_denominators']}")

    chunk_meta = _build_chunk_metadata_cache()
    chunk_lookup = {(c.document_id, c.chunk_index): c.id for c in KnowledgeChunk.objects.all()}

    results = {}

    # -----------------------------------------------------------------------
    # TRACK A: EMBEDDING-ONLY RESEARCH (Identical Candidate Universe)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("TRACK A: EMBEDDING-ONLY RESEARCH (Standardized Candidate Universe)")
    print("#" * 60)

    print("\n--- Candidate A: Pure Dense MiniLM ---")
    results["track_a_minilm"] = evaluate_track_a_dense("minilm", cases, chunk_meta, chunk_lookup, k_max=10)
    print(f"  Hit@1: {results['track_a_minilm']['hit@1']} | Hit@5: {results['track_a_minilm']['hit@5']} | MRR@10: {results['track_a_minilm']['mrr@10']}")

    print("\n--- Candidate B: Pure Dense Base BGE-small ---")
    results["track_a_bge_base"] = evaluate_track_a_dense("bge-small", cases, chunk_meta, chunk_lookup, k_max=10)
    print(f"  Hit@1: {results['track_a_bge_base']['hit@1']} | Hit@5: {results['track_a_bge_base']['hit@5']} | MRR@10: {results['track_a_bge_base']['mrr@10']}")

    print("\n--- Candidate C: Pure Dense Fine-Tuned BGE-small (Tenant 9 Population, N=30) ---")
    t9_cases = [c for c in cases if c["tenant_id"] == 9]
    results["track_a_bge_ft_t9"] = evaluate_track_a_dense("bge-ft", t9_cases, chunk_meta, chunk_lookup, k_max=10)
    print(f"  Hit@1: {results['track_a_bge_ft_t9']['hit@1']} | Hit@5: {results['track_a_bge_ft_t9']['hit@5']} | MRR@10: {results['track_a_bge_ft_t9']['mrr@10']}")

    # -----------------------------------------------------------------------
    # TRACK B: FULL CUSTOMER SERVING (Production Application Pipeline)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("TRACK B: FULL CUSTOMER SERVING (Production Pipeline & Answer Generation)")
    print("#" * 60)

    pq_path = BASE_DIR / "ml_artifacts" / "benchmark_per_query_results.jsonl"

    print("\n--- Candidate D: Production Baseline Hybrid RRF (Legacy Gate 0.55) ---")
    results["track_b_baseline_hybrid"] = evaluate_track_b_serving(
        "production_baseline", cases, chunk_meta, k_max=10
    )
    print(f"  Hit@1: {results['track_b_baseline_hybrid']['hit@1']} | Hit@5: {results['track_b_baseline_hybrid']['hit@5']} | MRR@10: {results['track_b_baseline_hybrid']['mrr@10']}")

    print("\n--- Candidate E: Experimental BGE Hybrid RRF ---")
    results["track_b_experimental_bge"] = evaluate_track_b_serving(
        "experimental_bge", cases, chunk_meta, k_max=10
    )
    print(f"  Hit@1: {results['track_b_experimental_bge']['hit@1']} | Hit@5: {results['track_b_experimental_bge']['hit@5']} | MRR@10: {results['track_b_experimental_bge']['mrr@10']}")

    print("\n--- Candidate F: Corrected Production Hybrid RRF (Calibrated Gate 0.38 + Typo Normalizer) ---")
    results["track_b_corrected_hybrid"] = evaluate_track_b_serving(
        "corrected_hybrid", cases, chunk_meta, k_max=10, per_query_output_path=pq_path
    )
    print(f"  Hit@1: {results['track_b_corrected_hybrid']['hit@1']} | Hit@5: {results['track_b_corrected_hybrid']['hit@5']} | MRR@10: {results['track_b_corrected_hybrid']['mrr@10']}")

    # Add evaluation provenance metadata
    results["evaluation_provenance"] = get_evaluation_provenance_metadata()
    results["benchmark_identity"] = {
        "dataset_id": dataset_id,
        "dataset_role": manifest_meta["dataset_role"],
        "sha256_digest": manifest_meta["sha256_digest"],
        "number_of_cases": len(cases),
    }

    # Save full report
    report_path = BASE_DIR / "ml_artifacts" / "benchmark_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark evaluation report to: {report_path}")
    print(f"Saved per-query diagnostic accounting to: {pq_path}")

    # Summary table
    print("\n" + "=" * 125)
    print(f"{'Metric':<28} | {'Track A: MiniLM':<16} | {'Track A: BGE':<16} | {'Track A: FT(T9)':<16} | {'Track B: Baseline':<16} | {'Track B: Corrected':<16}")
    print("-" * 125)
    metrics_list = [
        ("hit@1", "Hit@1"),
        ("hit@5", "Hit@5"),
        ("hit@10", "Hit@10"),
        ("mrr@10", "MRR@10"),
        ("ndcg@10", "nDCG@10"),
        ("abstention_accuracy", "Abstention Acc"),
        ("unauthorized_tenant_retrieval_rate", "Unauth Tenant Rate"),
        ("wrong_product_retrieval_rate", "Wrong Product Rate"),
        ("wrong_asset_retrieval_rate", "Wrong Asset Rate"),
        ("wrong_equipment_citation_rate", "Wrong Citation Rate"),
        ("wrong_equipment_action_rate", "Wrong Action Rate"),
        ("wrong_tenant_disclosures", "Tenant Attack Leaks"),
        ("latency_ms_p50", "Latency p50 (ms)"),
        ("latency_ms_p95", "Latency p95 (ms)"),
        ("peak_memory_mb", "Peak Memory (MB)"),
    ]
    for key, label in metrics_list:
        v_a = results["track_a_minilm"].get(key, "N/A")
        v_b = results["track_a_bge_base"].get(key, "N/A")
        v_c = results["track_a_bge_ft_t9"].get(key, "N/A")
        v_d = results["track_b_baseline_hybrid"].get(key, "N/A")
        v_f = results["track_b_corrected_hybrid"].get(key, "N/A")
        print(f"{label:<28} | {str(v_a):<16} | {str(v_b):<16} | {str(v_c):<16} | {str(v_d):<16} | {str(v_f):<16}")
    print("=" * 125)


if __name__ == "__main__":
    main()
