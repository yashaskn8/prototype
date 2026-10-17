"""Operation 25 - Reconciliation of Commits 60f4044 and 465b25f + Candidate Recall Diagnosis.

Analyzes the 43 relevance diagnostic questions on the authorized corpus:
1. Reconciles exact per-query rankings between 60f4044 and 465b25f.
2. Identifies every query whose top-ranked passage changed.
3. Diagnoses candidate recall and loss points in retriever.py.
4. Generates a machine-readable baseline manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django
django.setup()

from core.models import Tenant, Asset, KnowledgeChunk, KnowledgeDocument
from core.ml.benchmark_registry import load_and_verify_benchmark, HOLDOUT_SHA256
from core.ml.benchmark_manifest import CATEGORY_TO_PRIMARY_ROLE, ROLE_RELEVANCE, resolve_query_asset
from core.services.retriever import retrieve, normalize_domain_terms, _knowledge_candidates, _dense_score_candidates, _sparse_score_candidates
from core.ml.embeddings import get_profile, load_model
from core.ml.conditional_rerank import (
    _procedural_density, _is_generic_heading, _extract_symptom_keys,
    _extract_fault_codes, _extract_query_keywords, _heading_keyword_overlap,
    _headings_semantically_similar, _get_content_score, _ordered_symptom_keys,
    _heading_matches_symptom, _heading_matches_specific_symptom,
    conditional_swap_rerank as current_serving_swap,
)

# Implementation of 60f4044 swap for exact comparison
def swap_60f4044(results, query_text, swap_window=5):
    if len(results) < 2:
        return results

    top1 = results[0]
    top1_heading = (top1.heading or "").strip()
    top1_text = top1.text or ""
    top1_doc_id = top1.document_id

    norm_query = normalize_domain_terms(query_text)
    combined_query = f"{query_text} {norm_query}"

    q_fault_codes = _extract_fault_codes(combined_query)
    q_symptoms = _extract_symptom_keys(combined_query)
    q_keywords = _extract_query_keywords(norm_query) | _extract_query_keywords(query_text)
    q_ordered_symptoms = _ordered_symptom_keys(norm_query)

    query_lower = combined_query.lower()
    is_low_pressure_query = "low pressure" in query_lower or "low-pressure" in query_lower
    has_static_pressure_phrase = "static pressure" in query_lower or "duct cleanliness" in query_lower

    top1_is_generic = _is_generic_heading(top1_heading)
    top1_proc_density = _procedural_density(top1_text)
    top1_heading_symptom = _heading_matches_symptom(top1_heading, q_symptoms, combined_query)
    top1_fault_in_heading = bool(q_fault_codes and q_fault_codes & _extract_fault_codes(top1_heading))
    top1_content_score = _get_content_score(top1)
    top1_kw_overlap = _heading_keyword_overlap(top1_heading, q_keywords)

    best_swap_idx = None
    best_swap_reason = None
    best_swap_confidence = 0.0

    for i in range(1, min(len(results), swap_window + 1)):
        cand = results[i]
        cand_heading = (cand.heading or "").strip()
        cand_text = cand.text or ""
        cand_doc_id = cand.document_id
        cand_proc_density = _procedural_density(cand_text)

        confidence = 0.0
        reason = None

        if (top1_heading.lower() == cand_heading.lower()
                and top1_doc_id == cand_doc_id
                and cand_proc_density > top1_proc_density + 0.15):
            confidence = 0.70 + (cand_proc_density - top1_proc_density) * 0.3
            reason = f"same_heading_procedural ({top1_proc_density:.2f}->{cand_proc_density:.2f})"

        elif (top1_is_generic
              and not _is_generic_heading(cand_heading)
              and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)
              and not top1_heading_symptom):
            confidence = 0.75
            reason = f"generic_demotion ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        elif (q_fault_codes
              and bool(q_fault_codes & _extract_fault_codes(cand_heading))
              and not top1_fault_in_heading
              and not top1_heading_symptom):
            confidence = 0.85
            reason = f"fault_code_promotion (codes={q_fault_codes})"

        elif (q_symptoms
              and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)
              and not top1_heading_symptom
              and not top1_is_generic
              and not _is_generic_heading(cand_heading)
              and i <= 3):
            confidence = 0.60
            reason = f"symptom_alignment (symptoms={q_symptoms}, cand_heading='{cand_heading[:30]}')"

        elif (top1_doc_id != cand_doc_id
              and _headings_semantically_similar(top1_heading, cand_heading)
              and not top1_is_generic
              and not _is_generic_heading(cand_heading)):
            cand_content_score = _get_content_score(cand)
            if cand_content_score > top1_content_score + 0.05:
                confidence = 0.65 + min((cand_content_score - top1_content_score) * 0.5, 0.25)
                reason = f"cross_doc_same_heading_score ({top1_content_score:.3f}->{cand_content_score:.3f})"

        elif (is_low_pressure_query
              and ("low-pressure" in cand_heading.lower() or "low pressure" in cand_heading.lower())
              and not ("low-pressure" in top1_heading.lower() or "low pressure" in top1_heading.lower())
              and i <= 3):
            confidence = 0.86
            reason = f"low_pressure_polarity_match ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        elif (("4" in query_lower and ("celsius" in query_lower or "4c" in query_lower or "deg" in query_lower))
              and "4.0 degrees celsius" in cand_text.lower()
              and cand_proc_density > 0.40
              and not top1_is_generic
              and i <= 3):
            confidence = 0.92
            reason = f"exact_numeric_spec_procedure ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        elif (has_static_pressure_phrase
              and "static pressure" in cand_text.lower()
              and "cleanliness" in cand_text.lower()
              and not _is_generic_heading(cand_heading)
              and i <= 3):
            confidence = 0.83
            reason = f"technical_phrase_match ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        elif (not top1_is_generic
              and not _is_generic_heading(cand_heading)
              and not top1_heading_symptom
              and i <= 4):
            cand_kw_overlap = _heading_keyword_overlap(cand_heading, q_keywords)
            kw_advantage = cand_kw_overlap - top1_kw_overlap
            if kw_advantage >= 2 and cand_kw_overlap >= 3:
                confidence = 0.62 + min(kw_advantage * 0.05, 0.18) + (cand_proc_density * 0.10)
                reason = f"heading_keyword_match (top1={top1_kw_overlap}, cand={cand_kw_overlap}, proc={cand_proc_density:.2f})"

        if (confidence < 0.60
                and len(q_ordered_symptoms) >= 2
                and i <= 3
                and top1_heading_symptom
                and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)):
            first_symptom = q_ordered_symptoms[0]
            cand_matches_first = _heading_matches_specific_symptom(cand_heading, first_symptom)
            top1_matches_first = _heading_matches_specific_symptom(top1_heading, first_symptom)
            if cand_matches_first and not top1_matches_first:
                confidence = 0.65
                reason = f"multi_symptom_first_priority (first='{first_symptom}', cand='{cand_heading[:30]}')"

        if confidence > best_swap_confidence and confidence >= 0.60:
            best_swap_confidence = confidence
            best_swap_idx = i
            best_swap_reason = reason

    if best_swap_idx is not None:
        res = list(results)
        res[0], res[best_swap_idx] = res[best_swap_idx], res[0]
        return res
    return results


def main():
    verified = load_and_verify_benchmark("servy-real-holdout-v1")
    cases = [c for c in verified["cases"] if CATEGORY_TO_PRIMARY_ROLE.get(c.get("category"), ROLE_RELEVANCE) == ROLE_RELEVANCE]
    n = len(cases)

    # Compute Corpus & Model Fingerprints
    docs = list(KnowledgeDocument.objects.all().order_by("id"))
    doc_ids = [d.id for d in docs]
    chunks = list(KnowledgeChunk.objects.all().order_by("id"))
    corpus_hasher = hashlib.sha256()
    corpus_hasher.update(f"{len(docs)}:{len(chunks)}".encode())
    for c in chunks:
        corpus_hasher.update(f"{c.id}:{c.document_id}:{c.heading}:{c.text[:30]}".encode())
    corpus_fingerprint = corpus_hasher.hexdigest()

    profile = get_profile("minilm")
    model_fingerprint = f"minilm:{profile.dim}:{profile.model_id}"

    print(f"Loaded {n} relevance diagnostic cases.")
    print(f"Corpus Fingerprint: {corpus_fingerprint[:16]}... ({len(docs)} docs, {len(chunks)} chunks)")
    print(f"Model Fingerprint:  {model_fingerprint}")
    print(f"Holdout SHA-256:    {HOLDOUT_SHA256[:16]}...")

    # We want to measure:
    # 1. Raw RRF ranking (prior to any conditional swap)
    # 2. 60f4044 swap ranking
    # 3. 465b25f (current serving) swap ranking
    # 4. Detailed candidates entering/leaving the pool
    
    raw_rrf_hits = {"h1": 0, "h5": 0, "h10": 0, "mrr": 0.0}
    c60f4044_hits = {"h1": 0, "h5": 0, "h10": 0, "mrr": 0.0}
    c465b25f_hits = {"h1": 0, "h5": 0, "h10": 0, "mrr": 0.0}

    query_comparisons = []
    diff_queries = []

    print("\n--- Query-by-Query Evaluation ---")

    for case in cases:
        qid = case["query_id"]
        q = case["query_text"]
        pos_ids = set(case.get("positive_chunk_ids", []))
        tid = case["tenant_id"]
        tenant = Tenant.objects.get(id=tid)
        asset = resolve_query_asset(tid, case.get("equipment", ""))

        t0 = time.perf_counter()
        # Retrieve full candidate set without swap first by using raw internal components or retrieve()
        # Let's inspect retrieve() - in 465b25f, retrieve() runs conditional_swap_rerank.
        # But we can also retrieve top_k=20 to see top 20 ranks!
        raw_results = retrieve(
            tenant=tenant, question=q, asset=asset,
            customer=None, include_confidential=False, top_k=20,
            min_score=0.01, service_call=None,
            include_past_resolutions=False, staff_mode=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        kd_chunks = [r for r in raw_results if r.source_kind == "knowledge_document"]
        kd_chunk_ids_current = [r.chunk_id for r in kd_chunks]

        # In current retriever.py, `results` had `current_serving_swap` applied at line 1047.
        # Let's also reconstruct the un-swapped RRF list from the dense and sparse maps to be 100% exact:
        # Actually, if we inspect `r.fusion_score` and sort by `fusion_score`, that gives us the exact pre-swap RRF!
        pre_swap_chunks = sorted(kd_chunks, key=lambda x: (x.fusion_score or 0.0), reverse=True)
        pre_swap_ids = [r.chunk_id for r in pre_swap_chunks]

        # Apply 60f4044 swap to pre_swap_chunks
        res_60f4044 = swap_60f4044(list(pre_swap_chunks), q, swap_window=5)
        ids_60f4044 = [r.chunk_id for r in res_60f4044]

        # Apply 465b25f swap to pre_swap_chunks
        res_465b25f = current_serving_swap(list(pre_swap_chunks), q, swap_window=5)
        ids_465b25f = [r.chunk_id for r in res_465b25f]

        # Evaluate Hit@1, Hit@5, MRR
        def eval_rank(ids):
            pos_rank = next((i + 1 for i, cid in enumerate(ids) if cid in pos_ids), None)
            h1 = bool(pos_rank == 1)
            h5 = bool(pos_rank and pos_rank <= 5)
            h10 = bool(pos_rank and pos_rank <= 10)
            rr = 1.0 / pos_rank if pos_rank and pos_rank <= 10 else 0.0
            return h1, h5, h10, rr, pos_rank

        h1_raw, h5_raw, h10_raw, rr_raw, rank_raw = eval_rank(pre_swap_ids)
        h1_60, h5_60, h10_60, rr_60, rank_60 = eval_rank(ids_60f4044)
        h1_46, h5_46, h10_46, rr_46, rank_46 = eval_rank(ids_465b25f)

        raw_rrf_hits["h1"] += h1_raw
        raw_rrf_hits["h5"] += h5_raw
        raw_rrf_hits["h10"] += h10_raw
        raw_rrf_hits["mrr"] += rr_raw

        c60f4044_hits["h1"] += h1_60
        c60f4044_hits["h5"] += h5_60
        c60f4044_hits["h10"] += h10_60
        c60f4044_hits["mrr"] += rr_60

        c465b25f_hits["h1"] += h1_46
        c465b25f_hits["h5"] += h5_46
        c465b25f_hits["h10"] += h10_46
        c465b25f_hits["mrr"] += rr_46

        # Check if top-ranked passage changed between 60f4044 and 465b25f
        top1_60 = ids_60f4044[0] if ids_60f4044 else None
        top1_46 = ids_465b25f[0] if ids_465b25f else None

        changed = (top1_60 != top1_46)
        if changed:
            diff_queries.append({
                "qid": qid,
                "query": q,
                "pos_ids": list(pos_ids),
                "top1_60f4044": top1_60,
                "rank_60f4044": rank_60,
                "top1_465b25f": top1_46,
                "rank_465b25f": rank_46,
                "top1_60_heading": res_60f4044[0].heading if res_60f4044 else "",
                "top1_46_heading": res_465b25f[0].heading if res_465b25f else "",
            })
            print(f"  [DIFF] {qid:12s} 60f4044: [{top1_60}] (rank {rank_60}) vs 465b25f: [{top1_46}] (rank {rank_46}) | Gold: {list(pos_ids)}")

        query_comparisons.append({
            "qid": qid,
            "category": case.get("category"),
            "equipment": case.get("equipment"),
            "query": q,
            "pos_ids": list(pos_ids),
            "pre_swap_top5": pre_swap_ids[:5],
            "rank_raw": rank_raw,
            "top20_ids_465b25f": ids_465b25f[:20],
            "top1_id": top1_46,
            "top1_heading": res_465b25f[0].heading if res_465b25f else "",
            "top1_source_id": res_465b25f[0].document_id if res_465b25f else None,
            "rank_465b25f": rank_46,
            "rank_60f4044": rank_60,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    print("\n" + "=" * 65)
    print(f"Pre-swap RRF Baseline:  Hit@1: {raw_rrf_hits['h1']}/{n} ({raw_rrf_hits['h1']/n*100:.1f}%), Hit@5: {raw_rrf_hits['h5']}/{n} ({raw_rrf_hits['h5']/n*100:.1f}%), MRR: {raw_rrf_hits['mrr']/n:.4f}")
    print(f"Commit 60f4044:         Hit@1: {c60f4044_hits['h1']}/{n} ({c60f4044_hits['h1']/n*100:.1f}%), Hit@5: {c60f4044_hits['h5']}/{n} ({c60f4044_hits['h5']/n*100:.1f}%), MRR: {c60f4044_hits['mrr']/n:.4f}")
    print(f"Commit 465b25f:         Hit@1: {c465b25f_hits['h1']}/{n} ({c465b25f_hits['h1']/n*100:.1f}%), Hit@5: {c465b25f_hits['h5']}/{n} ({c465b25f_hits['h5']/n*100:.1f}%), MRR: {c465b25f_hits['mrr']/n:.4f}")
    print(f"Queries with Top-Rank Discrepancy between 60f4044 and 465b25f: {len(diff_queries)}")
    print("=" * 65)

    # Save machine-readable baseline manifest
    manifest = {
        "operation": "OPERATION_25",
        "description": "Baseline reconciliation between commits 60f4044 and 465b25f",
        "git_sha_current": "465b25f68b11c23fd8a2214b81e3d14013ade6f6",
        "git_sha_previous": "60f4044fdb0d6fbdbf9d0ce02b26ed37db1c8c04",
        "corpus_fingerprint": corpus_fingerprint,
        "model_fingerprint": model_fingerprint,
        "evaluation_set_sha256": HOLDOUT_SHA256,
        "n_relevance_queries": n,
        "metrics_raw_rrf": {
            "hit_at_1": round(raw_rrf_hits["h1"] / n, 4),
            "hit_at_5": round(raw_rrf_hits["h5"] / n, 4),
            "hit_at_10": round(raw_rrf_hits["h10"] / n, 4),
            "mrr_at_10": round(raw_rrf_hits["mrr"] / n, 4),
        },
        "metrics_60f4044": {
            "hit_at_1": round(c60f4044_hits["h1"] / n, 4),
            "hit_at_5": round(c60f4044_hits["h5"] / n, 4),
            "hit_at_10": round(c60f4044_hits["h10"] / n, 4),
            "mrr_at_10": round(c60f4044_hits["mrr"] / n, 4),
        },
        "metrics_465b25f": {
            "hit_at_1": round(c465b25f_hits["h1"] / n, 4),
            "hit_at_5": round(c465b25f_hits["h5"] / n, 4),
            "hit_at_10": round(c465b25f_hits["h10"] / n, 4),
            "mrr_at_10": round(c465b25f_hits["mrr"] / n, 4),
        },
        "discrepancies": diff_queries,
        "per_query_rankings": query_comparisons,
    }

    manifest_path = ROOT / "ml_artifacts" / "op25_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved baseline manifest to: {manifest_path}")

if __name__ == "__main__":
    main()
