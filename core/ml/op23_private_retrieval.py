"""Read-only MiniLM passage retrieval over an admitted private source universe.

This baseline never writes Chroma or changes customer serving. It does not
generate technical answers; its predictions measure retrieval only.
"""
from __future__ import annotations

import time

import numpy as np


class PrivateRetrievalError(ValueError):
    pass


def predict_minilm_passages(cases: list[dict], admission: dict, model) -> list[dict]:
    sources = {s["source_id"]: s for s in admission["sources"] if s.get("source_id")}
    predictions = []
    for case in cases:
        started = time.perf_counter()
        base = {"query_id": case["query_id"], "tenant_id": admission["tenant_id"],
                "external_asset_id": case.get("external_asset_id"), "answered": False}
        if case["answerability"] == "NEEDS_ASSET_CLARIFICATION":
            predictions.append({**base, "decision": "EQUIPMENT_CLARIFICATION_REQUIRED",
                                "candidate_passage_ids": [], "ranked_passage_ids": [],
                                "retrieved_tenant_ids": [], "retrieval_latency_ms": 0.0})
            continue
        universe = case["searched_source_universe"]
        if not universe.get("search_complete") or set(universe.get("indexed_source_ids") or []) != set(universe["source_ids"]):
            raise PrivateRetrievalError("Cannot predict from incomplete private source index")
        passages = []
        for sid in universe["source_ids"]:
            source = sources.get(sid)
            if (not source or source.get("source_verification") != "VERIFIED_ORIGINAL_COMPANY_SOURCE"
                    or source.get("tenant_id") != admission["tenant_id"]
                    or source.get("external_asset_id") != case["external_asset_id"]):
                raise PrivateRetrievalError("Source outside the approved tenant or asset universe")
            passages.extend((source, passage) for passage in source["passages"])
        if not passages:
            raise PrivateRetrievalError("Approved source universe contains no passages")
        texts = [f"{source['attachment_name']}\n{source['asset_code']}\n{passage['text']}"
                 for source, passage in passages]
        vectors = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=8), dtype=np.float32)
        query_vector = np.asarray(model.encode([case["question_text"]], normalize_embeddings=True), dtype=np.float32)[0]
        if vectors.ndim != 2 or vectors.shape[0] != len(passages) or vectors.shape[1] != query_vector.shape[0]:
            raise PrivateRetrievalError("Embedding dimensions do not match admitted passages")
        scores = vectors @ query_vector
        order = sorted(range(len(passages)), key=lambda i: (-float(scores[i]), passages[i][1]["passage_id"]))
        first_source, first_passage = passages[order[0]]
        predictions.append({**base, "decision": "RETRIEVAL_ONLY_NO_ANSWER_GENERATED",
                            "candidate_passage_ids": [passage["passage_id"] for _, passage in passages],
                            "ranked_passage_ids": [passages[i][1]["passage_id"] for i in order],
                            "selected_passage_id": first_passage["passage_id"],
                            "selected_source_id": first_source["source_id"],
                            "retrieved_tenant_ids": [admission["tenant_id"]] * len(passages),
                            "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    return predictions
