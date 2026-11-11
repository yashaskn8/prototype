#!/usr/bin/env python3
"""Offline read-only comparison for Servy's section-aware reranking experiment.

Questions JSONL fields: query_id, asset_id, question, expected_chunk or
gold_chunk_ids, and reranked_top10_chunks or baseline_ranked_chunk_ids.
Corpus JSONL fields: chunk_id, asset_id, document_id, title, heading, text, doc_type.
The candidate list may contain up to 20 ordered IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from core.ml.section_aware_rerank import experimental_evidence_section_rerank
from core.services.retriever import RetrievedChunk


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def normalize_id(value) -> str:
    return str(value).strip()


def as_id_list(row: dict, keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        if not isinstance(value, list):
            value = [value]
        return [normalize_id(item) for item in value if item is not None]
    return []


def build_chunk(row: dict, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        source_kind="knowledge_document",
        source_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        chunk_id=int(row["chunk_id"]),
        title=str(row.get("title") or ""),
        heading=str(row.get("heading") or ""),
        text=str(row.get("text") or ""),
        score=1.0 / rank,
        doc_type=str(row.get("doc_type") or ""),
        source_url="",
        confidential=False,
        fusion_score=1.0 / rank,
        fusion_rank=rank,
    )


def metrics(rows: list[dict], key: str) -> dict:
    count = len(rows)
    if not count:
        return {"n": 0}
    out = {"n": count}
    for k in (1, 5, 10):
        hits = sum(bool(set(row["_gold"]) & set(row[key][:k])) for row in rows)
        out[f"hit@{k}"] = round(hits / count, 6)
    reciprocal = []
    for row in rows:
        gold = set(row["_gold"])
        rank = next((i for i, chunk_id in enumerate(row[key], 1) if chunk_id in gold), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    out["mrr"] = round(sum(reciprocal) / count, 6)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    corpus_rows = read_jsonl(args.corpus)
    corpus = {}
    for row in corpus_rows:
        if not {"chunk_id", "asset_id", "document_id"} <= set(row):
            raise ValueError("Corpus rows require chunk_id, asset_id, and document_id")
        key = normalize_id(row["chunk_id"])
        if key in corpus:
            raise ValueError(f"Duplicate corpus chunk_id: {key}")
        corpus[key] = row

    evaluated, skipped, latencies = [], [], []
    for row in questions:
        qid = str(row.get("query_id") or row.get("candidate_id") or len(evaluated) + 1)
        asset_id = normalize_id(row.get("asset_id", ""))
        query = str(row.get("question") or row.get("query") or "").strip()
        gold = as_id_list(row, ("gold_chunk_ids", "expected_chunks", "expected_chunk"))
        candidate_ids = as_id_list(
            row, ("baseline_ranked_chunk_ids", "reranked_top10_chunks",
                 "bge_xlmr_top10_chunks", "baseline_top10_chunks")
        )
        if not query or not asset_id or not gold or not candidate_ids:
            skipped.append({"query_id": qid, "reason": "missing question, asset, gold label or candidate IDs"})
            continue

        candidates, missing = [], []
        for chunk_id in candidate_ids[:20]:
            passage = corpus.get(chunk_id)
            if passage is None:
                missing.append(chunk_id)
                continue
            # Hard fail-closed evaluation scope: foreign-asset candidates never enter ranking.
            if normalize_id(passage["asset_id"]) != asset_id:
                continue
            candidates.append(build_chunk(passage, len(candidates) + 1))
        if not candidates:
            skipped.append({"query_id": qid, "reason": "no same-asset corpus candidates"})
            continue

        baseline_order = [normalize_id(item.chunk_id) for item in candidates]
        started = time.perf_counter()
        experimental = experimental_evidence_section_rerank(
            list(candidates), query, top_k=min(10, len(candidates))
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        evaluated.append({
            "query_id": qid,
            "asset_id": asset_id,
            "question_type": str(row.get("question_type") or "UNKNOWN"),
            "_gold": gold,
            "baseline": baseline_order,
            "experimental": [normalize_id(item.chunk_id) for item in experimental],
            "missing_corpus_chunks": missing,
        })

    if not evaluated:
        raise SystemExit("No scorable queries. Check JSONL fields and same-asset IDs.")

    result = {
        "evaluation_kind": "INTERNAL_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION",
        "method": "incoming ranking + evidence-aware reranking + section-intent adjustment",
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "input_question_count": len(questions),
        "evaluated_question_count": len(evaluated),
        "skipped_question_count": len(skipped),
        "skipped": skipped,
        "candidate_recall": {
            f"recall@{k}": round(
                sum(bool(set(row["_gold"]) & set(row["baseline"][:k])) for row in evaluated) / len(evaluated), 6
            ) for k in (5, 10, 20)
        },
        "baseline": metrics(evaluated, "baseline"),
        "experimental": metrics(evaluated, "experimental"),
        "mean_experimental_rerank_ms": round(statistics.mean(latencies), 3),
        "p95_experimental_rerank_ms": round(
            sorted(latencies)[min(len(latencies)-1, math.ceil(0.95 * len(latencies))-1)], 3
        ),
        "by_asset": {},
        "by_question_type": {},
        "provenance_warning": (
            "Metrics inherit the supplied question bank, gold mappings and corpus. "
            "The tool does not verify original manuals or human relevance judgments."
        ),
        "production_modified": False,
        "training_performed": False,
    }
    for field, target in (("asset_id", "by_asset"), ("question_type", "by_question_type")):
        grouped = defaultdict(list)
        for item in evaluated:
            grouped[item[field]].append(item)
        result[target] = {
            name: {"baseline": metrics(group, "baseline"), "experimental": metrics(group, "experimental")}
            for name, group in sorted(grouped.items())
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "evaluated": len(evaluated),
        "skipped": len(skipped),
        "candidate_recall": result["candidate_recall"],
        "baseline": result["baseline"],
        "experimental": result["experimental"],
        "provenance_warning": result["provenance_warning"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
