"""Read-only paired ranking diagnostics for research checkpoints.

Provides auditable, honest candidate-pool ceilings and paired query movements.
Neither development-set improvement nor this analysis authorizes promotion.
Per-query output can include proprietary IDs; keep it in approved private storage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL = "servy-frozen-pool-paired-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(results, frozen, model_name):
    rows = results.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{model_name}: missing per-query evaluation")
    indexed = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        if not qid or qid in indexed or qid not in frozen:
            raise ValueError(f"{model_name}: duplicate/unknown query ID: {qid}")
        expected = [str(x) for x in frozen[qid]]
        candidate_ids = [str(x) for x in row["candidate_chunk_ids"]]
        ranked_ids = [str(x) for x in row["ranked_chunk_ids"]]
        gold_ids = {str(x) for x in row["gold_chunk_ids"]}
        if not expected or len(expected) != len(set(expected)):
            raise ValueError(f"{model_name}: invalid frozen candidates: {qid}")
        if candidate_ids != expected or len(ranked_ids) != len(expected) or set(ranked_ids) != set(expected):
            raise ValueError(f"{model_name}: candidate scope or membership changed: {qid}")
        actual_original_rank = next((i for i, cid in enumerate(expected, 1) if cid in gold_ids), None)
        actual_rank = next((i for i, cid in enumerate(ranked_ids, 1) if cid in gold_ids), None)
        if row["original_gold_rank"] != actual_original_rank or row["gold_rank"] != actual_rank:
            raise ValueError(f"{model_name}: reported gold rank contradicts candidate ordering: {qid}")
        indexed[qid] = row
    if set(indexed) != set(frozen):
        raise ValueError(f"{model_name}: missing query rows")
    return indexed


def compare_frozen_pool(before, after, frozen, *, pool_size=5):
    """Compare two full evaluate_one results on identical frozen candidates.

    The top-1 ceiling is bounded by original recall within the head pool.
    A gold chunk outside the pool cannot move to rank 1 when the tail is fixed.
    """
    if not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    old = _validate(before, frozen, "before")
    new = _validate(after, frozen, "after")
    movement = {
        "up": 0, "down": 0, "unchanged_rank": 0,
        "found": 0, "lost": 0, "both_missing": 0,
    }
    rows = []
    head_reachable = 0
    candidate_reachable = 0
    changed_top_one = 0
    regression_from_top_one = 0
    improvement_to_top_one = 0
    improved_gold = 0
    worsened_gold = 0

    for qid in frozen:
        first, second = old[qid], new[qid]
        if first["gold_chunk_ids"] != second["gold_chunk_ids"]:
            raise ValueError(f"Gold labels changed during evaluation: {qid}")
        original_rank = first["original_gold_rank"]
        if original_rank != second["original_gold_rank"]:
            raise ValueError(f"Frozen candidate rank changed: {qid}")
        a, b = first["gold_rank"], second["gold_rank"]
        # A gold rank cannot disappear when both models only reorder the same candidates.
        if (a is None) != (b is None):
            raise ValueError(f"Inconsistent candidate membership: {qid}")
        if a is None:
            label = "both_missing"
        elif b < a:
            label = "up"
            improved_gold += 1
        elif b > a:
            label = "down"
            worsened_gold += 1
        else:
            label = "unchanged_rank"
        movement[label] += 1

        if original_rank is not None:
            candidate_reachable += 1
            if original_rank <= pool_size:
                head_reachable += 1
        if first["ranked_chunk_ids"][0] != second["ranked_chunk_ids"][0]:
            changed_top_one += 1
        if a == 1 and b != 1:
            regression_from_top_one += 1
        if a != 1 and b == 1:
            improvement_to_top_one += 1

        rows.append({
            "query_id": qid,
            "original_gold_rank": original_rank,
            "before_gold_rank": a,
            "after_gold_rank": b,
            "gold_rank_movement": label,
            "before_top_chunk_id": first["ranked_chunk_ids"][0],
            "after_top_chunk_id": second["ranked_chunk_ids"][0],
            "changed_top_one": first["ranked_chunk_ids"][0] != second["ranked_chunk_ids"][0],
            "head_pool_reachable": original_rank is not None and original_rank <= pool_size,
            "candidate_top10_reachable": original_rank is not None and original_rank <= 10,
        })
    n = len(frozen)
    pct = lambda v: round(100 * v / n, 2)
    summary = {
        "protocol": PROTOCOL,
        "queries": n,
        "pool_size": pool_size,
        "candidate_recall_at_10_count": sum(
            row["candidate_top10_reachable"] for row in rows
        ),
        "candidate_recall_at_10_pct": pct(sum(row["candidate_top10_reachable"] for row in rows)),
        "head_pool_oracle_hit_at_1_ceiling_count": head_reachable,
        "head_pool_oracle_hit_at_1_ceiling_pct": pct(head_reachable),
        "queries_with_gold_outside_head_but_inside_candidates": candidate_reachable - head_reachable,
        "queries_missing_gold_from_candidates": n - candidate_reachable,
        "before_top1_count": sum(row["before_gold_rank"] == 1 for row in rows),
        "after_top1_count": sum(row["after_gold_rank"] == 1 for row in rows),
        "before_missed_head_reachable": sum(
            row["head_pool_reachable"] and row["before_gold_rank"] != 1 for row in rows
        ),
        "after_missed_head_reachable": sum(
            row["head_pool_reachable"] and row["after_gold_rank"] != 1 for row in rows
        ),
        "query_gold_rank_movements": movement,
        "queries_with_different_top_chunk": changed_top_one,
        "queries_improved_to_top1": improvement_to_top_one,
        "queries_regressed_from_top1": regression_from_top_one,
        "queries_with_improved_gold_rank": improved_gold,
        "queries_with_worsened_gold_rank": worsened_gold,
        "evaluation_status": "REUSED_PROVISIONAL_DEVELOPMENT_ONLY",
        "independent_holdout_verified": False,
        "production_promotion_allowed": False,
        "model_decision": "KEEP_V5_UNTIL_INDEPENDENT_EVIDENCE",
    }
    return {"summary": summary, "per_query": rows}


def write_jsonl_exclusive(path, records):
    """Never silently overwrite a previous private experiment output."""
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
