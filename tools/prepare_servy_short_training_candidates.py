"""Prepare private, unapproved answer candidates for the frozen 300 short questions.

All company data is read through the SELECT-only account. Chroma opens a private
copy, and no candidate produced here is an approved training label.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
from tools.servy_500_benchmark import (
    activate_index_copy, assets_and_sources, file_manifest, operational_counts,
    prepare_index_copy, private_folder, read_only_database, serving, sha,
)


BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
BANK = BASE / "servy_short_questions_20261104_01" / "short_questions_300_candidate.jsonl"
OUT = BASE / "servy_short_answer_candidates_20261104_01"
ASSETS = {59: 14, 60: 13, 61: 14}


def _rows():
    rows = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 300 or len({r.get("id") for r in rows}) != 300:
        raise RuntimeError("Expected 300 distinct questions")
    if sha(BANK) != "fc4969a24ce274c6a9608c2c4f867989341663a0f952653e748d81fa271f41e8":
        raise RuntimeError("Frozen question bank hash changed")
    for row in rows:
        if (row.get("tenant") != "skydream"
            or ASSETS.get(row.get("asset_id")) != row.get("customer_id")
            or hashlib.sha256(row["question"].encode()).hexdigest() != row.get("question_sha256")
            or row.get("approved_for_training") is not False):
            raise RuntimeError("Question scope, text, or approval state changed")
    return rows


def _mechanical_check(result, allowed):
    presentation = result.get("presentation") or {}
    citations = presentation.get("citations") or []
    cited = {c.get("id"): c for c in citations}
    sections = presentation.get("sections") or []
    claims = [claim for section in sections for claim in section.get("claims", [])
              if claim.get("kind") not in ("uncertainty", "next_step")]
    citations_in_scope = all(
        (c.get("document_id"), c.get("chunk_id")) in allowed for c in citations
    )
    quotes_exact = all(
        bool(claim.get("source_text")) and any(
            claim["source_text"] in allowed.get(
                (cited[e].get("document_id"), cited[e].get("chunk_id")), ""
            ) for e in claim.get("evidence_ids", []) if e in cited
        ) for claim in claims
    )
    four_points = all(any(s.get("number") == n for s in sections) for n in range(1, 5))
    return {"citations_in_authorized_scope": citations_in_scope,
            "claim_quotes_exact": quotes_exact if claims else None,
            "four_point_structure": four_points,
            "claim_count": len(claims), "citation_count": len(citations),
            "semantic_correctness": "PENDING_EXPERT_REVIEW"}


def main():
    # Explicitly pin the offline extractive route for this process. The shared
    # application may use "auto", which can contact an Ollama endpoint.
    os.environ["SERVY_LLM_PROVIDER"] = "extractive"
    os.environ["SERVY_ALLOW_REMOTE_LLM"] = "0"
    configure_process()
    verify_readonly_identity()
    config = serving()
    if config["llm_provider"] != "extractive":
        raise RuntimeError("Candidate run requires the strict offline extractive provider")
    rows = _rows()
    private_folder(OUT)
    results_path = OUT / "answer_candidates.jsonl"
    if results_path.exists() or (OUT / "chroma_index_copy").exists():
        raise RuntimeError("This private candidate run already exists; no duplicate calls")
    from django.conf import settings
    from tools.servy_real_demo import _answer

    model_path = Path(settings.SERVY_RERANKER_MODEL_PATH).resolve()
    model_before = file_manifest(model_path)
    with read_only_database():
        _, assets, coverage, source_rows = assets_and_sources(rows)
        if set(assets) != set(ASSETS) or any(
            coverage[str(a)]["customer_id"] != c
            or coverage[str(a)]["document_count"] != 3
            or not coverage[str(a)]["original_documentation_eligible"]
            for a, c in ASSETS.items()
        ):
            raise RuntimeError("Nine-document authorized scope is not intact")
        counts_before = operational_counts()
    allowed = {int(a): {(c["document_id"], c["chunk_id"]): c["text"] for c in chunks}
               for a, chunks in source_rows.items()}
    source_path, _ = prepare_index_copy(OUT)
    original_manifest = file_manifest(source_path)
    activate_index_copy(OUT)
    # Deterministic 20% holdout by ID; all rows remain unapproved candidates.
    split = {row["id"]: ("holdout" if int(hashlib.sha256(
        ("servy-short-holdout-v1:" + row["id"]).encode()).hexdigest(), 16) % 5 == 0
        else "development") for row in rows}
    with (OUT / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    outcomes = Counter()
    structural = Counter()
    with read_only_database(), results_path.open("x", encoding="utf-8") as output:
        for n, row in enumerate(rows, 1):
            try:
                actual = _answer(row["asset_id"], row["question"])
                response_type = actual.get("response_type")
                outcome = {"DIAGNOSTIC_GUIDANCE": "ANSWERED",
                           "CLARIFICATION": "CLARIFICATION",
                           "INSUFFICIENT_EVIDENCE": "REFUSED"}.get(response_type, "ERROR")
                check = _mechanical_check(actual, allowed[row["asset_id"]])
                candidate = {"id": row["id"], "question_sha256": row["question_sha256"],
                             "asset_id": row["asset_id"], "customer_id": row["customer_id"],
                             "split": split[row["id"]], "outcome": outcome,
                             "candidate_behavior": response_type,
                             "candidate_answer": actual.get("presentation"),
                             "candidate_citations": actual.get("citations"),
                             "candidate_source_document_id": row.get("source_document_id"),
                             "candidate_source_chunk_id": row.get("source_chunk_id"),
                             "source_chunk_sha256": row.get("source_chunk_sha256"),
                             "mechanical_check": check,
                             "expert_review_status": "PENDING_REVIEW",
                             "approved_for_training": False}
            except Exception as exc:
                outcome = "ERROR"
                candidate = {"id": row["id"], "question_sha256": row["question_sha256"],
                             "asset_id": row["asset_id"], "customer_id": row["customer_id"],
                             "split": split[row["id"]], "outcome": outcome,
                             "error_type": type(exc).__name__,
                             "expert_review_status": "PENDING_REVIEW",
                             "approved_for_training": False}
            output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            output.flush()
            outcomes[outcome] += 1
            if candidate.get("mechanical_check"):
                check = candidate["mechanical_check"]
                if not check["citations_in_authorized_scope"] or check["claim_quotes_exact"] is False:
                    structural["citation_or_quote_mismatch"] += 1
            if n % 25 == 0:
                print(json.dumps({"completed": n, "outcomes": dict(outcomes)}), flush=True)
        if operational_counts() != counts_before:
            raise RuntimeError("Operational record counts changed during evaluation")
    if file_manifest(source_path) != original_manifest or file_manifest(model_path) != model_before:
        raise RuntimeError("Original index or V5 checkpoint changed")
    report = {"questions": len(rows), "completed": sum(outcomes.values()),
              "outcomes": dict(outcomes), "structural_issues": dict(structural),
              "approved_training_labels": 0, "expert_review_status": "PENDING_REVIEW",
              "semantic_accuracy": None, "paid_api_calls": 0,
              "database_account": "servy_eval_ro@127.0.0.1",
              "question_bank_sha256": sha(BANK),
              "original_index_unchanged": True, "v5_checkpoint_unchanged": True}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report, "private_directory": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
