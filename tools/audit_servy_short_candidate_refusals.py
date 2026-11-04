"""Build a private review queue from the frozen 300-question candidate run.

This audit flags possible incorrect refusals but never promotes an automatic
candidate to approved ground truth or declares semantic answerability.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
BANK = BASE / "servy_short_questions_20261104_01" / "short_questions_300_candidate.jsonl"
OUT = BASE / "servy_short_answer_candidates_20261104_01"


def main():
    questions = {row["id"]: row for row in
                 (json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines())}
    results = [json.loads(line) for line in
               (OUT / "answer_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(questions) != 300 or len(results) != 300 or len({r["id"] for r in results}) != 300:
        raise RuntimeError("Complete, unique 300-question run required")
    if set(questions) != {r["id"] for r in results}:
        raise RuntimeError("Candidate IDs differ from the frozen bank")
    queue = []
    priorities = Counter()
    matrix = Counter()
    for result in results:
        row = questions[result["id"]]
        if result["question_sha256"] != row["question_sha256"] or (
            hashlib.sha256(row["question"].encode()).hexdigest() != row["question_sha256"]
        ):
            raise RuntimeError("Question hash mismatch")
        kind, outcome = row["question_kind"], result["outcome"]
        matrix[(kind, outcome)] += 1
        if outcome == "REFUSED" and row.get("source_chunk_id"):
            priority = "POSSIBLE_INCORRECT_REFUSAL_SOURCE_ANCHOR_PRESENT"
        elif outcome == "REFUSED":
            priority = "REFUSAL_NEEDS_EVIDENCE_REVIEW"
        elif outcome == "ANSWERED":
            priority = "ANSWER_NEEDS_CLAIM_AND_CITATION_REVIEW"
        elif outcome == "CLARIFICATION":
            priority = "CLARIFICATION_NEEDS_REVIEW"
        else:
            priority = "ERROR_NEEDS_REVIEW"
        priorities[priority] += 1
        queue.append({"id": row["id"], "asset_id": row["asset_id"],
                      "customer_id": row["customer_id"], "question_sha256": row["question_sha256"],
                      "split": result["split"], "question_kind": kind, "actual_outcome": outcome,
                      "source_document_id": row.get("source_document_id"),
                      "source_chunk_id": row.get("source_chunk_id"),
                      "source_chunk_sha256": row.get("source_chunk_sha256"),
                      "candidate_citations": result.get("candidate_citations", []),
                      "mechanical_check": result.get("mechanical_check"),
                      "review_priority": priority,
                      "answerability_judgment": "NEEDS_EXPERT_REVIEW",
                      "technical_correctness": "NEEDS_EXPERT_REVIEW",
                      "approved_for_training": False})
    target = OUT / "review_queue.jsonl"
    with target.open("x", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"total": 300, "outcomes_by_question_kind": {
        f"{kind}:{outcome}": count for (kind, outcome), count in sorted(matrix.items())},
        "review_priorities": dict(priorities),
        "confirmed_incorrect_refusals": None,
        "confirmed_correct_refusals": None,
        "approved_training_labels": 0,
        "semantic_review_status": "PENDING_EXPERT_REVIEW"}
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
