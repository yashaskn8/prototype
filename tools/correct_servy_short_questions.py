"""Version the 300-question bank after removing generated manual sequence labels.

The original bank is immutable. This is a question-quality correction, not an
approved answer key or permission to train a model.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
from tools.servy_500_benchmark import read_only_database, sha

BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
FOLDER = BASE / "servy_short_questions_20261104_01"
SOURCE = FOLDER / "short_questions_300_candidate.jsonl"
TARGET = FOLDER / "short_questions_300_v2_candidate.jsonl"
CHANGES = FOLDER / "short_questions_300_v2_change_log.json"
EXPECTED_SHA = "fc4969a24ce274c6a9608c2c4f867989341663a0f952653e748d81fa271f41e8"


def main():
    if sha(SOURCE) != EXPECTED_SHA or TARGET.exists() or CHANGES.exists():
        raise RuntimeError("Original bank changed or corrected version already exists")
    configure_process()
    verify_readonly_identity()
    from core.services.customer_answer import source_chunk
    from tools.servy_real_demo import _asset
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    changes = []
    with read_only_database():
        for row in rows:
            if row["question_kind"] != "source":
                continue
            tenant, customer, asset = _asset(row["asset_id"])
            chunk = source_chunk(tenant=tenant, customer=customer, asset=asset,
                                 document_id=row["source_document_id"],
                                 chunk_id=row["source_chunk_id"])
            if hashlib.sha256(chunk.text.encode()).hexdigest() != row["source_chunk_sha256"]:
                raise RuntimeError("Source anchor changed")
            title_part = chunk.document.title.split(" - ", 1)[-1][:38]
            clean_part = re.sub(r"^\d{1,3}\s+", "", title_part)
            old = row["question"]
            new = old.replace(title_part, clean_part, 1)
            if new == old or title_part not in old or not clean_part:
                raise RuntimeError("Expected one generated manual sequence prefix")
            row["question"] = new
            row["question_sha256"] = hashlib.sha256(new.encode()).hexdigest()
            row["expert_review_status"] = "PENDING_REVIEW"
            row["approved_for_training"] = False
            changes.append({"id": row["id"], "reason": "removed_generated_manual_sequence_prefix",
                            "old_question_sha256": hashlib.sha256(old.encode()).hexdigest(),
                            "new_question_sha256": row["question_sha256"]})
    if len(rows) != 300 or len(changes) != 120 or len({r["id"] for r in rows}) != 300:
        raise RuntimeError("Corrected bank count mismatch")
    with TARGET.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    CHANGES.write_text(json.dumps({"original_sha256": EXPECTED_SHA,
                                   "corrected_sha256": sha(TARGET),
                                   "changes": changes,
                                   "expert_review_status": "PENDING_REVIEW",
                                   "training_allowed": False}, indent=2), encoding="utf-8")
    print(json.dumps({"original_sha256": EXPECTED_SHA, "corrected_sha256": sha(TARGET),
                      "questions": 300, "corrected_source_questions": len(changes),
                      "approved_training_labels": 0}))


if __name__ == "__main__":
    main()
