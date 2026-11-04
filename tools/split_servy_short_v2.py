"""Freeze a leakage-aware split for unapproved 300-question V6 candidates."""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
FOLDER = BASE / "servy_short_questions_20261104_01"
BANK = FOLDER / "short_questions_300_v2_candidate.jsonl"
TARGET = FOLDER / "split_v2.json"


def group(row):
    kind = row["question_kind"]
    if kind == "source":
        return f"source:{row['asset_id']}:{row['source_chunk_id']}"
    # The same generic/safety template appears across all three assets.
    return f"{kind}:{row['id'].rsplit('-', 1)[-1]}"


def main():
    if TARGET.exists():
        raise RuntimeError("V2 split already frozen")
    rows = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 300 or len({r["id"] for r in rows}) != 300:
        raise RuntimeError("Corrected bank is incomplete")
    groups = defaultdict(list)
    for row in rows:
        groups[group(row)].append(row)
    split = {}
    for key, members in groups.items():
        bucket = int(hashlib.sha256(("servy-short-v2:" + key).encode()).hexdigest(), 16) % 5
        for row in members:
            split[row["id"]] = "holdout" if bucket == 0 else "development"
    by_text = defaultdict(set)
    by_source = defaultdict(set)
    for row in rows:
        by_text[row["question_sha256"]].add(split[row["id"]])
        if row["question_kind"] == "source":
            by_source[(row["asset_id"], row["source_chunk_id"])].add(split[row["id"]])
    if any(len(value) > 1 for value in by_text.values()) or any(
        len(value) > 1 for value in by_source.values()
    ):
        raise RuntimeError("Question text or source anchor leaked across splits")
    payload = {"dataset_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
               "algorithm": "template-and-source-anchor-group-hash-v2",
               "assignments": split, "counts": dict(Counter(split.values())),
               "approved_labels": 0, "expert_review_status": "PENDING_REVIEW"}
    TARGET.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "cross_split_duplicate_text": 0,
                      "cross_split_source_anchor": 0, "approved_labels": 0}))


if __name__ == "__main__":
    main()
