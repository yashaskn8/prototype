"""Fail-closed read-only admission for next Servy BGE research fine-tuning run.

Input is the private, externally produced V5 111-row candidate subset.
No private source files, models, API calls, or trained weights enter GitHub.
Passing means reproducible RESEARCH fine-tuning inputs, not expert sign-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from servy_v4_mentor_experiment import audited_inputs, provisional_pairs


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def admit(*, split_dir, corpus, families, candidates, candidate_train, manifest):
    integrity = audited_inputs(split_dir, corpus, families, candidates)
    source_train = read_jsonl(Path(split_dir) / "train_unapproved_candidate.jsonl")
    proposed_train = read_jsonl(candidate_train)
    raw_manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if raw_manifest.get("expert_approved") is not False:
        raise ValueError("Candidate manifest cannot claim expert approval")
    if raw_manifest.get("production_promotion_allowed") is not False:
        raise ValueError("Cannot promote machine-triaged research data")
    if raw_manifest.get("candidate_train_sha256", raw_manifest.get("v5_candidate_sha256")) != sha256(candidate_train):
        raise ValueError("Candidate subset hash mismatch")
    if raw_manifest.get("original_train_sha256") != sha256(Path(split_dir) / "train_unapproved_candidate.jsonl"):
        raise ValueError("Original train fingerprint mismatch")
    if raw_manifest.get("accepted_training_queries") != len(proposed_train):
        raise ValueError("Candidate count differs from manifest")
    if not proposed_train or len(proposed_train) > len(source_train):
        raise ValueError("Empty or oversized training subset")
    # Use exact JSON equivalence: AI is not permitted to silently rewrite
    # evidence passages, golds, or negatives during candidate filtering.
    original = {
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in source_train
    }
    selected = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in proposed_train]
    if len(set(selected)) != len(selected) or any(row not in original for row in selected):
        raise ValueError("Candidate subset contains modified or duplicate original rows")

    corpus_rows = read_jsonl(corpus)
    by_chunk = {str(c["chunk_id"]): c for c in corpus_rows}
    if len(by_chunk) != len(corpus_rows):
        raise ValueError("Duplicate corpus IDs")
    pairs = provisional_pairs(proposed_train, by_chunk)
    positive = sum(pair["label"] == 1.0 for pair in pairs)
    negative = sum(pair["label"] == 0.0 for pair in pairs)
    if negative < len(proposed_train):
        raise ValueError("At least one independently verified negative needed per query")
    if not integrity["source_isolation_verified_against_proposed_families"]:
        raise ValueError("Source family split requires verification")
    return {
        "status": "READY_FOR_EXPLORATORY_FINE_TUNING",
        "production_promotion_allowed": False,
        "independent_holdout_verified": False,
        "expert_labels_approved": False,
        "train_queries": len(proposed_train),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "dev_queries": integrity["dev_queries"],
        "quarantined_negative_references": integrity["quarantined_negative_references"],
        "original_train_sha256": sha256(Path(split_dir) / "train_unapproved_candidate.jsonl"),
        "candidate_train_sha256": sha256(candidate_train),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("split-dir", "corpus", "families", "candidates", "candidate-train", "manifest"):
        p.add_argument("--" + option, required=True)
    args = vars(p.parse_args())
    result = admit(**{k.replace("-", "_"): v for k, v in args.items()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
