"""Local-only Servy V4 recovery reconciliation and expert-review preparation.

Reads private data already on the operator's computer, not GitHub. Writes
unapproved candidate exports and review packages under private_training_data.
No DB access, training, model downloads, approvals, or public uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected nonempty JSONL object records")
    return rows


def refs(row, field):
    val = row.get(field) or []
    if not isinstance(val, list):
        val = [val]
    return [str(x) for x in val if x is not None and str(x).strip()]


def normalize(text):
    return " ".join(re.findall(r"\w+", str(text or "").casefold()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconcile(train, dev, corpus, quarantine=(), historical_holdout=()):
    """Structural evidence only. Never certifies expert labels or source groups."""
    issues = Counter()
    by_id = {}
    documents = defaultdict(lambda: {"ids": set(), "groups": set(),
                                     "versions": set(), "checksums": set()})
    for chunk in corpus:
        cid = str(chunk.get("chunk_id") or "").strip()
        tid = str(chunk.get("tenant_id") or "").strip()
        doc = str(chunk.get("document_id") or "").strip()
        text = str(chunk.get("passage") or "").strip()
        if not (cid and tid and doc and text):
            issues["INVALID_CORPUS_RECORD"] += 1
            continue
        if cid in by_id:
            issues["DUPLICATE_CHUNK_ID"] += 1
            continue
        by_id[cid] = chunk
        info = documents[(tid, doc)]
        info["ids"].add(cid)
        if chunk.get("source_group"):
            info["groups"].add(str(chunk["source_group"]).strip())
        if chunk.get("source_version"):
            info["versions"].add(str(chunk["source_version"]).strip())
        checksum = str(chunk.get("source_checksum_sha256") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", checksum):
            info["checksums"].add(checksum)
        elif checksum:
            issues["INVALID_SOURCE_CHECKSUM"] += 1

    inventory = {}
    review = []
    proposed = {}
    expected_quarantine = Counter()
    active_negative_references = set()
    row_by_split_qid = {}
    for split, rows in (("train", train), ("dev", dev)):
        inv = {"chunks": set(), "docs": set(), "checksums": set(),
               "passages": set(), "queries": set()}
        new_rows = []
        seen_qids = set()
        for row in rows:
            tid = str(row.get("tenant_id") or "").strip()
            qid = str(row.get("query_id") or "").strip()
            query = str(row.get("query_text") or row.get("query") or "").strip()
            if not tid or not qid or not query:
                issues["INCOMPLETE_QUERY_IDENTITY"] += 1
            if (tid, qid) in seen_qids:
                issues["DUPLICATE_QUERY_ID"] += 1
            seen_qids.add((tid, qid))
            if (split, qid) in row_by_split_qid:
                # The quarantine ledger identifies rows by split/query_id,
                # not tenant. Ambiguous IDs cannot be reconciled reliably.
                issues["AMBIGUOUS_QUARANTINE_QUERY_ID"] += 1
            row_by_split_qid[(split, qid)] = tid
            inv["queries"].add((tid, normalize(query)))
            pos_ids = refs(row, "positive_chunk_ids") or refs(row, "positive_chunk_id")
            neg_ids = refs(row, "negative_chunk_ids") or refs(row, "negative_chunk_id")
            quarantined_ids = refs(row, "quarantined_negative_chunk_ids")
            if len(quarantined_ids) != len(set(quarantined_ids)):
                issues["DUPLICATED_QUARANTINE_IN_PROPOSAL"] += 1
            if len(neg_ids) != len(set(neg_ids)):
                issues["DUPLICATED_ACTIVE_NEGATIVE"] += 1
            if set(neg_ids) & set(quarantined_ids):
                issues["QUARANTINED_NEGATIVE_STILL_ACTIVE"] += 1
            if set(pos_ids) & set(quarantined_ids):
                issues["QUARANTINE_CONTAINS_POSITIVE"] += 1
            for cid in neg_ids:
                active_negative_references.add((split, qid, cid))
            for cid in quarantined_ids:
                expected_quarantine[(split, qid, cid)] += 1
                chunk = by_id.get(cid)
                if chunk is None:
                    issues["QUARANTINED_REFERENCE_MISSING_FROM_CORPUS"] += 1
                elif str(chunk["tenant_id"]) != tid:
                    issues["QUARANTINED_REFERENCE_CROSS_TENANT"] += 1
            if not pos_ids:
                issues["MISSING_POSITIVE_REFERENCE"] += 1
            if set(pos_ids) & set(neg_ids):
                issues["SAME_ROW_POSITIVE_NEGATIVE_ID"] += 1
            positives, negatives, evidence = [], [], []
            for role, ids, output in (
                ("positive", pos_ids, positives), ("negative", neg_ids, negatives)):
                for cid in ids:
                    chunk = by_id.get(cid)
                    if not chunk:
                        issues["MISSING_CHUNK_REFERENCE"] += 1
                        continue
                    if str(chunk["tenant_id"]) != tid:
                        issues["CROSS_TENANT_REFERENCE"] += 1
                        continue
                    output.append(str(chunk["passage"]))
                    document = str(chunk["document_id"])
                    info = documents[(tid, document)]
                    inv["docs"].add((tid, document))
                    inv["chunks"].add((tid, cid))
                    inv["passages"].add((tid, normalize(chunk["passage"])))
                    inv["checksums"].update((tid, sha) for sha in info["checksums"])
                    # A label row's source_group describes its positive
                    # evidence, not every candidate negative's manual.
                    if role == "positive" and row.get("source_group"):
                        info["groups"].add(str(row["source_group"]).strip())
                    evidence.append({"role": role, "chunk_id": cid,
                                     "document_id": document,
                                     "passage": chunk["passage"]})
            if set(map(normalize, positives)) & set(map(normalize, negatives)):
                issues["SAME_ROW_POSITIVE_NEGATIVE_PASSAGE"] += 1
            recovered = dict(row)
            recovered.pop("source_group", None)
            recovered["unverified_declared_source_group"] = row.get("source_group")
            recovered["positive_texts"] = positives
            recovered["negative_texts"] = negatives
            recovered["review_status"] = row.get("review_status", "unverified")
            recovered["reconciliation_status"] = "UNAPPROVED_CANDIDATE"
            # Keep the true reviewed negative state only if it already exists.
            new_rows.append(recovered)
            review.append({
                "split": split, "query_id": qid, "query_text": query,
                "source_group_claim": row.get("source_group"),
                "original_review_status": row.get("review_status"),
                "references": evidence,
                "positive_decision": "PENDING",
                "negative_decision": "PENDING" if neg_ids else "NOT_APPLICABLE",
                "reviewer_id": None,
                "review_evidence_reference": None,
            })
        inventory[split] = inv
        proposed[split] = new_rows

    observed_quarantine = Counter()
    for entry in quarantine:
        split = str(entry.get("split") or "").strip()
        qid = str(entry.get("query_id") or "").strip()
        cid = str(entry.get("negative_chunk_id") or "").strip()
        if split not in ("train", "dev") or not qid or not cid:
            issues["INVALID_QUARANTINE_LEDGER_ENTRY"] += 1
            continue
        key = (split, qid, cid)
        observed_quarantine[key] += 1
        if (split, qid) not in row_by_split_qid:
            issues["QUARANTINE_ORPHAN_QUERY"] += 1
        if key in active_negative_references:
            issues["QUARANTINE_LEDGER_REFERENCE_STILL_ACTIVE"] += 1
        if entry.get("review_status") == "approved":
            # Historical quarantine is a pending decision, never approved
            # merely because it was present in a reconstructed ledger.
            issues["QUARANTINE_UNEXPECTED_APPROVED_STATUS"] += 1
    if expected_quarantine != observed_quarantine:
        missing = expected_quarantine - observed_quarantine
        extra = observed_quarantine - expected_quarantine
        issues["QUARANTINE_LEDGER_MISSING_REFERENCES"] += sum(missing.values())
        issues["QUARANTINE_LEDGER_EXTRA_REFERENCES"] += sum(extra.values())

    for doc in documents.values():
        if len(doc["groups"]) > 1:
            issues["CONFLICTING_SOURCE_GROUP_CLAIMS"] += 1
        if len(doc["versions"]) > 1:
            issues["CONFLICTING_SOURCE_VERSION"] += 1
        if len(doc["checksums"]) > 1:
            issues["CONFLICTING_DOCUMENT_CHECKSUM"] += 1

    train_i, dev_i = inventory["train"], inventory["dev"]
    overlap = {
        "chunks": len(train_i["chunks"] & dev_i["chunks"]),
        "documents": len(train_i["docs"] & dev_i["docs"]),
        "checksums": len(train_i["checksums"] & dev_i["checksums"]),
        "identical_passages": len(train_i["passages"] & dev_i["passages"]),
        "normalized_queries": len(train_i["queries"] & dev_i["queries"]),
    }
    if any(overlap.values()):
        issues["TRAIN_DEV_LEAKAGE"] += 1

    missing_holdout_gold = set()
    holdout_docs = set()
    for row in historical_holdout:
        tid = str(row.get("tenant_id") or "").strip()
        for field in ("positive_chunk_ids", "negative_chunk_ids"):
            for cid in refs(row, field):
                if cid not in by_id:
                    if field == "positive_chunk_ids":
                        missing_holdout_gold.add(cid)
                    continue
                holdout_docs.add((tid, str(by_id[cid]["document_id"])))
    holdout_report = {
        "role": "HISTORICAL_DIAGNOSTIC_ONLY",
        "unresolved_positive_chunk_ids": len(missing_holdout_gold),
        "shared_train_documents": len(holdout_docs & train_i["docs"]),
        "shared_dev_documents": len(holdout_docs & dev_i["docs"]),
        "independent": False,
    }
    source_review = []
    for (tid, doc), info in sorted(documents.items()):
        source_review.append({
            "tenant_id": tid, "document_id": doc,
            "chunk_count": len(info["ids"]),
            "document_versions": sorted(info["versions"]),
            "source_document_sha256": sorted(info["checksums"]),
            "unverified_declared_groups": sorted(info["groups"]),
            "conflicting_group_claims": len(info["groups"]) > 1,
            "verified_canonical_source_family": None,
            "approved_source_version": None,
            "reviewer_id": None, "evidence_reference": None,
        })
    quarantine_refs = len(quarantine)
    report = {
        "status": ("BLOCKED_DATA_INTEGRITY" if any(
            name not in {"CONFLICTING_SOURCE_GROUP_CLAIMS"}
            for name in issues
        ) else "BLOCKED_SOURCE_FAMILY_VERIFICATION"),
        "corpus_chunks": len(by_id),
        "train_rows": len(train), "dev_rows": len(dev),
        "quarantined_negatives_preserved": quarantine_refs,
        "quarantine_ledger_matches_split_proposals":
            expected_quarantine == observed_quarantine
            and not any(key.startswith("QUARANTINE_") or
                        key.startswith("DUPLICATED_QUARANTINE") or
                        key == "AMBIGUOUS_QUARANTINE_QUERY_ID"
                        for key in issues),
        "train_dev_overlap": overlap,
        "historical_holdout": holdout_report,
        "documents_requiring_canonical_family_review": len(source_review),
        "unapproved_label_rows": sum(
            1 for x in train + dev if x.get("review_status") != "approved"),
        "unapproved_negative_rows": sum(
            1 for x in train + dev
            if (refs(x, "negative_chunk_ids") or refs(x, "negative_chunk_id"))
            and (x.get("negative_review_status") != "approved"
                 or not x.get("negative_reviewer_id"))),
        "conflicting_declared_document_groups":
            issues["CONFLICTING_SOURCE_GROUP_CLAIMS"],
        "issue_counts": dict(sorted(issues.items())),
        "source_family_verification_complete": False,
        "expert_label_verification_complete": False,
        "new_independent_holdout_verified": False,
        "ready_for_training": False,
    }
    return report, proposed, review, source_review


def process(proposal_dir, corpus_path, historical_holdout_path, output):
    proposal = Path(proposal_dir)
    output = Path(output)
    if "private_training_data" not in {s.lower() for s in output.parts}:
        raise ValueError("Outputs must stay in private_training_data")
    if output.exists():
        raise FileExistsError("Existing report directory will not be overwritten")
    paths = {
        "train": proposal / "train_proposed.jsonl",
        "dev": proposal / "dev_proposed.jsonl",
        "corpus": Path(corpus_path),
        "quarantine": proposal / "quarantined_negatives.jsonl",
    }
    if historical_holdout_path and Path(historical_holdout_path).exists():
        paths["historical_holdout"] = Path(historical_holdout_path)
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    report, proposed, queue, families = reconcile(
        rows["train"], rows["dev"], rows["corpus"],
        rows["quarantine"], rows.get("historical_holdout", ()))
    report["input_sha256"] = {name: digest(path) for name, path in paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    to_write = {
        "train_unapproved_candidate.jsonl": proposed["train"],
        "dev_unapproved_candidate.jsonl": proposed["dev"],
        "expert_review_queue.jsonl": queue,
        "source_family_review_queue.jsonl": families,
    }
    for name, items in to_write.items():
        with (output / name).open("x", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "readiness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "corpus_chunks": report["corpus_chunks"],
        "train_rows": report["train_rows"],
        "dev_rows": report["dev_rows"],
        "review_rows": len(queue),
        "documents_requiring_review": len(families),
        "train_dev_overlap": report["train_dev_overlap"],
        "ready_for_training": False,
        "report": str(output / "readiness_report.json"),
    }, indent=2))
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--proposal-dir")
    cli.add_argument("--corpus")
    cli.add_argument("--historical-holdout")
    cli.add_argument("--out-dir")
    a = cli.parse_args()
    base = Path("private_training_data")
    recovery = base / "recovery_v4"
    proposals = sorted(recovery.glob("split_proposal_*"))
    corpora = sorted(recovery.glob("servy_corpus_recovered_*.jsonl"))
    if not (a.proposal_dir or proposals) or not (a.corpus or corpora):
        cli.error("Recovery inputs missing; no data reconstructed or invented")
    prop = Path(a.proposal_dir) if a.proposal_dir else proposals[-1]
    corpus = Path(a.corpus) if a.corpus else corpora[-1]
    holdout = (a.historical_holdout or
               str(base / "frozen_independent_holdout_benchmark.jsonl"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(a.out_dir) if a.out_dir else recovery / ("review_bundle_" + stamp)
    result = process(prop, corpus, holdout, target)
    raise SystemExit(0 if result["ready_for_training"] else 2)


if __name__ == "__main__":
    main()
