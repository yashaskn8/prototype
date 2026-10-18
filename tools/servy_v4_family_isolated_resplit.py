"""Offline, non-adjudicating source-family-isolated V4 split proposal.

Every active positive and negative source is an isolation constraint.
No human decision, original file, production system, or model is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from servy_v4_private_reconciliation import (
    digest, normalize, read_jsonl, reconcile, refs, source_sha256,
)


class Blocked(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Blocked(message)


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def resplit(train, dev, corpus, recommendations, quarantine):
    """Create a fully lossless provisional split or explain why impossible."""
    items = [("train", dict(r)) for r in train] + [("dev", dict(r)) for r in dev]
    n = len(items)
    require(n > 1 and dev, "NEED_BOTH_ORIGINAL_SPLITS")
    chunks = {}
    documents = defaultdict(set)
    for row in corpus:
        cid = str(row.get("chunk_id") or "")
        tenant = str(row.get("tenant_id") or "")
        doc = str(row.get("document_id") or "")
        require(cid and cid not in chunks and tenant and doc and str(row.get("passage") or ""),
                "MISSING_OR_DUPLICATE_CORPUS_ID")
        checksum = source_sha256(row)
        require(checksum, "MISSING_ORIGINAL_SOURCE_CHECKSUM:" + cid)
        chunks[cid] = row
        documents[(tenant, doc)].add(checksum)
    require(all(len(h) == 1 for h in documents.values()), "CONFLICTING_SOURCE_CHECKSUM")
    families = {}
    for recommendation in recommendations:
        key = (str(recommendation.get("tenant_id") or ""),
               str(recommendation.get("document_id") or ""))
        family = str(recommendation.get("proposed_canonical_source_family") or "").strip()
        require(key in documents and family and key not in families,
                "AMBIGUOUS_OR_UNKNOWN_FAMILY:" + repr(key))
        require(recommendation.get("evidence_status") == "REGISTRY_METADATA_MATCH",
                "REQUIRE_REVALIDATED_METADATA_MATCH:" + repr(key))
        families[key] = family
    require(set(families) == set(documents), "UNCOVERED_DOCUMENT_FAMILY")

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen_keys = {}
    seen_ids = {}
    pair_counts = Counter()
    for i, (old, row) in enumerate(items):
        tenant = str(row.get("tenant_id") or "")
        qid = str(row.get("query_id") or "")
        question = normalize(row.get("query_text") or row.get("query"))
        require(tenant and qid and question and (tenant, qid) not in seen_ids,
                "DUPLICATE_OR_MISSING_QUERY_IDENTITY:" + qid)
        seen_ids[(tenant, qid)] = old
        positive = refs(row, "positive_chunk_ids") or refs(row, "positive_chunk_id")
        negative = refs(row, "negative_chunk_ids") or refs(row, "negative_chunk_id")
        quarantined = refs(row, "quarantined_negative_chunk_ids")
        require(positive and len(set(positive + negative + quarantined)) ==
                len(positive + negative + quarantined),
                "CONTRADICTORY_SOURCE_REFERENCES:" + qid)
        pair_counts["positive"] += len(positive)
        pair_counts["negative"] += len(negative)
        keys = {("question", question)}
        for cid in positive + negative:
            chunk = chunks.get(cid)
            require(chunk is not None and str(chunk["tenant_id"]) == tenant,
                    "MISSING_OR_CROSS_TENANT_CHUNK:" + qid + "/" + cid)
            doc = str(chunk["document_id"])
            keys.add(("family", tenant, families[(tenant, doc)]))
            keys.add(("document", tenant, doc))
            keys.add(("checksum", next(iter(documents[(tenant, doc)]))))
            keys.add(("passage", normalize(chunk["passage"])))
        for key in keys:
            union(i, seen_keys.setdefault(key, i))
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    groups = sorted(groups.values(), key=lambda g: (len(g), hashlib.sha256(
        "|".join(sorted(str(items[i][1]["query_id"]) for i in g)).encode()).hexdigest()))
    # O(components * total rows); stable subset-sum selection, never splits a component.
    sums = {0: ()}
    for gid, group in enumerate(groups):
        for count, chosen in list(sums.items()):
            new = count + len(group)
            if new < n and new not in sums:
                sums[new] = chosen + (gid,)
    report = {
        "original_train_rows": len(train), "original_dev_rows": len(dev),
        "total_queries": n, "target_dev_rows": len(dev),
        "components": len(groups),
        "component_sizes_descending": sorted((len(g) for g in groups), reverse=True),
        "original_pair_counts": dict(pair_counts),
        "human_approval_complete": False, "training_ready": False,
        "machine_family_proposals_only": True, "original_files_modified": False,
    }
    feasible = [count for count in sums if 0 < count < n]
    if not feasible:
        report["status"] = "BLOCKED_SINGLE_CONNECTED_COMPONENT"
        report["reason"] = ("All queries connected through active positives/negatives, source "
                            "families, checksums, passages, or normalized questions. "
                            "A no-drop split is impossible without independent expert "
                            "review of labels/family definitions.")
        return None, report
    wanted = min(feasible, key=lambda count: (abs(count - len(dev)), count))
    chosen = set(sums[wanted])
    membership = {i: ("dev" if gid in chosen else "train")
                  for gid, group in enumerate(groups) for i in group}
    new_train, new_dev, moved = [], [], []
    owners = {}
    for i, (old, row) in enumerate(items):
        updated = membership[i]
        owners[(str(row["tenant_id"]), str(row["query_id"]))] = updated
        (new_dev if updated == "dev" else new_train).append(row)
        if old != updated:
            moved.append({"tenant_id": str(row["tenant_id"]),
                          "query_id": str(row["query_id"]),
                          "from": old, "to": updated})
    new_ledger = []
    for record in quarantine:
        qid, old = str(record.get("query_id") or ""), str(record.get("split") or "")
        matches = [key for key, split in seen_ids.items()
                   if key[1] == qid and split == old]
        require(len(matches) == 1, "AMBIGUOUS_QUARANTINE_OWNER:" + old + "/" + qid)
        revised = dict(record)
        revised["split"] = owners[matches[0]]
        new_ledger.append(revised)
    require(len(new_train) + len(new_dev) == n and len(new_ledger) == len(quarantine),
            "SILENT_DATA_LOSS")
    report.update({
        "status": "PROVISIONAL_RESPLIT_ONLY",
        "new_train_rows": len(new_train), "new_dev_rows": len(new_dev),
        "moved_query_count": len(moved), "moved_query_ids": moved,
        "quarantined_negatives_preserved": len(new_ledger),
    })
    return (new_train, new_dev, new_ledger), report


def run(bundle, corpus, recommendations, quarantine, out_dir):
    bundle, corpus, recommendations, quarantine, out_dir = map(
        lambda x: Path(x).resolve(), (bundle, corpus, recommendations, quarantine, out_dir))
    require("private_training_data" in {p.casefold() for p in out_dir.parts},
            "PRIVATE_OUTPUT_DIRECTORY_REQUIRED")
    require(not out_dir.exists(), "NO_OVERWRITE_ALLOWED")
    inputs = {
        "train": bundle / "train_unapproved_candidate.jsonl",
        "dev": bundle / "dev_unapproved_candidate.jsonl",
        "source_queue": bundle / "source_family_review_queue.jsonl",
        "label_queue": bundle / "expert_review_queue.jsonl",
        "corpus": corpus, "recommendations": recommendations, "quarantine": quarantine,
    }
    for name, path in inputs.items():
        require(path.is_file(), "MISSING_INPUT:" + name + ":" + str(path))
    loaded = {name: read_jsonl(path) for name, path in inputs.items()}
    manifest_path = recommendations.parent / "ai_source_evidence_manifest.json"
    require(manifest_path.is_file(), "MISSING_AI_EVIDENCE_MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("corpus_sha256") == digest(corpus),
            "AI_EVIDENCE_CORPUS_HASH_MISMATCH")
    require(manifest.get("documents_analyzed") == len(loaded["recommendations"]),
            "AI_EVIDENCE_DOCUMENT_COUNT_MISMATCH")
    # Never silently discard a prior recorded decision, even if proposed.
    require(all(x.get("decision") == "PENDING" for x in loaded["source_queue"]),
            "SOURCE_REVIEWS_PRESENT_STOP_FOR_HUMAN_MIGRATION")
    require(all(ref.get("decision") == "PENDING"
                for row in loaded["label_queue"] for ref in row.get("references", [])),
            "LABEL_REVIEWS_PRESENT_STOP_FOR_HUMAN_MIGRATION")
    output, report = resplit(loaded["train"], loaded["dev"], loaded["corpus"],
                             loaded["recommendations"], loaded["quarantine"])
    report["input_sha256"] = {name: digest(path) for name, path in inputs.items()}
    out_dir.mkdir(parents=True, exist_ok=False)
    if output is None:
        (out_dir / "resplit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    new_train, new_dev, ledger = output
    if not all(any(refs(row, "negative_chunk_ids") or refs(row, "negative_chunk_id")
                   for row in split) for split in (new_train, new_dev)):
        report["status"] = "BLOCKED_ONE_CLASS_SPLIT"
        (out_dir / "resplit_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report
    structural, proposals, queue, sources = reconcile(
        new_train, new_dev, loaded["corpus"], quarantine=ledger)
    report["reconciliation_issues"] = structural["issue_counts"]
    report["reconciliation_overlap"] = structural["train_dev_overlap"]
    report["quarantine_consistent"] = structural["quarantine_ledger_matches_split_proposals"]
    if (any(structural["train_dev_overlap"].values()) or
        not structural["quarantine_ledger_matches_split_proposals"] or
        any(x != "CONFLICTING_SOURCE_GROUP_CLAIMS" for x in structural["issue_counts"])):
        report["status"] = "BLOCKED_RECONCILIATION"
        (out_dir / "resplit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    require({(str(x.get("tenant_id")), str(x.get("document_id"))) for x in sources} ==
            {(str(x.get("tenant_id")), str(x.get("document_id")))
             for x in loaded["source_queue"]},
            "SOURCE_REVIEW_QUEUE_NO_LONGER_MATCHES_CORPUS")
    artifacts = {
        "train_unapproved_candidate.jsonl": proposals["train"],
        "dev_unapproved_candidate.jsonl": proposals["dev"],
        "source_family_review_queue.jsonl": loaded["source_queue"],
        "expert_review_queue.jsonl": queue,
        "quarantined_negatives.jsonl": ledger,
    }
    for name, rows in artifacts.items():
        write_rows(out_dir / name, rows)
    report["output_sha256"] = {name: digest(out_dir / name) for name in artifacts}
    (out_dir / "readiness_report.json").write_text(json.dumps(structural, indent=2), encoding="utf-8")
    (out_dir / "resplit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--family-recommendations", required=True)
    p.add_argument("--quarantine", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    report = run(args.bundle, args.corpus, args.family_recommendations,
                 args.quarantine, args.out_dir)
    # Avoid flooding PowerShell with hundreds of moved query identifiers.
    print(json.dumps({k: v for k, v in report.items() if k != "moved_query_ids"}, indent=2))
    if report["status"] != "PROVISIONAL_RESPLIT_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
