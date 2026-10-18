"""Finalize explicitly adjudicated private Servy V4 reviews into candidate training inputs.

Offline, no ML runtime, no database, no upload. This validates the *structure*
and audit trail of externally supplied expert decisions; it cannot verify human
identity, competence, actual source authority or corporate Colab permission.
Nothing here automatically approves, infers, or drops weak labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from servy_v4_private_reconciliation import read_jsonl, refs, normalize, source_sha256


class ReviewBlocked(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ReviewBlocked(reason)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(train, dev, corpus, reviews, families, reviewer_manifest):
    allowed = reviewer_manifest.get("authorized_reviewer_ids")
    require(isinstance(allowed, list) and allowed and
            all(isinstance(x, str) and x.strip() for x in allowed),
            "Missing reviewer allowlist")
    allowed = set(allowed)
    require(bool(str(reviewer_manifest.get("approval_reference") or "").strip()),
            "External reviewer authorization reference required")

    chunks, documents = {}, defaultdict(list)
    for chunk in corpus:
        cid = str(chunk.get("chunk_id") or "")
        tenant = str(chunk.get("tenant_id") or "")
        doc = str(chunk.get("document_id") or "")
        require(cid and tenant and doc and str(chunk.get("passage") or "").strip(),
                "Invalid corpus record")
        require(cid not in chunks, "Duplicate chunk ID")
        chunks[cid] = chunk
        documents[(tenant, doc)].append(chunk)

    doc_review = {}
    for item in families:
        key = (str(item.get("tenant_id") or ""), str(item.get("document_id") or ""))
        require(key in documents and key not in doc_review,
                "Missing source document or duplicate source-family decision")
        doc_review[key] = item
    require(set(doc_review) == set(documents), "All source documents require an explicit review")

    corpus_export = []
    checksum_families = defaultdict(set)
    for key, source_chunks in documents.items():
        decision = doc_review[key]
        family = str(decision.get("verified_canonical_source_family") or "").strip()
        version = str(decision.get("approved_source_version") or "").strip()
        reviewer = str(decision.get("reviewer_id") or "").strip()
        evidence = str(decision.get("evidence_reference") or "").strip()
        require(decision.get("decision") == "approved" and family and version
                and reviewer in allowed and evidence,
                "Unapproved or unattested source-family mapping: " + repr(key))
        declared_versions = {
            str(chunk.get("source_version") or "").strip() for chunk in source_chunks
        }
        require(declared_versions == {version}, "Source-version mismatch: " + repr(key))
        try:
            hashes = {source_sha256(chunk) for chunk in source_chunks}
        except ValueError as error:
            raise ReviewBlocked(str(error)) from error
        require(hashes and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes),
                "Missing or malformed original source checksum: " + repr(key))
        require(len(hashes) == 1, "Conflicting document checksums: " + repr(key))
        expected_hashes = decision.get("source_document_sha256")
        require(isinstance(expected_hashes, list) and set(expected_hashes) == hashes,
                "Reviewed checksum differs from original source bytes: " + repr(key))
        for h in hashes:
            checksum_families[h].add(family)
        for chunk in source_chunks:
            upgraded = dict(chunk)
            upgraded["source_group"] = family
            upgraded["source_group_status"] = "VERIFIED"
            upgraded["source_group_reviewer_id"] = reviewer
            upgraded["source_group_evidence_reference"] = evidence
            corpus_export.append(upgraded)
    require(all(len(names) == 1 for names in checksum_families.values()),
            "Identical original source checksum assigned conflicting canonical families")

    decisions = {}
    for review in reviews:
        key = (str(review.get("split") or ""), str(review.get("query_id") or ""))
        require(key[0] in ("train", "dev") and key[1] and key not in decisions,
                "Invalid or duplicate label review identity")
        decisions[key] = review

    exports, inv = {}, {}
    used_reviews = set()
    for split, rows in (("train", train), ("dev", dev)):
        output = []
        all_families, all_docs, all_texts, all_queries = set(), set(), set(), set()
        global_checksums, global_passages, global_questions = set(), set(), set()
        for row in rows:
            qid = str(row.get("query_id") or "")
            tenant = str(row.get("tenant_id") or "")
            question = str(row.get("query_text") or row.get("query") or "").strip()
            key = (split, qid)
            require(qid and tenant and question and key in decisions,
                    "Unreviewed training example: " + repr(key))
            review = decisions[key]
            require(str(review.get("query_text") or "").strip() == question,
                    "Reviewed question text differs from training export: " + repr(key))
            used_reviews.add(key)
            pos = refs(row, "positive_chunk_ids") or refs(row, "positive_chunk_id")
            neg = refs(row, "negative_chunk_ids") or refs(row, "negative_chunk_id")
            require(pos and len(pos + neg) == len(set(pos + neg)),
                    "Missing/duplicate source IDs: " + repr(key))
            expected = {("positive", cid) for cid in pos} | {
                ("negative", cid) for cid in neg}
            details = review.get("references")
            require(isinstance(details, list), "Missing per-passage review details")
            actual = {}
            for entry in details:
                role = entry.get("role")
                cid = str(entry.get("chunk_id") or "")
                pair = (role, cid)
                require(pair in expected and pair not in actual,
                        "Unexpected/duplicate passage review: " + repr(key))
                actual[pair] = entry
            require(set(actual) == expected, "Every positive and negative needs a decision: " + repr(key))
            pos_texts, neg_texts = [], []
            pos_reviewers, neg_reviewers = set(), set()
            positive_families = set()
            for role, cid in sorted(expected, key=lambda x: (x[0] != "positive",
                                                            (pos + neg).index(x[1]))):
                chunk = chunks.get(cid)
                require(chunk is not None and str(chunk["tenant_id"]) == tenant,
                        "Missing or cross-tenant source: " + repr(key))
                doc = str(chunk["document_id"])
                require(str(actual[(role, cid)].get("document_id")) == doc,
                        "Review source document ID mismatch: " + repr(key))
                require(str(actual[(role, cid)].get("passage")) == str(chunk["passage"]),
                        "Passage text changed after review: " + repr(key))
                reviewer = str(actual[(role, cid)].get("reviewer_id") or "").strip()
                evidence = str(actual[(role, cid)].get("review_evidence_reference") or "").strip()
                require(actual[(role, cid)].get("decision") == "approved" and
                        reviewer in allowed and evidence,
                        "Expert passage decision missing or rejected: " + repr(key))
                fam = str(doc_review[(tenant, doc)]["verified_canonical_source_family"]).strip()
                all_families.add((tenant, fam))
                all_docs.add((tenant, doc))
                all_texts.add((tenant, normalize(chunk["passage"])))
                global_passages.add(normalize(chunk["passage"]))
                global_checksums.update(
                    str(checksum).lower()
                    for checksum in doc_review[(tenant, doc)]["source_document_sha256"]
                )
                if role == "positive":
                    pos_texts.append(str(chunk["passage"]))
                    pos_reviewers.add(reviewer)
                    positive_families.add(fam)
                else:
                    neg_texts.append(str(chunk["passage"]))
                    neg_reviewers.add(reviewer)
            # Do not silently discard older exported passage aliases that
            # contradict the actual adjudicated source passages.
            for alias in ("positive_text", "positive"):
                old = str(row.get(alias) or "").strip()
                if old:
                    require(old in pos_texts,
                            "Legacy positive passage conflicts with reviewed source: " + repr(key))
            for alias in ("negative_text", "negative", "candidate_negative"):
                old = str(row.get(alias) or "").strip()
                if old:
                    require(old in neg_texts,
                            "Legacy negative passage conflicts with reviewed source: " + repr(key))
            if isinstance(row.get("positive_texts"), list) and row["positive_texts"]:
                require(row["positive_texts"] == pos_texts,
                        "Original positive passage list differs from reviewed sources: " + repr(key))
            if isinstance(row.get("negative_texts"), list) and row["negative_texts"]:
                require(row["negative_texts"] == neg_texts,
                        "Original negative passage list differs from reviewed sources: " + repr(key))
            require(len(pos_reviewers) == 1 and len(neg_reviewers) <= 1,
                    "Different reviewers across one label row need separate adjudication")
            require(len(positive_families) == 1,
                    "Multiple positive families in one row need explicit group adjudication")
            require(not ({normalize(x) for x in pos_texts} &
                         {normalize(x) for x in neg_texts}),
                    "Identical positive/negative passage for one query")
            approved = dict(row)
            # The canonical exporter owns the training text fields. Remove
            # equivalent legacy aliases so the BCE runner cannot double-count
            # the same passage through multiple schema spellings.
            for alias in ("positive", "negative", "negative_text", "candidate_negative"):
                approved.pop(alias, None)
            approved["positive_texts"] = pos_texts
            approved["positive_text"] = pos_texts[0]
            approved["negative_texts"] = neg_texts
            approved["review_status"] = "approved"
            approved["reviewer_id"] = next(iter(pos_reviewers))
            if neg_texts:
                approved["negative_review_status"] = "approved"
                approved["negative_reviewer_id"] = next(iter(neg_reviewers))
            else:
                approved.pop("negative_review_status", None)
                approved.pop("negative_reviewer_id", None)
            approved["source_group"] = next(iter(positive_families))
            approved["reviewed_provenance_reference"] = reviewer_manifest["approval_reference"]
            output.append(approved)
            all_queries.add((tenant, normalize(question)))
            global_questions.add(normalize(question))
        exports[split] = output
        inv[split] = {
            "families": all_families, "docs": all_docs,
            "passages": all_texts, "queries": all_queries,
            "source_checksums_global": global_checksums,
            "passages_global": global_passages,
            "queries_global": global_questions,
        }
    require(used_reviews == set(decisions), "Orphan passage-review decisions")
    require(all(not (inv["train"][key] & inv["dev"][key])
                for key in ("families", "docs", "passages", "queries",
                            "source_checksums_global", "passages_global",
                            "queries_global")),
            "Reviewed source families/documents/passages/queries leak across train and dev")
    require(all(any(x.get("negative_texts") for x in exports[split])
                for split in ("train", "dev")),
            "BCE train and dev both require at least one reviewed negative")
    return exports, corpus_export, {
        "result": "REVIEW_RECORDS_STRUCTURALLY_CONSISTENT",
        "train_rows": len(exports["train"]),
        "dev_rows": len(exports["dev"]),
        "corpus_chunks": len(corpus_export),
        "reviewed_documents": len(documents),
        "reviewed_pairs": sum(
            len(x["positive_texts"]) + len(x["negative_texts"])
            for rows in exports.values() for x in rows),
        "note": ("Structural checks cannot prove human expertise, company permission, "
                 "or independent final evaluation. A separate strict preflight "
                 "and externally authorized input-hash manifest are still required."),
        "training_not_performed": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-bundle", required=True)
    p.add_argument("--reviewer-manifest", required=True,
                   help="External, locally held authorized reviewers and reference")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--corpus", required=True)
    a = p.parse_args()
    bundle, target = Path(a.review_bundle), Path(a.out_dir)
    if "private_training_data" not in {part.lower() for part in target.parts}:
        p.error("Approved source passages must remain in private_training_data")
    if target.exists():
        p.error("Refusing to overwrite any prior finalization")
    paths = {name: bundle / filename for name, filename in {
        "train": "train_unapproved_candidate.jsonl",
        "dev": "dev_unapproved_candidate.jsonl",
        "review": "expert_review_queue.jsonl",
        "families": "source_family_review_queue.jsonl",
    }.items()}
    # Use the unchanged local corpus; never infer it from review comments.
    paths["corpus"] = Path(a.corpus)
    manifest = json.loads(Path(a.reviewer_manifest).read_text(encoding="utf-8"))
    report_path = bundle / "readiness_report.json"
    if report_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        for role, file in paths.items():
            if role == "corpus" and prior.get("input_sha256", {}).get("corpus"):
                require(sha256(file) == prior["input_sha256"]["corpus"],
                        "Original corpus differs from recovery report")
    rows = {role: read_jsonl(file) for role, file in paths.items()}
    exports, upgraded, result = finalize(
        rows["train"], rows["dev"], rows["corpus"],
        rows["review"], rows["families"], manifest)
    target.mkdir(parents=True, exist_ok=False)
    files = {
        "train_reviewed.jsonl": exports["train"],
        "dev_reviewed.jsonl": exports["dev"],
        "corpus_reviewed.jsonl": upgraded,
    }
    for name, items in files.items():
        with (target / name).open("x", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["input_sha256"] = {name: sha256(file) for name, file in paths.items()}
    result["output_sha256"] = {name: sha256(target / name) for name in files}
    result["reviewer_manifest_sha256"] = sha256(a.reviewer_manifest)
    with (target / "finalization_report.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"status": result["result"], "output": str(target),
                      "reviewed_pairs": result["reviewed_pairs"],
                      "training_ready": False}, indent=2))


if __name__ == "__main__":
    main()
