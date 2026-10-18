"""Servy read-only V4 fine-tuning admission audit (standard library only).

No model downloads, GPU, training, database access, or uploaded private data.
Designed to reject provable data corruption and prominently label unknowns.
Outputs aggregate counts and record IDs ONLY, not source/query/passage text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def norm(value):
    return " ".join(str(value or "").casefold().split())


def ids(row, plural, singular):
    raw = row.get(plural, row.get(singular))
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raw = [raw]
    return [str(value).strip() for value in raw if value is not None and str(value).strip()]


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_num}: invalid JSONL") from exc
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        raise ValueError("Expected JSONL or CSV. Export exact Stage 2 rows as JSONL/CSV.")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: no valid data rows")
    return rows


def make_corpus(rows, add):
    by_chunk, by_passage = {}, defaultdict(set)
    for index, row in enumerate(rows, 1):
        chunk_id = str(row.get("chunk_id", row.get("passage_id", ""))).strip()
        tenant = str(row.get("tenant_id", "")).strip()
        doc_id = str(row.get("document_id", row.get("source_document_id", ""))).strip()
        passage = norm(row.get("passage", row.get("text", "")))
        if not chunk_id or not tenant or not doc_id or not passage:
            add("BLOCKER", "INVALID_CORPUS_ROW", f"corpus:{index}")
            continue
        if chunk_id in by_chunk:
            add("BLOCKER", "DUPLICATE_CHUNK_ID", f"corpus:{index}")
            continue
        key = (tenant, chunk_id)
        by_chunk[chunk_id] = {
            "chunk": chunk_id, "tenant": tenant, "document": doc_id,
            "passage": passage,
            "source_group": norm(row.get("source_group", "")),
            "passage_chars": len(str(row.get("passage", row.get("text", "")))),
            "asset_id": str(row.get("asset_id", "")).strip(),
            "customer_id": str(row.get("customer_id", "")).strip(),
        }
        by_passage[(tenant, passage)].add(chunk_id)
    if not by_chunk:
        add("BLOCKER", "NO_ELIGIBLE_CORPUS", "corpus")
    if any(x["passage_chars"] > 1400 for x in by_chunk.values()):
        add("REVIEW", "CHECK_TOKENIZER_TRUNCATION_FOR_LONG_PASSAGES", "corpus")
    return by_chunk, by_passage


def resolve_source(tenant, known_ids, passage, corpus, by_passage, add, label,
                   allow_no_passage=False):
    doc_ids, seen = set(), set()
    for chunk_id in known_ids:
        chunk = corpus.get(chunk_id)
        if chunk is None:
            add("BLOCKER", "UNKNOWN_CHUNK_ID", label)
            continue
        if chunk["tenant"] != tenant:
            add("BLOCKER", "CROSS_TENANT_SOURCE", label)
            continue
        seen.add(chunk_id)
        doc_ids.add(chunk["document"])
    passage_norm = norm(passage)
    if passage_norm:
        matches = by_passage.get((tenant, passage_norm), set())
        if not matches:
            if not known_ids:
                add("BLOCKER", "PASSAGE_NOT_IN_CORPUS", label)
            else:
                add("BLOCKER", "PASSAGE_TEXT_DIFFERS_FROM_CORPUS", label)
        else:
            if known_ids and not matches.intersection(seen):
                add("BLOCKER", "PASSAGE_CHUNK_ID_CONFLICT", label)
            if not known_ids:
                seen.update(matches)
                doc_ids.update(corpus[x]["document"] for x in matches)
                if len(matches) > 1:
                    # Without an explicit chunk ID, identical text may belong
                    # to different equipment/manual versions. Reject ambiguity.
                    add("BLOCKER", "AMBIGUOUS_IDENTICAL_PASSAGE", label)
    elif not known_ids and not allow_no_passage:
        add("BLOCKER", "UNRESOLVED_PASSAGE_SOURCE", label)
    return seen, doc_ids


def get_positive_text(row):
    values = row.get("positive_texts")
    fallback = values[0] if isinstance(values, list) and values else ""
    return row.get("positive_text") or row.get("positive") or fallback or ""


def get_negative_texts(row):
    texts = []
    for key in ("candidate_negative", "negative_text", "negative"):
        if row.get(key):
            texts.append(str(row[key]))
    values = row.get("negative_texts") or []
    if isinstance(values, list):
        texts.extend(str(x) for x in values if x)
    # Preserve order while deduplicating
    return list(dict.fromkeys(norm(x) for x in texts if norm(x)))


def inspect_split(name, rows, corpus, by_passage, add, allow_unanswerable=False):
    analyzed = []
    query_ids = set()
    for index, row in enumerate(rows, 1):
        tag = f"{name}:{str(row.get('query_id') or index)[:80]}"
        tenant = str(row.get("tenant_id", "")).strip()
        query = norm(row.get("query_text") or row.get("query"))
        if not tenant or not query:
            add("BLOCKER", "MISSING_QUERY_OR_TENANT", tag)
        if not query:
            continue
        qid = str(row.get("query_id") or "").strip()
        if qid:
            if qid in query_ids:
                add("BLOCKER", "DUPLICATE_QUERY_ID", tag)
            query_ids.add(qid)
        if not tenant:
            continue

        pos_ids = ids(row, "positive_chunk_ids", "positive_chunk_id")
        no_positive_expected = bool(allow_unanswerable and not pos_ids
                                    and not row.get("positive_text")
                                    and not row.get("positive"))
        if no_positive_expected and str(row.get("answerability") or "").upper() not in (
            "UNANSWERABLE_FROM_APPROVED_SOURCES", "UNANSWERABLE"
        ):
            add("BLOCKER", "HOLDOUT_NO_GOLD_WITHOUT_UNANSWERABLE_LABEL", tag)
        if not pos_ids and row.get("chunk_id") is not None:
            pos_ids = [str(row["chunk_id"])]
        if not get_positive_text(row) and not allow_unanswerable:
            add("BLOCKER", "MISSING_EXPORTED_TRAINING_PASSAGE", tag)
        pos, docs = resolve_source(
            tenant, pos_ids, get_positive_text(row), corpus, by_passage,
            add, f"{tag}:positive", allow_no_passage=no_positive_expected
        )
        neg_ids = ids(row, "negative_chunk_ids", "negative_chunk_id")
        if row.get("candidate_negative_chunk_id"):
            neg_ids.append(str(row["candidate_negative_chunk_id"]))
        negatives = get_negative_texts(row)
        if neg_ids and negatives:
            # All declared negative IDs must correspond to exported texts,
            # not just one of them. This previously leaked unrelated sources
            # into split comparisons while silently training different pairs.
            declared_texts = {
                corpus[cid]["passage"] for cid in neg_ids
                if cid in corpus and corpus[cid]["tenant"] == tenant
            }
            if declared_texts != set(negatives):
                add("BLOCKER", "NEGATIVE_ID_TEXT_SET_MISMATCH", tag)
        neg, neg_docs = resolve_source(
            tenant, neg_ids, negatives[0] if len(negatives) == 1 else "",
            corpus, by_passage, add, f"{tag}:negative",
            allow_no_passage=not (neg_ids or negatives),
        )
        # Each additional textual negative must be traceable independently
        if len(negatives) > 1:
            for number, passage in enumerate(negatives, 1):
                matching_ids = [
                    cid for cid in neg_ids
                    if cid in corpus and corpus[cid]["passage"] == passage
                ]
                cids, dids = resolve_source(
                    tenant, matching_ids, passage, corpus, by_passage, add,
                    f"{tag}:negative:{number}",
                )
                neg.update(cids)
                neg_docs.update(dids)

        if pos.intersection(neg):
            add("BLOCKER", "CONTRADICTORY_POSITIVE_AND_NEGATIVE", tag)
        if pos and neg and docs.intersection(neg_docs):
            add("REVIEW", "HARD_NEGATIVE_SAME_SOURCE_DOCUMENT", tag)
        pos_text = norm(get_positive_text(row))
        if pos_text and pos_text in negatives:
            add("BLOCKER", "IDENTICAL_POSITIVE_AND_NEGATIVE_TEXT", tag)
        if not pos and not (allow_unanswerable and no_positive_expected):
            add("BLOCKER", "UNRESOLVED_POSITIVE_SOURCE", tag)

        declared_doc = row.get("positive_document_id") or row.get("document_id")
        if declared_doc is not None and docs and str(declared_doc).strip() not in docs:
            add("BLOCKER", "DECLARED_DOCUMENT_MISMATCH", tag)

        # Source-family identity must come from the canonical corpus, not solely
        # from an optional label-row field. Include negative documents too.
        group = norm(row.get("source_group", ""))
        source_groups = {
            corpus[cid]["source_group"] for cid in pos | neg
            if cid in corpus and corpus[cid]["source_group"]
        }
        approved = str(row.get("review_status") or "").lower() == "approved"
        if not approved:
            if not allow_unanswerable:
                add("REVIEW", "NOT_HUMAN_VERIFIED", tag)

        analyzed.append({
            "tag": tag, "query": query, "tenant": tenant,
            "pos": pos, "neg": neg, "pos_docs": docs, "neg_docs": neg_docs,
            "groups": source_groups | ({group} if group else set()),
            "positive_text": pos_text,
            # Full canonical corpus passages, both positives and negatives,
            # detect copied guides under distinct chunk/document/source IDs.
            "passage_texts": {
                corpus[cid]["passage"] for cid in pos | neg if cid in corpus
            },
            "approved": approved,
            "asset_id": str(row.get("asset_id") or "").strip(),
            "customer_id": str(row.get("customer_id") or "").strip(),
        })
    return analyzed


def verify_disjoint(a, b, add, a_name="train", b_name="dev"):
    """An overlap of ANY training passage, including a negative, is suspicious."""
    a_queries = {(x["tenant"], x["query"]) for x in a}
    a_chunks = {x for row in a for x in row["pos"] | row["neg"]}
    a_docs = {x for row in a for x in row["pos_docs"] | row["neg_docs"]}
    a_groups = {x for row in a for x in row["groups"]}
    a_texts = {(r["tenant"], r["positive_text"]) for r in a if r["positive_text"]}
    a_passage_keys = {
        (r["tenant"], passage)
        for r in a for passage in r["passage_texts"] if passage
    }
    for row in b:
        if (row["tenant"], row["query"]) in a_queries:
            add("BLOCKER", "SPLIT_QUERY_COLLISION", row["tag"])
        if (row["pos"] | row["neg"]) & a_chunks:
            add("BLOCKER", "SPLIT_CHUNK_COLLISION", row["tag"])
        if (row["pos_docs"] | row["neg_docs"]) & a_docs:
            add("BLOCKER", "SPLIT_DOCUMENT_COLLISION", row["tag"])
        if row["groups"] & a_groups:
            add("BLOCKER", "SPLIT_SOURCE_GROUP_COLLISION", row["tag"])
        if (row["tenant"], row["positive_text"]) in a_texts:
            add("BLOCKER", "SPLIT_PASSAGE_TEXT_COLLISION", row["tag"])
        if any((row["tenant"], passage) in a_passage_keys
               for passage in row["passage_texts"]):
            add("BLOCKER", "SPLIT_PASSAGE_CONTENT_COLLISION", row["tag"])


def training_collisions(rows, add, corpus):
    by_question = defaultdict(list)
    by_positive = defaultdict(list)
    positive_chunks = set()
    positive_passages = set()
    for row in rows:
        for cid in row["pos"]:
            positive_chunks.add((row["tenant"], cid))
            if cid in corpus:
                positive_passages.add((row["tenant"], corpus[cid]["passage"]))
    for row in rows:
        by_question[(row["tenant"], row["query"])].append(row)
        for cid in row["pos"]:
            by_positive[(row["tenant"], cid)].append(row)
    # In-batch contrastive objectives can optimize a known positive as a
    # negative. A collision anywhere in the training pool is unsafe even
    # when it is absent from the same JSONL row.
    for row in rows:
        for cid in row["neg"]:
            if (row["tenant"], cid) in positive_chunks:
                add("BLOCKER", "TRAIN_NEGATIVE_IS_ANOTHER_POSITIVE", row["tag"])
            if cid in corpus and (row["tenant"], corpus[cid]["passage"]) in positive_passages:
                add("BLOCKER", "TRAIN_NEGATIVE_EQUIVALENT_POSITIVE_TEXT", row["tag"])
    for records in by_question.values():
        if len({r["positive_text"] for r in records}) > 1:
            add("REVIEW", "MULTI_POSITIVE_SAME_QUESTION_IN_BATCH", records[0]["tag"])
        if len(records) > 1 and len({r["positive_text"] for r in records}) == 1:
            add("REVIEW", "REPEATED_QUERY_POSITIVE_PAIR", records[0]["tag"])
    for records in by_positive.values():
        if len({r["query"] for r in records}) > 1:
            add("REVIEW", "SHARED_POSITIVE_ACROSS_QUESTIONS", records[0]["tag"])
    if len(by_question) < 100:
        add("REVIEW", "SMALL_UNIQUE_QUESTION_POPULATION", "training")
    if rows and not any(r["asset_id"] for r in rows):
        add("REVIEW", "NO_EQUIPMENT_ID_FOR_SCOPE_VALIDATION", "training")
    if rows and not any(r["customer_id"] for r in rows):
        add("REVIEW", "NO_CUSTOMER_ID_FOR_SCOPE_VALIDATION", "training")


def audit(train_path, dev_path, corpus_path, stage2_path=None, holdout_path=None, artifact_dir=None):
    findings = []
    def add(severity, code, ref):
        findings.append({"severity": severity, "code": code, "ref": ref})
    corpus, by_passage = make_corpus(read_rows(corpus_path), add)
    train = inspect_split("train", read_rows(train_path), corpus, by_passage, add)
    dev = inspect_split("dev", read_rows(dev_path), corpus, by_passage, add)
    verify_disjoint(train, dev, add)
    # Stage 2 is validated below; perform the complete conflict audit after
    # loading it so that later negatives cannot contradict Stage 1 positives.
    # Stage 2 negative source must NOT leak from dev or independent holdout.
    stage2 = []
    if stage2_path:
        stage2 = inspect_split(
            "stage2", read_rows(stage2_path), corpus, by_passage, add
        )
        if len(stage2) < 8:
            add("REVIEW", "STAGE2_FEW_RESOLVED_EXAMPLES", "stage2")
        # Stage 2 positives may repeat Stage 1 positives, but NEVER dev passages.
        verify_disjoint(stage2, dev, add, "stage2", "dev")
        # Invariant: Stage 2 training must derive from train split sources.
        train_query_keys = {(row["tenant"], row["query"]) for row in train}
        train_docs = {doc for row in train for doc in row["pos_docs"]}
        for row in stage2:
            if (row["tenant"], row["query"]) not in train_query_keys:
                add("BLOCKER", "STAGE2_QUERY_NOT_IN_TRAIN_SPLIT", row["tag"])
            if not row["pos_docs"].issubset(train_docs):
                add("BLOCKER", "STAGE2_POSITIVE_OUTSIDE_TRAIN_DOCS", row["tag"])
    else:
        add("REVIEW", "EXACT_STAGE2_USED_EXAMPLES_NOT_PROVIDED", "stage2")

    training_collisions(train + stage2, add, corpus)

    # Compare against frozen holdout *only for structural leakage*; never score it.
    if holdout_path:
        holdout = inspect_split(
            "holdout", read_rows(holdout_path), corpus, by_passage, add,
            allow_unanswerable=True,
        )
        verify_disjoint(train + stage2, holdout, add, "training", "holdout")
        verify_disjoint(dev, holdout, add, "dev", "holdout")

    if artifact_dir is None:
        add("REVIEW", "PERSISTENT_CHECKPOINT_DESTINATION_UNSET", "artifacts")
    else:
        resolved = str(artifact_dir).replace("\\", "/").rstrip("/")
        if resolved == "/content" or (
            resolved.startswith("/content/") and
            not resolved.startswith("/content/drive/")
        ):
            add("BLOCKER", "CHECKPOINT_TARGET_IS_EPHEMERAL_COLAB_STORAGE", "artifacts")
        elif "/content/drive/" in resolved:
            add("REVIEW", "VERIFY_COMPANY_APPROVAL_FOR_DRIVE_STORAGE", "artifacts")
    count = Counter(x["code"] for x in findings)
    blocking = sum(x["severity"] == "BLOCKER" for x in findings)
    warnings = sum(x["severity"] == "REVIEW" for x in findings)
    return {
        "schema_version": "v4-preflight-1",
        "purpose": "No training. Exact source and split integrity checks.",
        "status": "BLOCKED" if blocking else "REVIEW_REQUIRED",
        "supervision": "WEAK/UNVERIFIED until an authorized domain expert approves relevant labels",
        "fully_independent_test_verified": False,
        "numbers": {
            "corpus_chunks": len(corpus), "train_rows": len(train),
            "dev_rows": len(dev), "stage2_rows": len(stage2),
            "train_unique_queries": len({(r["tenant"], r["query"]) for r in train}),
            "train_source_documents": len({x for r in train for x in r["pos_docs"]}),
            "blockers": blocking, "review_items": warnings,
        },
        "issue_counts": dict(sorted(count.items())),
        "findings": findings[:300],
        "findings_truncated": len(findings) > 300,
        "training_not_performed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Exact first-stage exported train JSONL")
    parser.add_argument("--dev", required=True, help="Separate development JSONL")
    parser.add_argument("--corpus", required=True, help="Tenant-scoped document corpus JSONL")
    parser.add_argument("--stage2", help="EXACT used 27 selected Stage 2 examples JSONL/CSV")
    parser.add_argument("--holdout", help="Optional frozen independent holdout: leakage check only")
    parser.add_argument("--artifact-dir", help="Planned persistent checkpoint destination (no file writes)")
    parser.add_argument("--report", required=True, help="Private local report path")
    args = parser.parse_args()
    report = audit(args.train, args.dev, args.corpus, args.stage2, args.holdout,
                   artifact_dir=args.artifact_dir)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "numbers": report["numbers"],
        "issue_counts": report["issue_counts"], "report_path": str(path),
    }, indent=2))
    raise SystemExit(2 if report["status"] == "BLOCKED" else 0)


if __name__ == "__main__":
    main()
