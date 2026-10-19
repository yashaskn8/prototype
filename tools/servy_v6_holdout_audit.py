"""Fail-closed offline assessment of Servy holdout independence and scorable evidence.

Inputs remain private and read-only. No new holdout or labels are invented.
Review metadata is reported, never treated as authenticated authorization.
A clean structural result still does NOT prove chronological independence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


# Categories observed in the locally reviewed holdout. These names describe
# possible test intent only; they are not authenticated labels or proof that
# an abstention/refusal answer is correct.
POTENTIAL_ABSTENTION_OR_SECURITY_CATEGORIES = frozenset({
    "unsupported_question",
    "doc_id_attack",
    "revoked_doc_query",
    "tenant_isolation_attack",
    "out_of_domain",
    "boundary_injection",
})


class HoldoutAuditError(ValueError):
    pass


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise HoldoutAuditError("MISSING_INPUT:" + path.name)
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HoldoutAuditError(
                    f"INVALID_JSONL:{path.name}:{number}"
                ) from exc
            if not isinstance(value, dict):
                raise HoldoutAuditError("NON_OBJECT_RECORD:" + path.name)
            rows.append(value)
    if not rows:
        raise HoldoutAuditError("EMPTY_INPUT:" + path.name)
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def refs(row, role):
    value = row.get(role + "_chunk_ids")
    if value is None:
        value = row.get(role + "_chunk_id", [])
    if not isinstance(value, list):
        value = [value]
    return {str(x).strip() for x in value if x is not None and str(x).strip()}


def normalize(value):
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def identity(row):
    tenant = str(row.get("tenant_id") or "").strip()
    query = str(row.get("query_id") or "").strip()
    if not tenant or not query:
        raise HoldoutAuditError("MISSING_TENANT_OR_QUERY_ID")
    return tenant, query


def query_inventory(rows, *, label):
    keys, texts = set(), set()
    for row in rows:
        key = identity(row)
        if key in keys:
            raise HoldoutAuditError("DUPLICATE_QUERY_ID:" + label)
        keys.add(key)
        question = normalize(row.get("query_text") or row.get("query"))
        if not question:
            raise HoldoutAuditError("EMPTY_QUESTION:" + label)
        texts.add(question)
    return keys, texts


def index_corpus(corpus, families):
    chunks, by_document = {}, {}
    for chunk in corpus:
        cid = str(chunk.get("chunk_id") or "").strip()
        tenant = str(chunk.get("tenant_id") or "").strip()
        doc = str(chunk.get("document_id") or "").strip()
        if not cid or not tenant or not doc or cid in chunks:
            raise HoldoutAuditError("INVALID_OR_DUPLICATE_CORPUS_IDENTITY")
        chunks[cid] = chunk
    for family in families:
        key = (str(family.get("tenant_id") or "").strip(),
               str(family.get("document_id") or "").strip())
        name = str(family.get("proposed_canonical_source_family") or "").strip()
        if not all(key) or not name or key in by_document:
            raise HoldoutAuditError("INVALID_OR_DUPLICATE_FAMILY")
        by_document[key] = (key[0], name)
    actual_documents = {
        (str(c["tenant_id"]), str(c["document_id"])) for c in chunks.values()
    }
    if actual_documents != set(by_document):
        raise HoldoutAuditError("CORPUS_SOURCE_FAMILY_MAP_INCOMPLETE")
    return chunks, by_document


def used_families(rows, chunks, family_map):
    families, missing, cross_tenant = set(), set(), set()
    for row in rows:
        tenant, qid = identity(row)
        for cid in refs(row, "positive") | refs(row, "negative"):
            chunk = chunks.get(cid)
            if chunk is None:
                missing.add((tenant, qid, cid))
                continue
            if str(chunk["tenant_id"]) != tenant:
                cross_tenant.add((tenant, qid, cid))
                continue
            families.add(family_map[(tenant, str(chunk["document_id"]))])
    return families, missing, cross_tenant


def audit(holdout, selected, original, dev, corpus, families):
    """Counts and flags only. Never output private question or source text."""
    if not all((holdout, selected, original, dev, corpus, families)):
        raise HoldoutAuditError("EMPTY_AUDIT_POPULATION")
    chunk_map, family_map = index_corpus(corpus, families)
    hold_keys, hold_texts = query_inventory(holdout, label="holdout")
    selected_keys, selected_texts = query_inventory(selected, label="selected_train")
    original_keys, original_texts = query_inventory(original, label="original_train")
    dev_keys, dev_texts = query_inventory(dev, label="dev")

    if not selected_keys.issubset(original_keys):
        raise HoldoutAuditError("SELECTED_QUERIES_NOT_IN_ORIGINAL_TRAIN")

    populations = {
        "holdout": holdout,
        "selected_train": selected,
        "original_train": original,
        "dev": dev,
    }
    family_sets = {}
    unresolved = {}
    for name, rows in populations.items():
        fams, missing, cross = used_families(rows, chunk_map, family_map)
        family_sets[name] = fams
        unresolved[name] = {
            "missing_reference_count": len(missing),
            "cross_tenant_reference_count": len(cross),
        }

    def union_refs(rows, role):
        result = set()
        for row in rows:
            result.update(refs(row, role))
        return result

    hp, hn = union_refs(holdout, "positive"), union_refs(holdout, "negative")
    sp, sn = union_refs(selected, "positive"), union_refs(selected, "negative")
    tr, dv = sp | sn, union_refs(dev, "positive") | union_refs(dev, "negative")
    fulltrain = union_refs(original, "positive") | union_refs(original, "negative")
    shared_refs = tr & (hp | hn)
    affected_rows = sum(bool((refs(row, "positive") | refs(row, "negative")) & tr)
                        for row in holdout)
    # Distinguish no gold label at all (possibly an intentional abstention
    # case) from a broken gold reference. Neither is silently removed or
    # assigned an answer by this structural audit.
    empty_gold_cases = [row for row in holdout if not refs(row, "positive")]
    empty_gold_rows = len(empty_gold_cases)
    empty_gold_category_counts = Counter(
        str(row.get("category") or "").strip().casefold() or "UNSPECIFIED"
        for row in empty_gold_cases
    )
    candidate_abstention_rows = sum(
        count for category, count in empty_gold_category_counts.items()
        if category in POTENTIAL_ABSTENTION_OR_SECURITY_CATEGORIES
    )
    category_unrecognized_rows = empty_gold_rows - candidate_abstention_rows
    unresolved_gold_rows = sum(
        bool(refs(row, "positive") - set(chunk_map))
        for row in holdout
    )
    missing_gold_rows = empty_gold_rows + unresolved_gold_rows
    resolvable_gold_rows = len(holdout) - missing_gold_rows
    cross_tenant_gold_rows = sum(
        any(cid in chunk_map and
            str(chunk_map[cid]["tenant_id"]) != str(row["tenant_id"])
            for cid in refs(row, "positive")) for row in holdout
    )
    known_families = set(family_map.values())
    history_families = family_sets["original_train"] | family_sets["dev"]
    source_overlap = family_sets["holdout"] & family_sets["selected_train"]
    historical_overlap = family_sets["holdout"] & history_families
    structurally_isolated = not (
        source_overlap or historical_overlap or shared_refs or
        hold_keys & (selected_keys | original_keys | dev_keys) or
        hold_texts & (selected_texts | original_texts | dev_texts) or
        any(x["missing_reference_count"] or x["cross_tenant_reference_count"]
            for x in unresolved.values()) or
        missing_gold_rows or cross_tenant_gold_rows
    )
    return {
        "status": ("BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE"
                   if not structurally_isolated
                   else "STRUCTURAL_ISOLATION_ONLY_CHRONOLOGY_UNVERIFIED"),
        "evaluation_role": "HISTORICAL_DIAGNOSTIC_ONLY",
        "independent_holdout_verified": False,
        "valid_for_independent_accuracy_claims": False,
        "training_authorized": False,
        "production_promotion_allowed": False,
        "question_counts": {
            "holdout": len(holdout), "selected_train": len(selected),
            "original_train": len(original), "dev": len(dev),
        },
        "question_overlap": {
            "selected_train_ids": len(hold_keys & selected_keys),
            "original_train_ids": len(hold_keys & original_keys),
            "dev_ids": len(hold_keys & dev_keys),
            "selected_train_normalized_text": len(hold_texts & selected_texts),
            "original_train_normalized_text": len(hold_texts & original_texts),
            "dev_normalized_text": len(hold_texts & dev_texts),
        },
        "reference_overlap": {
            "holdout_positive_vs_train_positive": len(hp & sp),
            "holdout_positive_vs_train_negative": len(hp & sn),
            "holdout_negative_vs_train_positive": len(hn & sp),
            "holdout_negative_vs_train_negative": len(hn & sn),
            "distinct_shared_chunk_ids_with_selected_train": len(shared_refs),
            "holdout_questions_affected_by_selected_train": affected_rows,
            "shared_with_original_train_chunk_ids": len((hp | hn) & fulltrain),
            "shared_with_dev_chunk_ids": len((hp | hn) & dv),
        },
        "source_families": {
            "corpus_families": len(known_families),
            "original_train_families": len(family_sets["original_train"]),
            "selected_train_families": len(family_sets["selected_train"]),
            "dev_families": len(family_sets["dev"]),
            "holdout_families": len(family_sets["holdout"]),
            "holdout_shared_with_selected_train": len(source_overlap),
            "holdout_shared_with_historical_train_or_dev": len(historical_overlap),
            "unused_families_after_original_train_and_dev":
                len(known_families - history_families),
        },
        "evidence_integrity": {
            "holdout_questions_with_missing_or_empty_gold": missing_gold_rows,
            "holdout_questions_with_no_positive_references": empty_gold_rows,
            "holdout_questions_with_unresolved_positive_references": unresolved_gold_rows,
            "holdout_questions_with_resolvable_positive_references": resolvable_gold_rows,
            "empty_gold_category_counts": dict(sorted(empty_gold_category_counts.items())),
            "empty_gold_in_potential_abstention_or_security_categories":
                candidate_abstention_rows,
            "empty_gold_with_unrecognized_or_missing_category": category_unrecognized_rows,
            "empty_gold_semantic_interpretation": "UNDETERMINED_POSSIBLE_ABSTENTION",
            "negative_only_expected_behavior_verified": False,
            "negative_only_test_results_scored": False,
            "holdout_questions_with_cross_tenant_gold": cross_tenant_gold_rows,
            "reference_errors_by_split": unresolved,
        },
        "review_metadata_unverified": {
            "status_counts": dict(Counter(
                str(row.get("review_status") or "") for row in holdout
            )),
            "rows_with_nonempty_reviewer_field": sum(
                bool(str(row.get("reviewer") or "").strip()) for row in holdout
            ),
            "reviewer_identity_authenticated": False,
        },
        "next_requirement":
            "New authorized source-family-disjoint evidence and independently verified review provenance",
    }


def audit_files(*, holdout, selected_train, original_train, dev, corpus, families):
    result = audit(
        read_jsonl(holdout), read_jsonl(selected_train),
        read_jsonl(original_train), read_jsonl(dev),
        read_jsonl(corpus), read_jsonl(families),
    )
    result["input_sha256"] = {
        "holdout": sha256(holdout),
        "selected_train": sha256(selected_train),
        "original_train": sha256(original_train),
        "dev": sha256(dev),
        "corpus": sha256(corpus),
        "families": sha256(families),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("holdout", "selected-train", "original-train",
                   "dev", "corpus", "families"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument(
        "--out", type=Path, help="Optional local metadata JSON; refuses overwrite"
    )
    args = parser.parse_args()
    try:
        report = audit_files(**{
            key: getattr(args, key) for key in
            ("holdout", "selected_train", "original_train", "dev", "corpus", "families")
        })
        output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.out is not None:
            target = args.out.expanduser().resolve()
            if not target.parent.is_dir():
                raise HoldoutAuditError("OUTPUT_PARENT_MISSING")
            with target.open("x", encoding="utf-8") as stream:
                stream.write(output)
        print(output, end="")
    except (HoldoutAuditError, OSError, ValueError) as exc:
        parser.exit(2, "BLOCKED:" + str(exc) + "\n")


if __name__ == "__main__":
    main()
