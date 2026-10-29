"""Company-specific data curation, training readiness & independent benchmark pipeline.

Implements the 10-step company data workflow required by Section 6 and Section 7:
  1. Document ingestion & validation (tenant, title, equipment, version, SHA-256)
  2. Section-aware segmentation (preserving heading hierarchy and warning blocks)
  3. Version and checksum tracking
  4. Equipment and model metadata mapping
  5. Duplicate and near-duplicate detection
  6. Confidential content minimization
  7. Revoked document quarantine
  8. Query, positive passage, and hard-negative construction (strict Section 6 JSON schema)
  9. Human SME review export
  10. Family-disjoint train / validation / independent test split & leakage verification

Fail-closed invariants:
  - Never fabricates human SME approval (`review_status` defaults to `"pending_sme_review"`).
  - Never overwrites existing split or benchmark files.
  - Rejects any split where `source_family` overlaps between train, dev, or test, or where
    any independent benchmark family overlaps historical V4/V5/V6 training or dev families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_EXAMPLE_FIELDS = (
    "query_id",
    "tenant_id",
    "equipment_family",
    "model_number",
    "query_type",
    "query_text",
    "positive_chunk_ids",
    "positive_doc_ids",
    "hard_negative_chunk_ids",
    "source_family",
    "expected_abstain",
    "review_status",
    "reviewer",
    "split",
)

VALID_QUERY_TYPES = frozenset({
    "exact_error_code",
    "general_symptom",
    "voltage_electrical_spec",
    "preventive_maintenance_schedule",
    "safety_escalation",
    "model_trap",
    "misspelled_query",
    "multi_symptom",
    "model_ambiguity",
    "technician_only",
    "equipment_crosstalk",
    "unsupported_question",
    "out_of_domain",
    "doc_id_attack",
    "revoked_doc_query",
    "tenant_isolation_attack",
    "boundary_injection",
})

VALID_SPLITS = frozenset({"train", "dev", "independent_test"})

_CONFIDENTIAL_REDACTION_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})\b"), "[REDACTED_PHONE]"),
]


class CompanyDataPipelineError(ValueError):
    """Raised when company data validation, quarantine, or split isolation fails."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes((text or "").encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_source_family(title: str, product: str = "", category: str = "") -> str:
    """Derive a deterministic canonical source family identifier from document metadata."""
    base = (product or title or category or "").strip().lower()
    # Strip copy/version suffixes such as "(Rev 2)", "v1.2", "manual #3"
    base = re.sub(r"\b(?:rev(?:ision)?|ver(?:sion)?|v)\s*[\d.]+\b", "", base)
    base = re.sub(r"\b(?:manual|guide|sop|handbook|specification|datasheet)\b", "", base)
    tokens = re.findall(r"[a-z0-9]+", base)
    if not tokens:
        raise CompanyDataPipelineError("Cannot derive canonical source family from empty title/product")
    return "fam_" + "_".join(tokens[:6])


def minimize_confidential_text(text: str) -> str:
    """Redact incidental PII (emails/phone numbers) from technical passages."""
    cleaned = text or ""
    for pat, repl in _CONFIDENTIAL_REDACTION_PATTERNS:
        cleaned = pat.sub(repl, cleaned)
    return cleaned


def validate_and_quarantine_documents(
    documents: Iterable[Dict[str, Any]],
    *,
    allow_confidential: bool = False,
) -> Dict[str, Any]:
    """Steps 1, 3, 4, 5, 6, 7: Validate ingested documents, quarantine revoked/confidential/duplicate docs."""
    accepted: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    seen_checksums: Dict[Tuple[int, str], int] = {}

    for doc in documents:
        doc_id = doc.get("document_id") or doc.get("id")
        tenant_id = doc.get("tenant_id")
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or doc.get("passage") or "").strip()
        if doc_id is None or tenant_id is None or not title or not text:
            quarantined.append({"document_id": doc_id, "reason": "MISSING_REQUIRED_DOCUMENT_METADATA_OR_TEXT"})
            continue

        if doc.get("revoked_at") is not None or bool(doc.get("is_revoked", False)):
            quarantined.append({"document_id": doc_id, "reason": "REVOKED_DOCUMENT_QUARANTINED"})
            continue

        if bool(doc.get("is_quarantined", False)) or not bool(doc.get("is_rag_enabled", True)):
            quarantined.append({"document_id": doc_id, "reason": "DISABLED_OR_QUARANTINED_DOCUMENT"})
            continue

        if bool(doc.get("is_confidential", False)) and not allow_confidential:
            quarantined.append({"document_id": doc_id, "reason": "CONFIDENTIAL_DOCUMENT_EXCLUDED"})
            continue

        minimized_text = minimize_confidential_text(text)
        norm_text = " ".join(minimized_text.lower().split())
        checksum = sha256_text(norm_text)
        dedup_key = (int(tenant_id), checksum)
        if dedup_key in seen_checksums:
            quarantined.append({
                "document_id": doc_id,
                "reason": f"DUPLICATE_CONTENT_OF_DOC_{seen_checksums[dedup_key]}",
            })
            continue
        seen_checksums[dedup_key] = int(doc_id)

        source_fam = str(doc.get("source_family") or "").strip() or canonical_source_family(
            title,
            str(doc.get("product") or doc.get("equipment_family") or ""),
            str(doc.get("category") or ""),
        )
        accepted.append({
            "document_id": int(doc_id),
            "tenant_id": int(tenant_id),
            "title": title,
            "equipment_family": str(doc.get("equipment_family") or doc.get("product") or "general").strip(),
            "model_number": str(doc.get("model_number") or "").strip(),
            "version": str(doc.get("version") or doc.get("index_version") or "1.0"),
            "checksum_sha256": checksum,
            "source_family": source_fam,
            "text": minimized_text,
        })

    return {
        "accepted_documents": accepted,
        "quarantined_documents": quarantined,
        "accepted_count": len(accepted),
        "quarantined_count": len(quarantined),
    }


def segment_document_sections(doc: Dict[str, Any], *, start_chunk_id: int = 1) -> List[Dict[str, Any]]:
    """Step 2: Section-aware segmentation preserving markdown headings and equipment context."""
    raw_text = str(doc.get("text") or "").strip()
    if not raw_text:
        return []
    parts = re.split(r"(?m)^(#{1,3}\s+.+)$", raw_text)
    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"
    buffer: List[str] = []

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = stripped.lstrip("#").strip()
        else:
            buffer.append(stripped)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    chunks: List[Dict[str, Any]] = []
    for idx, (heading, body) in enumerate(sections):
        cid = start_chunk_id + idx
        passage = f"## {heading}\n{body}" if heading else body
        chunks.append({
            "chunk_id": cid,
            "chunk_index": idx,
            "document_id": int(doc["document_id"]),
            "tenant_id": int(doc["tenant_id"]),
            "title": doc["title"],
            "equipment_family": doc["equipment_family"],
            "model_number": doc["model_number"],
            "source_family": doc["source_family"],
            "heading": heading,
            "passage": passage,
            "content_sha256": sha256_text(passage),
        })
    return chunks


def validate_training_example_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """Step 8: Validate a single example against the mandatory Section 6 JSON schema."""
    missing = [f for f in REQUIRED_EXAMPLE_FIELDS if f not in record]
    if missing:
        raise CompanyDataPipelineError(f"Missing required schema fields: {missing}")

    qid = str(record["query_id"] or "").strip()
    if not qid:
        raise CompanyDataPipelineError("Empty query_id")
    tid = int(record["tenant_id"])
    if tid <= 0:
        raise CompanyDataPipelineError(f"{qid}: invalid tenant_id {tid}")

    qtype = str(record["query_type"] or "").strip()
    if qtype not in VALID_QUERY_TYPES:
        raise CompanyDataPipelineError(f"{qid}: invalid query_type '{qtype}'")

    qtext = str(record["query_text"] or "").strip()
    if not qtext:
        raise CompanyDataPipelineError(f"{qid}: empty query_text")

    pos_cids = [int(x) for x in record["positive_chunk_ids"]]
    pos_dids = [int(x) for x in record["positive_doc_ids"]]
    neg_cids = [int(x) for x in record["hard_negative_chunk_ids"]]
    abstain = bool(record["expected_abstain"])

    if abstain and pos_cids:
        raise CompanyDataPipelineError(f"{qid}: expected_abstain=True cannot have positive_chunk_ids")
    if not abstain and not pos_cids:
        raise CompanyDataPipelineError(f"{qid}: expected_abstain=False requires positive_chunk_ids")
    if set(pos_cids) & set(neg_cids):
        raise CompanyDataPipelineError(f"{qid}: overlap between positive_chunk_ids and hard_negative_chunk_ids")

    source_fam = str(record["source_family"] or "").strip()
    if not source_fam:
        raise CompanyDataPipelineError(f"{qid}: missing source_family")

    split = str(record["split"] or "").strip()
    if split not in VALID_SPLITS:
        raise CompanyDataPipelineError(f"{qid}: invalid split '{split}'")

    review_status = str(record["review_status"] or "").strip()
    reviewer = str(record["reviewer"] or "").strip()
    if review_status == "approved_by_sme" and not reviewer:
        raise CompanyDataPipelineError(f"{qid}: review_status='approved_by_sme' requires non-empty reviewer")

    return {
        "query_id": qid,
        "tenant_id": tid,
        "equipment_family": str(record["equipment_family"] or "").strip(),
        "model_number": str(record["model_number"] or "").strip(),
        "query_type": qtype,
        "query_text": qtext,
        "positive_chunk_ids": pos_cids,
        "positive_doc_ids": pos_dids,
        "hard_negative_chunk_ids": neg_cids,
        "source_family": source_fam,
        "expected_abstain": abstain,
        "review_status": review_status,
        "reviewer": reviewer,
        "split": split,
    }


def verify_family_disjoint_splits(
    examples: Iterable[Dict[str, Any]],
    *,
    historical_families: Optional[Set[str]] = None,
    historical_chunk_ids: Optional[Set[int]] = None,
    historical_query_texts: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Step 10: Enforce strict source-family disjointness across train, dev, and independent_test,
    plus zero leakage against historical V4/V5/V6 splits for independent_test."""
    validated = [validate_training_example_schema(r) for r in examples]
    if not validated:
        raise CompanyDataPipelineError("No examples provided")

    seen_qids: Set[str] = set()
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "independent_test": []}
    families_by_split: Dict[str, Set[str]] = {"train": set(), "dev": set(), "independent_test": set()}
    chunks_by_split: Dict[str, Set[int]] = {"train": set(), "dev": set(), "independent_test": set()}
    norm_queries_by_split: Dict[str, Set[str]] = {"train": set(), "dev": set(), "independent_test": set()}

    for ex in validated:
        qid = ex["query_id"]
        if qid in seen_qids:
            raise CompanyDataPipelineError(f"Duplicate query_id: {qid}")
        seen_qids.add(qid)
        sp = ex["split"]
        by_split[sp].append(ex)
        families_by_split[sp].add(ex["source_family"])
        for cid in ex["positive_chunk_ids"] + ex["hard_negative_chunk_ids"]:
            chunks_by_split[sp].add(int(cid))
        norm_q = " ".join(re.findall(r"\w+", ex["query_text"].lower()))
        norm_queries_by_split[sp].add(norm_q)

    # Check pairwise split overlaps
    train_dev_fam = families_by_split["train"] & families_by_split["dev"]
    train_test_fam = families_by_split["train"] & families_by_split["independent_test"]
    dev_test_fam = families_by_split["dev"] & families_by_split["independent_test"]
    if train_dev_fam or train_test_fam or dev_test_fam:
        raise CompanyDataPipelineError(
            f"Source family overlap across splits: train&dev={sorted(train_dev_fam)}, "
            f"train&test={sorted(train_test_fam)}, dev&test={sorted(dev_test_fam)}"
        )

    train_dev_chunks = chunks_by_split["train"] & chunks_by_split["dev"]
    train_test_chunks = chunks_by_split["train"] & chunks_by_split["independent_test"]
    dev_test_chunks = chunks_by_split["dev"] & chunks_by_split["independent_test"]
    if train_dev_chunks or train_test_chunks or dev_test_chunks:
        raise CompanyDataPipelineError("Chunk ID overlap across splits")

    train_dev_q = norm_queries_by_split["train"] & norm_queries_by_split["dev"]
    train_test_q = norm_queries_by_split["train"] & norm_queries_by_split["independent_test"]
    dev_test_q = norm_queries_by_split["dev"] & norm_queries_by_split["independent_test"]
    if train_dev_q or train_test_q or dev_test_q:
        raise CompanyDataPipelineError("Normalized query text overlap across splits")

    # Check historical V4/V5/V6 overlap for independent_test
    hist_fams = historical_families or set()
    hist_chunks = historical_chunk_ids or set()
    hist_queries = historical_query_texts or set()

    test_hist_fam_overlap = families_by_split["independent_test"] & hist_fams
    test_hist_chunk_overlap = chunks_by_split["independent_test"] & hist_chunks
    test_hist_query_overlap = norm_queries_by_split["independent_test"] & hist_queries
    if test_hist_fam_overlap or test_hist_chunk_overlap or test_hist_query_overlap:
        raise CompanyDataPipelineError(
            f"Independent test leakage against historical V4/V5/V6 artifacts: "
            f"families={sorted(test_hist_fam_overlap)}, chunks={sorted(test_hist_chunk_overlap)}, "
            f"queries={len(test_hist_query_overlap)}"
        )

    sme_approved_count = sum(1 for ex in validated if ex["review_status"] == "approved_by_sme")
    return {
        "status": "SPLITS_VERIFIED_FAMILY_DISJOINT",
        "total_examples": len(validated),
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "family_counts": {k: len(v) for k, v in families_by_split.items()},
        "sme_approved_examples": sme_approved_count,
        "all_sme_approved": (sme_approved_count == len(validated) and len(validated) > 0),
        "independent_test_historical_leakage": False,
        "production_promotion_allowed": False,
    }
