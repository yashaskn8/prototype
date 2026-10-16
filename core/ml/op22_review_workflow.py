"""Strict review and source-separation gates for future technical evaluations.

The historical Operation 20/21 benchmark is diagnostic material and never a
source of automatically approved labels. This module makes no approval itself.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class ReviewValidationError(ValueError):
    pass


def validate_approved_case(row: dict, authorized_reviewers: set[str]) -> dict:
    """Validate a human-reviewed judgment against an external reviewer allowlist."""
    if row.get("review_status") != "APPROVED":
        raise ReviewValidationError("PENDING_REVIEW cases cannot become ground truth")
    reviewer = row.get("reviewer_identity")
    if not reviewer or reviewer not in authorized_reviewers:
        raise ReviewValidationError("Reviewer identity is not in the authorized allowlist")
    if not row.get("reviewed_at") or not row.get("approval_evidence"):
        raise ReviewValidationError("Approval timestamp and evidence are required")
    if row.get("source_organisation") is None or row.get("tenant_id") is None:
        raise ReviewValidationError("Tenant and source organisation are required")
    source_hash = row.get("source_sha256") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ReviewValidationError("A full source SHA-256 is required")
    if row.get("source_document_id") is None or row.get("source_attachment_id") is None:
        raise ReviewValidationError("Verified source document and attachment IDs are required")
    if row.get("source_verification") != "VERIFIED_ORIGINAL_COMPANY_SOURCE":
        raise ReviewValidationError("Source has not been independently reconciled")
    if not row.get("query_id") or not row.get("question_text"):
        raise ReviewValidationError("Query identity and text are required")
    status = row.get("answerability")
    if status not in {"ANSWERABLE", "UNANSWERABLE_FROM_APPROVED_SOURCES", "NEEDS_ASSET_CLARIFICATION"}:
        raise ReviewValidationError("An explicit answerability judgment is required")
    positives = row.get("valid_positive_chunk_ids") or []
    if status == "ANSWERABLE" and not positives:
        raise ReviewValidationError("Answerable cases require at least one approved positive")
    if status != "ANSWERABLE" and positives:
        raise ReviewValidationError("Unanswerable cases cannot retain positive labels")
    if len(set(positives)) != len(positives):
        raise ReviewValidationError("Duplicate positive IDs")
    alternatives = row.get("valid_alternative_chunk_ids") or []
    if not set(alternatives).issubset(set(positives)):
        raise ReviewValidationError("Every approved alternative passage must be included in valid positives")
    if set(row.get("reviewed_negative_chunk_ids") or []) & set(positives):
        raise ReviewValidationError("A valid passage cannot also be a reviewed training negative")
    return row


def require_source_separation(rows: list[dict], excluded_source_hashes: set[str]) -> None:
    """Fail if final-test sources overlap any training/development/history source."""
    if not rows:
        raise ReviewValidationError("Cannot freeze an empty evaluation set")
    ids = set()
    for row in rows:
        if row["source_sha256"] in excluded_source_hashes:
            raise ReviewValidationError("Final-test source overlaps a prior development or historical source")
        if row["query_id"] in ids:
            raise ReviewValidationError("Duplicate query ID")
        ids.add(row["query_id"])


def freeze_reviewed_dataset(
    rows: list[dict], destination: Path, *, authorized_reviewers: set[str],
    excluded_source_hashes: set[str],
) -> dict:
    """Write a versioned immutable-by-convention JSONL once; refuse overwrite."""
    for row in rows:
        validate_approved_case(row, authorized_reviewers)
    require_source_separation(rows, excluded_source_hashes)
    destination = Path(destination)
    if destination.exists():
        raise ReviewValidationError("Refusing to overwrite a frozen evaluation set")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    return {"dataset_path": str(destination), "case_count": len(rows),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
