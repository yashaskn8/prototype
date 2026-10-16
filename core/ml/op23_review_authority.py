"""Verify externally signed technical decisions; this module never signs reviews.

The trust-root JSON must be injected by the authorized deployment/identity owner,
outside candidate and private evaluation files. No real key is configured here.
"""
from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core.ml.op23_source_admission import canonical_bytes, sha256


class ApprovalError(ValueError):
    pass


def signed_content(case: dict) -> bytes:
    """Bind every case field, including source version and exact passage IDs."""
    return canonical_bytes({key: value for key, value in case.items() if key != "approval"})


def verify_signed_review(case: dict, trusted_keys: dict, *, expected_org: str, tenant_id: int) -> dict:
    if case.get("review_status") != "APPROVED":
        raise ApprovalError("Only approved decisions may enter ground truth")
    approval = case.get("approval") or {}
    key = trusted_keys.get(approval.get("key_id"))
    if not key or not key.get("active") or key.get("role") != "technical_reviewer":
        raise ApprovalError("No active trusted technical-review key")
    if str(key.get("organisation")) != str(expected_org) or key.get("tenant_id") != tenant_id:
        raise ApprovalError("Review key is not authorized for this organisation and tenant")
    if str(case.get("source_organisation")) != str(expected_org) or case.get("tenant_id") != tenant_id:
        raise ApprovalError("Case organisation or tenant mismatch")
    if case.get("reviewer_identity") != key.get("subject"):
        raise ApprovalError("Reviewer identity does not match trusted key subject")
    try:
        reviewed_at = datetime.fromisoformat(case["reviewed_at"].replace("Z", "+00:00"))
        if reviewed_at.tzinfo is None or reviewed_at > datetime.now(timezone.utc):
            raise ValueError("Invalid review time")
        if key.get("not_before") and reviewed_at < datetime.fromisoformat(key["not_before"].replace("Z", "+00:00")):
            raise ValueError("Review predates key authorization")
        if key.get("not_after") and reviewed_at > datetime.fromisoformat(key["not_after"].replace("Z", "+00:00")):
            raise ValueError("Review postdates key authorization")
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(key["public_key_b64"], validate=True))
        signature = base64.b64decode(approval["signature_b64"], validate=True)
        public_key.verify(signature, signed_content(case))
    except (KeyError, ValueError, TypeError, binascii.Error, InvalidSignature) as exc:
        raise ApprovalError("Review signature, timestamp or trusted key is invalid") from exc
    return {"reviewer_subject": key["subject"], "review_key_id": approval["key_id"],
            "signed_content_sha256": sha256(signed_content(case))}


def verify_configuration_lock(lock: dict, trusted_keys: dict, *, expected_org: str,
                              tenant_id: int, configuration_sha256: str) -> None:
    """Require an independent evaluation owner to seal the final configuration."""
    approval = lock.get("approval") or {}
    key = trusted_keys.get(approval.get("key_id"))
    if (lock.get("status") != "FROZEN" or lock.get("configuration_sha256") != configuration_sha256
            or str(lock.get("source_organisation")) != str(expected_org) or lock.get("tenant_id") != tenant_id
            or not key or not key.get("active") or key.get("role") != "evaluation_owner"
            or str(key.get("organisation")) != str(expected_org) or key.get("tenant_id") != tenant_id
            or lock.get("owner_identity") != key.get("subject")):
        raise ApprovalError("Final configuration lacks authorized evaluation-owner lock")
    try:
        frozen_at = datetime.fromisoformat(lock["frozen_at"].replace("Z", "+00:00"))
        if frozen_at.tzinfo is None or frozen_at > datetime.now(timezone.utc):
            raise ValueError("Invalid configuration freeze time")
        if key.get("not_before") and frozen_at < datetime.fromisoformat(key["not_before"].replace("Z", "+00:00")):
            raise ValueError("Freeze predates owner authorization")
        if key.get("not_after") and frozen_at > datetime.fromisoformat(key["not_after"].replace("Z", "+00:00")):
            raise ValueError("Freeze postdates owner authorization")
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(key["public_key_b64"], validate=True))
        public_key.verify(base64.b64decode(approval["signature_b64"], validate=True), signed_content(lock))
    except (KeyError, ValueError, TypeError, binascii.Error, InvalidSignature) as exc:
        raise ApprovalError("Final configuration lock signature is invalid") from exc
