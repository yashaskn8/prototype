"""
Dedicated Industrial Safety Approval Authority for Servy.

STRICT CONCURRENCY & SEPARATION OF POWERS:
Permission to review training examples does NOT grant authority to approve
industrial repair instructions. Training review authorization and industrial
safety procedure authorization are strictly segregated.
"""

import logging
from typing import Any, Optional, Tuple
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

PROHIBITED_REVIEWER_KEYWORDS = (
    "training", "label", "annotator", "dataset", "ml", "evaluator",
    "synthetic", "bot", "auto", "automated", "test", "dummy", "placeholder",
    "self-approved", "unverified", "unknown", "anonymous", "none",
)

HAZARDOUS_PREREQUISITES = {
    "live_circuit", "open_cabinet", "bypass_interlock", "high_voltage",
    "refrigerant_gauge", "three_phase", "arc_flash", "capacitive_discharge",
}

TECHNICIAN_HAZARDS = {
    "electrical", "three_phase", "refrigerant", "chemical", "pressure",
    "high_voltage", "combustion", "mechanical_pinch",
}


def is_authorized_safety_approver(reviewer_identity: str, tenant_id: Optional[int] = None) -> bool:
    """
    Verify whether a reviewer has explicit, positive industrial safety approval authority.
    Rejects:
    - Training reviewers attempting to authorize physical procedures
    - Synthetic, bot, or automated actors
    - Unvetted reviewer strings
    """
    if not reviewer_identity or not isinstance(reviewer_identity, str):
        return False

    ident = reviewer_identity.strip()
    if not ident:
        return False

    ident_lower = ident.lower()

    # Gate 0: Immediate rejection of training-only or synthetic roles
    for prohibited in PROHIBITED_REVIEWER_KEYWORDS:
        if prohibited in ident_lower:
            return False

    # Check authorized safety approvers list
    authorized_approvers = getattr(
        settings,
        "SERVY_AUTHORIZED_SAFETY_APPROVERS",
        [
            "Certified OEM Safety Engineer",
            "Director of Product Safety",
            "Senior Industrial Safety Auditor",
            "dr-yashas-kn",
            "admin",
        ],
    )
    for auth in authorized_approvers:
        if ident_lower == auth.strip().lower():
            return True

    # Check against database User model:
    # Requires explicit permission ("core.can_approve_industrial_safety") or superuser.
    # Generic staff accounts without explicit safety authority are strictly rejected.
    try:
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.filter(username=ident, is_active=True).first()
        if user and (user.is_superuser or user.has_perm("core.can_approve_industrial_safety")):
            # Verify user has not been flagged as training-only
            if not any(k in user.username.lower() for k in PROHIBITED_REVIEWER_KEYWORDS):
                return True
    except Exception:
        pass

    return False


def validate_procedure_safety_approval(
    app: Any,
    step_text: str,
    asset: Optional[Any] = None,
    tenant: Optional[Any] = None,
) -> Tuple[bool, str, str]:
    """
    Validate a database ProcedureApproval against all 11 industrial safety gates.
    Returns (is_approved, status_classification, explanation_reason).
    """
    now = timezone.now()
    lower_step = (step_text or "").lower()

    # Gate 1: Revocation
    if app.revoked_at is not None or getattr(app, "approval_status", "") == "REVOKED":
        return False, "PROHIBITED", "Procedure approval has been revoked."

    # Gate 2: Effective and expiration dates
    if app.effective_date and app.effective_date > now:
        return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure approval is not yet effective."
    if app.expiration_date and app.expiration_date < now:
        return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure approval has expired."

    # Gate 3: Applicable asset scope
    asset_code = getattr(asset, "asset_code", "") if asset else ""
    if isinstance(asset, dict):
        asset_code = asset.get("asset_code", "")
    if app.asset_scope != "*" and app.asset_scope != asset_code:
        return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure does not apply to this specific asset code."

    # Gate 4: Exact equipment model match
    asset_model = getattr(asset, "model_number", "") if asset else ""
    if isinstance(asset, dict):
        asset_model = asset.get("model_number", "")
    if app.equipment_model and app.equipment_model != "*":
        if not asset_model or app.equipment_model.strip().lower() != asset_model.strip().lower():
            return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure applies to a different equipment model."

    # Gate 5: Equipment manufacturer match
    asset_mfr = getattr(asset, "manufacturer", "") if asset else ""
    if not asset_mfr and hasattr(asset, "product") and asset.product:
        asset_mfr = getattr(asset.product, "manufacturer", "")
    if isinstance(asset, dict):
        asset_mfr = asset.get("manufacturer", "")
    if app.equipment_manufacturer and app.equipment_manufacturer != "*":
        if not asset_mfr or app.equipment_manufacturer.strip().lower() != str(asset_mfr).strip().lower():
            return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure applies to a different equipment manufacturer."

    # Gate 6: Source document identity and integrity
    if not app.document or not app.document.is_rag_enabled or app.document.is_confidential:
        return False, "PROHIBITED", "Source technical document is invalid, confidential, or unapproved."
    if app.document_version and getattr(app.document, "index_version", None):
        if str(app.document_version) != str(app.document.index_version):
            return False, "UNKNOWN_REQUIRES_REVIEW", "Procedure document version does not match active index version."

    # Gate 7: Dedicated safety approval authority & genuine decision provenance
    tenant_id = getattr(tenant, "id", None) if tenant else None
    if not app.reviewer_identity or not is_authorized_safety_approver(app.reviewer_identity, tenant_id):
        return False, "UNKNOWN_REQUIRES_REVIEW", f"Reviewer '{app.reviewer_identity}' lacks authorized safety approval clearance."
    evidence = (app.approval_evidence or "").strip().lower()
    if len(evidence) < 10 or any(p in evidence for p in ("placeholder", "test", "dummy", "self-approved", "n/a", "training")):
        return False, "UNKNOWN_REQUIRES_REVIEW", "Missing verified safety audit decision provenance."

    # Gate 8: Safety Prerequisites Check
    prereqs = app.prerequisites or []
    if any(isinstance(p, str) and p.lower() in HAZARDOUS_PREREQUISITES for p in prereqs):
        return False, "TECHNICIAN_ONLY", f"Procedure requires certified technician safety prerequisites ({', '.join(prereqs)})."

    # Gate 9: Keywords match with step text
    keywords = app.procedure_keywords or []
    if not any(kw in lower_step for kw in keywords):
        return False, "UNKNOWN_REQUIRES_REVIEW", "Step does not match approved procedure keywords."

    # Gate 10: Secondary prohibited/technician keyword check in step text
    from core.services.customer_diagnostic import PROHIBITED_KEYWORDS, TECHNICIAN_ONLY_KEYWORDS
    if any(kw in lower_step for kw in PROHIBITED_KEYWORDS):
        return False, "PROHIBITED", "Procedure contains prohibited safety violations or bypass steps."
    if any(kw in lower_step for kw in TECHNICIAN_ONLY_KEYWORDS):
        return False, "TECHNICIAN_ONLY", "Procedure contains hazardous secondary work requiring a certified technician."

    # Gate 11: Procedure approval hazards check
    app_hazards = app.hazards or []
    if any(h in TECHNICIAN_HAZARDS for h in app_hazards):
        return False, "TECHNICIAN_ONLY", f"Procedure involves certified technician-only hazards ({', '.join(app_hazards)})."

    return True, "CUSTOMER_APPROVED", f"Authorized customer procedure ({app.procedure_id}) with verified manufacturer safety approval."
