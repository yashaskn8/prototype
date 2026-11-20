"""Customer feedback on one assistant answer (rating, optional reason and comment)."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError

from core.models import AnswerFeedback

RATINGS = {value for value, _ in AnswerFeedback.RATINGS}
REASONS = {value for value, _ in AnswerFeedback.REASONS}


def _snapshot(contract: dict) -> dict:
    """How this answer was produced (no customer text)."""
    outcome = contract.get("outcome") or {}
    citations = (contract.get("presentation") or {}).get("citations") or []
    from core.ml.reranker import active_runtime
    return {
        "response_status": str(outcome.get("status") or "")[:40],
        "response_mode": str(outcome.get("response_mode") or contract.get("response_mode") or "")[:40],
        "model_versions": {
            "embedding_profile": str(getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "bge-small")),
            "reranker_model": outcome.get("reranker_model"),
            "reranker_runtime": active_runtime(),
            "generation_model": str(getattr(settings, "SERVY_CUSTOMER_GENERATION_MODEL", "")) or None,
            "generation_applied": bool(outcome.get("generation_applied")),
        },
        "cited_document_versions": sorted({(c.get("document_id"), c.get("document_version"))
                                          for c in citations if c.get("document_id")}, key=str),
    }


def save_feedback(*, tenant, customer, conversation, message, data: dict) -> AnswerFeedback:
    """Create or update this customer's feedback for one of their assistant answers."""
    rating = str(data.get("rating") or "")
    reason = str(data.get("reason") or "")
    comment = " ".join(str(data.get("comment") or "").split())
    if rating not in RATINGS:
        raise ValidationError("rating must be 'helpful' or 'not_helpful'.")
    if reason not in REASONS:
        raise ValidationError("Unknown feedback reason.")
    if len(comment) > 500:
        raise ValidationError("comment must be at most 500 characters.")
    if rating == "helpful":
        reason = ""
    snap = _snapshot(message.structured_contract or {})
    feedback, _created = AnswerFeedback.objects.update_or_create(
        assistant_message=message,
        defaults={"tenant": tenant, "customer": customer, "conversation": conversation,
                  "asset": conversation.asset, "rating": rating, "reason": reason, "comment": comment,
                  "response_status": snap["response_status"], "response_mode": snap["response_mode"],
                  "model_versions": snap["model_versions"],
                  "cited_document_versions": [list(v) for v in snap["cited_document_versions"]],
                  # A changed rating needs a fresh review.
                  "review_status": "new", "export_approved": False},
    )
    return feedback


def as_dict(feedback: AnswerFeedback | None) -> dict | None:
    if feedback is None:
        return None
    return {"rating": feedback.rating, "reason": feedback.reason, "comment": feedback.comment,
            "updated_at": feedback.updated_at}


def regression_cases(queryset) -> list[dict]:
    """Reviewed, approved negative feedback as evaluation cases; only reviewer-written text."""
    cases = []
    for item in queryset.filter(rating="not_helpful", review_status="reviewed", export_approved=True) \
            .exclude(eval_question=""):
        cases.append({"asset_id": item.asset_id, "question": item.eval_question, "problem": item.reason or "unspecified",
                      "response_mode": item.response_mode, "model_versions": item.model_versions,
                      "cited_document_versions": item.cited_document_versions,
                      "source": f"feedback-{item.id}"})
    return cases
