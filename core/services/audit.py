from core.models import AuditEvent


def _client_ip(request):
    # Do not trust X-Forwarded-For by default in a local prototype.
    value = request.META.get("REMOTE_ADDR") or None
    return value if value and len(value) <= 45 else None


def audit_event(request, action, *, tenant=None, object_type="", object_id="", metadata=None):
    if tenant is None:
        membership = getattr(request, "servy_membership", None)
        tenant = getattr(membership, "tenant", None)
    if tenant is None:
        return None
    safe_metadata = metadata or {}
    # Keep audit records metadata-oriented; do not dump prompts, passwords, or
    # document bodies into logs.
    return AuditEvent.objects.create(
        tenant=tenant,
        user=request.user if getattr(request, "user", None) and request.user.is_authenticated else None,
        action=str(action)[:120],
        object_type=str(object_type)[:80],
        object_id=str(object_id)[:80],
        ip_address=_client_ip(request),
        metadata=safe_metadata,
    )
