"""Read authoritative customer-facing handoff state from existing ServiceCall records."""

from django.db.models import F, Q
from types import SimpleNamespace

from core.models import CallUpdate, ServiceCall, StaffProfile

_UNSET = object()


def authorized_service_calls(ctx):
    """Service calls visible to the authenticated staff scope in ``ctx``."""
    tenant = ctx["tenant"]
    qs = ServiceCall.objects.filter(tenant=tenant).select_related(
        "tenant", "customer", "site", "site__zone", "asset", "asset__product", "zone", "zone__branch", "technician"
    ).filter(customer__tenant_id=tenant.id).filter(
        Q(site__isnull=True) | Q(site__tenant_id=tenant.id, site__customer_id=F("customer_id"))
    ).filter(
        Q(asset__isnull=True) | (
            Q(asset__tenant_id=tenant.id)
            & (Q(asset__customer__isnull=True) | Q(asset__customer_id=F("customer_id")))
        )
    ).filter(
        Q(technician__isnull=True) | Q(technician__tenant_id=tenant.id, technician__role="technician")
    ).filter(
        Q(zone__isnull=True) | Q(zone__tenant_id=tenant.id)
    )

    if ctx["role"] != "technician":
        return qs
    profile = ctx.get("staff_profile")
    if (not profile or not profile.is_active or profile.role != "technician"
            or profile.tenant_id != tenant.id or profile.user_id != ctx["user"].id):
        return qs.none()

    assigned = Q(technician=profile)
    incoming = Q(status="open", technician__isnull=True)
    if profile.zone_id:
        incoming &= Q(zone_id=profile.zone_id)
    elif profile.branch_id:
        incoming &= Q(zone__branch_id=profile.branch_id)
    return qs.filter(assigned | incoming)


def acknowledgement_for(call):
    """Return the current assignee's persisted acknowledgement event, if any."""
    if not call.technician_id:
        return None
    return (
        CallUpdate.objects.filter(
            tenant_id=call.tenant_id,
            service_call_id=call.id,
            author_id=call.technician_id,
            status="acknowledged",
        )
        .order_by("created_at", "id")
        .first()
    )


def acknowledgements_for_calls(calls):
    """Fetch current-assignee acknowledgement events in one query."""
    call_list = list(calls)
    assignees = {call.id: call.technician_id for call in call_list if call.technician_id}
    if not assignees:
        return {}
    events = CallUpdate.objects.filter(
        tenant_id=call_list[0].tenant_id,
        service_call_id__in=assignees,
        status="acknowledged",
    ).order_by("created_at", "id").values("id", "service_call_id", "author_id", "created_at")
    result = {}
    for event in events:
        call_id = event["service_call_id"]
        if event["author_id"] == assignees.get(call_id) and call_id not in result:
            result[call_id] = SimpleNamespace(**event)
    return result


def has_authorized_queue_recipient(call):
    """Whether an active, login-capable technician can receive this call."""
    qs = StaffProfile.objects.filter(
        tenant_id=call.tenant_id,
        role="technician",
        is_active=True,
        user__is_active=True,
        user__tenant_memberships__tenant_id=call.tenant_id,
        user__tenant_memberships__role="technician",
        user__tenant_memberships__is_active=True,
        user__tenant_memberships__tenant__is_active=True,
    )
    if call.technician_id:
        return qs.filter(pk=call.technician_id).exists()

    if call.zone_id:
        branch_id = call.zone.branch_id if call.zone_id and call.zone else None
        scope = Q(zone_id=call.zone_id)
        if branch_id:
            scope |= Q(zone__isnull=True, branch_id=branch_id)
        qs = qs.filter(scope)
    elif call.site_id and call.site.zone_id:
        branch_id = call.site.zone.branch_id if call.site.zone.branch_id else None
        scope = Q(zone_id=call.site.zone_id)
        if branch_id:
            scope |= Q(zone__isnull=True, branch_id=branch_id)
        qs = qs.filter(scope)

    return qs.distinct().exists()


def calls_with_authorized_queue_recipient(calls):
    """Resolve queue visibility for a set of unassigned calls without per-ticket queries."""
    call_list = [call for call in calls if call.status == "open" and not call.technician_id]
    if not call_list:
        return set()
    tenant_id = call_list[0].tenant_id
    technicians = list(StaffProfile.objects.filter(
        tenant_id=tenant_id,
        role="technician",
        is_active=True,
        user__is_active=True,
        user__tenant_memberships__tenant_id=tenant_id,
        user__tenant_memberships__role="technician",
        user__tenant_memberships__is_active=True,
        user__tenant_memberships__tenant__is_active=True,
    ).select_related("zone", "branch").distinct())
    visible = set()
    for call in call_list:
        zone_id = call.zone_id or (call.site.zone_id if call.site_id and call.site else None)
        zone = call.zone if call.zone_id else (call.site.zone if call.site_id and call.site else None)
        branch_id = zone.branch_id if zone else None
        for technician in technicians:
            if zone_id:
                in_scope = technician.zone_id == zone_id or (
                    technician.zone_id is None and branch_id and technician.branch_id == branch_id
                )
            elif call.site_id and call.site and call.site.zone_id:
                in_scope = False
            else:
                in_scope = True
            if in_scope:
                visible.add(call.id)
                break
    return visible


def handoff_state(call, acknowledgement=_UNSET, queue_recipient=_UNSET):
    """Map existing ServiceCall state and acknowledgement events to UI language."""
    service_status = (call.status or "").lower()
    if service_status in {"closed", "resolved"}:
        return "resolved"
    if service_status == "marked_for_closure":
        return "closure_review"
    if service_status == "in_progress":
        return "in_progress"
    if acknowledgement is _UNSET:
        acknowledgement = acknowledgement_for(call)
    if acknowledgement:
        return "acknowledged"
    if call.technician_id:
        return "assigned"
    if queue_recipient is _UNSET:
        queue_recipient = has_authorized_queue_recipient(call)
    if queue_recipient:
        return "queued"
    return "awaiting_assignment"


def customer_ticket_state(call, acknowledgement=_UNSET, queue_recipient=_UNSET):
    """Serialize the state that customers may safely use to track a handoff."""
    if acknowledgement is _UNSET:
        acknowledgement = acknowledgement_for(call)
    return {
        "handoff_state": handoff_state(call, acknowledgement, queue_recipient),
        "technician_acknowledged": acknowledgement is not None,
        "acknowledged_at": acknowledgement.created_at if acknowledgement else None,
    }


def customer_ticket_states(calls):
    """Bulk form of ``customer_ticket_state`` for ticket lists and polling."""
    call_list = list(calls)
    acknowledgements = acknowledgements_for_calls(call_list)
    queued_ids = calls_with_authorized_queue_recipient(call_list)
    return {
        call.id: customer_ticket_state(
            call,
            acknowledgements.get(call.id),
            call.id in queued_ids,
        )
        for call in call_list
    }
