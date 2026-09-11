from datetime import timedelta

from django.db import IntegrityError, transaction
from django.db.models import Count, Q, Max
from django.utils import timezone

from core.models import ServiceCall, StaffProfile, SLAPolicy


def choose_technician(tenant, branch=None, zone=None):
    qs = StaffProfile.objects.filter(tenant=tenant, role="technician", is_active=True)
    if zone:
        zone_qs = qs.filter(zone=zone)
        if zone_qs.exists():
            qs = zone_qs
    if branch:
        branch_qs = qs.filter(branch=branch)
        if branch_qs.exists():
            qs = branch_qs
    return qs.annotate(
        active_calls=Count("service_calls", filter=Q(service_calls__status__in=["assigned", "in_progress"]))
    ).order_by("active_calls", "full_name").first()


def _sla_for(tenant, priority="normal"):
    return SLAPolicy.objects.filter(tenant=tenant, priority=priority, is_active=True).order_by("response_minutes").first()


def create_call_from_interaction(interaction):
    interaction.refresh_from_db()
    if interaction.escalated_call:
        return interaction.escalated_call

    tenant = interaction.tenant
    zone = interaction.site.zone if interaction.site else None
    branch = zone.branch if zone and zone.branch else None
    technician = choose_technician(tenant, branch=branch, zone=zone)
    priority = "normal"
    sla = _sla_for(tenant, priority)
    now = timezone.now()
    response_minutes = sla.response_minutes if sla else 120
    resolution_minutes = sla.resolution_minutes if sla else 720

    for _ in range(3):
        try:
            with transaction.atomic():
                from core.models import CustomerInteraction
                locked = CustomerInteraction.objects.filter(id=interaction.id).first()
                if not locked:
                    raise ValueError("Interaction record not found.")
                if locked.escalated_call:
                    return locked.escalated_call

                last_id = ServiceCall.objects.filter(tenant=tenant).aggregate(m=Max("servy_id"))["m"] or 42830
                ref_list = []

                if interaction.source_refs:
                    for r in interaction.source_refs:
                        ref_title = r.get("reference") or r.get("title")
                        if ref_title:
                            ref_list.append(ref_title)
                ref_text = "\n".join(f"- {ref}" for ref in ref_list) if ref_list else "- None cited"

                complaint_body = (
                    f"Customer issue:\n{interaction.question}\n\n"
                    f"Troubleshooting already attempted:\n{interaction.answer}\n\n"
                    f"Source documents used:\n{ref_text}\n\n"
                    f"Customer reported that the issue was not resolved (escalated from Customer AI Support at {now.strftime('%Y-%m-%d %H:%M:%S UTC')})."
                )

                call = ServiceCall.objects.create(
                    tenant=tenant,
                    servy_id=last_id + 1,
                    call_type="Service",
                    complaint_type="AI Self-Service Escalation",
                    complaint_text=complaint_body,
                    customer=interaction.customer,
                    site=interaction.site,
                    asset=interaction.asset,
                    zone=zone,
                    sla_policy=sla,
                    technician=technician,
                    status="assigned" if technician else "open",
                    priority=priority,
                    contact_name=interaction.customer.contact_name,
                    contact_phone=interaction.customer.phone,
                    contact_email=interaction.customer.email,
                    response_due_at=now + timedelta(minutes=response_minutes),
                    resolution_due_at=now + timedelta(minutes=resolution_minutes),
                )
                locked.resolved = False
                locked.escalated_call = call
                locked.save(update_fields=["resolved", "escalated_call"])
                interaction.refresh_from_db()
                return call
        except IntegrityError:
            # Handle SQLite concurrency contention: OneToOne constraint on escalated_call
            # or servy_id collision. Rollback occurs automatically in transaction.atomic().
            interaction.refresh_from_db()
            if interaction.escalated_call:
                return interaction.escalated_call
            continue

    interaction.refresh_from_db()
    if interaction.escalated_call:
        return interaction.escalated_call
    raise RuntimeError("Could not allocate a unique service call ID after multiple attempts.")

