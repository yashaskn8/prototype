import time
from datetime import timedelta

from django.db import IntegrityError, OperationalError, transaction
from django.db.models import Count, Q, Max
from django.utils import timezone

from core.models import ServiceCall, StaffProfile, SLAPolicy, Tenant


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

    # Prefer technicians with a linked user account so they can actually
    # log in to see and act on the assigned call.
    loginable_qs = qs.filter(user__isnull=False)
    pick_from = loginable_qs if loginable_qs.exists() else qs

    return pick_from.annotate(
        active_calls=Count("service_calls", filter=Q(service_calls__status__in=["assigned", "in_progress"]))
    ).order_by("active_calls", "full_name").first()


def _sla_for(tenant, priority="normal"):
    return SLAPolicy.objects.filter(tenant=tenant, priority=priority, is_active=True).order_by("response_minutes").first()


def create_service_call_record(
    tenant,
    complaint_type: str,
    complaint_text: str,
    customer,
    site=None,
    asset=None,
    zone=None,
    priority="normal",
    sla=None,
    technician=None,
    contact_name="",
    contact_phone="",
    contact_email="",
    call_type="Service",
    technician_notes="",
    max_attempts=5,
) -> ServiceCall:
    """Safely create a ServiceCall handling concurrent servy_id allocation with retry."""
    now = timezone.now()
    if not sla:
        sla = _sla_for(tenant, priority)
    response_minutes = sla.response_minutes if sla else 120
    resolution_minutes = sla.resolution_minutes if sla else 720

    c_name = contact_name or (customer.contact_name if customer else "") or (customer.name if customer else "")
    c_phone = contact_phone or (customer.phone if customer else "")
    c_email = contact_email or (customer.email if customer else "")

    for attempt in range(max_attempts):
        try:
            with transaction.atomic():
                # Acquire exclusive lock on the Tenant row under InnoDB (select_for_update)
                # to serialize ticket ID allocation per tenant, eliminating race conditions.
                Tenant.objects.select_for_update().filter(id=tenant.id).first()
                last_id = ServiceCall.objects.filter(tenant=tenant).aggregate(m=Max("servy_id"))["m"] or 42830
                call = ServiceCall.objects.create(
                    tenant=tenant,
                    servy_id=last_id + 1,
                    call_type=call_type,
                    complaint_type=complaint_type,
                    complaint_text=complaint_text,
                    technician_notes=technician_notes,
                    customer=customer,
                    site=site,
                    asset=asset,
                    zone=zone,
                    sla_policy=sla,
                    technician=technician,
                    status="assigned" if technician else "open",
                    priority=priority,
                    contact_name=c_name,
                    contact_phone=c_phone,
                    contact_email=c_email,
                    response_due_at=now + timedelta(minutes=response_minutes),
                    resolution_due_at=now + timedelta(minutes=resolution_minutes),
                )
                return call
        except (IntegrityError, OperationalError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))

    raise RuntimeError("Could not allocate a unique service call ID after multiple attempts.")


def create_call_from_interaction(interaction, customer_note: str = "", session_id: str = None):
    from core.models import CustomerInteraction, DiagnosticEvent, DiagnosticSession
    from core.services.customer_diagnostic import check_for_safety_hazard

    interaction.refresh_from_db()
    if interaction.escalated_call_id:
        return interaction.escalated_call

    tenant = interaction.tenant

    # Incident-level deduplication with strict multi-tenant and asset verification:
    # Incident identity is strictly server-authoritative via session UUID.
    # Natural-language text matching and heuristic substring checks are strictly forbidden.
    target_session = None
    if session_id:
        # Validate tenant, customer, and session ownership
        target_session = DiagnosticSession.objects.filter(
            tenant=tenant,
            customer=interaction.customer,
            session_id=session_id,
        ).first()
        if interaction.asset_id and target_session and target_session.asset_id != interaction.asset_id:
            target_session = None

    is_hazard, _ = check_for_safety_hazard(interaction.question or "")

    if target_session and target_session.escalated_call_id:
        existing_call = target_session.escalated_call

        # Ownership validation: Tenant, Customer, Asset, Site, and ServiceCall must match strictly
        if existing_call.tenant_id != tenant.id or existing_call.customer_id != interaction.customer_id:
            target_session = None
        elif interaction.asset_id and existing_call.asset_id != interaction.asset_id:
            target_session = None
        elif interaction.site_id and existing_call.site_id and existing_call.site_id != interaction.site_id:
            target_session = None
        elif existing_call.status in ("resolved", "closed") or target_session.status in ("RESOLVED", "CLOSED"):
            # Existing resolved incident followed by a new incident produces a new ticket
            target_session = None
        elif is_hazard and existing_call.priority != "critical":
            # Safety hazard override: A hazardous complaint cannot be silently absorbed into a routine ticket
            target_session = None
        else:
            # Same session incident retry: reuse existing ticket without violating OneToOne unique constraint
            if not interaction.escalated_call_id and not CustomerInteraction.objects.filter(escalated_call=existing_call).exists():
                interaction.escalated_call = existing_call
                interaction.resolved = False
                interaction.save(update_fields=["escalated_call", "resolved"])
            return existing_call

    zone = interaction.site.zone if interaction.site else None
    branch = zone.branch if zone and zone.branch else None
    technician = choose_technician(tenant, branch=branch, zone=zone)

    # Server-authoritative priority: safety hazards receive critical SLA priority
    is_hazard, _ = check_for_safety_hazard(interaction.question or "")
    if is_hazard or "IMMEDIATE SAFETY RISK" in (interaction.answer or ""):
        priority = "critical"
    else:
        priority = "normal"

    sla = _sla_for(tenant, priority)
    now = timezone.now()

    for attempt in range(5):
        try:
            with transaction.atomic():
                locked = CustomerInteraction.objects.select_for_update().filter(id=interaction.id).first()
                if not locked:
                    raise ValueError("Interaction record not found.")
                if locked.escalated_call_id:
                    return locked.escalated_call

                # Incident-level check under lock: ensure exactly one ServiceCall per incident
                session = None
                if target_session:
                    session = DiagnosticSession.objects.select_for_update().filter(id=target_session.id).first()
                    if session and session.escalated_call_id:
                        locked.escalated_call = session.escalated_call
                        locked.resolved = False
                        locked.save(update_fields=["escalated_call", "resolved"])
                        return session.escalated_call

                ref_list = []
                if interaction.source_refs:
                    for r in interaction.source_refs:
                        ref_title = r.get("reference") or r.get("title")
                        if ref_title:
                            ref_list.append(ref_title)
                ref_text = "\n".join(f"- {ref}" for ref in ref_list) if ref_list else "- None cited"

                # Keep the customer's authentic original reported complaint clean as primary complaint
                authentic_complaint = (interaction.question or "").strip()

                clean_note = (customer_note or "").strip()[:500]

                # Separate diagnostic guidance, references, and customer observations into technician notes
                tech_notes = (
                    f"Customer AI Self-Service Escalation at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                    f"Customer Original Complaint:\n{authentic_complaint}\n\n"
                    f"Customer Escalation Observation / Note:\n{clean_note or 'None provided'}\n\n"
                    f"Sources Cited:\n{ref_text}\n\n"
                    f"AI Diagnostic Guidance Provided:\n{interaction.answer[:3000]}"
                )

                call = create_service_call_record(
                    tenant=tenant,
                    complaint_type="AI Self-Service Escalation",
                    complaint_text=authentic_complaint,
                    technician_notes=tech_notes,
                    customer=interaction.customer,
                    site=interaction.site,
                    asset=interaction.asset,
                    zone=zone,
                    priority=priority,
                    sla=sla,
                    technician=technician,
                    contact_name=interaction.customer.contact_name,
                    contact_phone=interaction.customer.phone,
                    contact_email=interaction.customer.email,
                )
                locked.resolved = False
                locked.escalated_call = call
                locked.save(update_fields=["resolved", "escalated_call"])

                # Link active DiagnosticSession and record explicit observation event
                if session:
                    if not session.escalated_call_id:
                        session.escalated_call = call
                        session.status = "ESCALATED"
                        session.version += 1
                        session.save(update_fields=["escalated_call", "status", "version", "updated_at"])
                elif locked.asset_id:
                    session = DiagnosticSession.objects.create(
                        tenant=tenant,
                        customer=locked.customer,
                        site=locked.site,
                        asset=locked.asset,
                        complaint=authentic_complaint,
                        status="ESCALATED",
                        escalated_call=call,
                        version=1,
                    )

                if session:
                    if clean_note:
                        last_seq = session.events.aggregate(m=Max("seq_num"))["m"] or 0
                        DiagnosticEvent.objects.create(
                            tenant=tenant,
                            session=session,
                            seq_num=last_seq + 1,
                            event_type="OBSERVATION_RECORDED",
                            actor_role="customer",
                            payload={
                                "observation": clean_note,
                                "outcome": "UNABLE_TO_PERFORM" if "unable" in clean_note.lower() else "CUSTOMER_NOTE",
                                "interaction_id": interaction.id,
                            },
                        )

                    last_seq = session.events.aggregate(m=Max("seq_num"))["m"] or 0
                    DiagnosticEvent.objects.create(
                        tenant=tenant,
                        session=session,
                        seq_num=last_seq + 1,
                        event_type="SESSION_ESCALATED",
                        actor_role="customer",
                        payload={
                            "reason": "Escalated from Customer AI Support",
                            "service_call_id": call.id,
                            "servy_id": call.servy_id,
                            "interaction_id": interaction.id,
                            "customer_note": clean_note,
                        },
                    )

                interaction.refresh_from_db()
                return call
        except (IntegrityError, OperationalError):
            interaction.refresh_from_db()
            if interaction.escalated_call_id:
                return interaction.escalated_call
            time.sleep(0.05 * (attempt + 1))
            continue

    interaction.refresh_from_db()
    if interaction.escalated_call_id:
        return interaction.escalated_call
    raise RuntimeError("Could not allocate a unique service call ID after multiple attempts.")
