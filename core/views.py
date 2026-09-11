from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, HttpResponse
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.utils.http import url_has_allowed_host_and_scheme

from .forms import (
    CallCopilotForm, CallImportForm, CustomerRAGQueryForm,
    KnowledgeDocumentForm, StaffCustomerSupportForm, StaffRAGQueryForm,
)

from .models import (
    ApprovalRequest, Asset, BusinessVocabulary, ChecklistTemplate, CorporateIdentity,
    CustomerInteraction, CustomizationSetting, EmailNotificationRule, ExpenseClaim,
    InventoryItem, KnowledgeDocument, LocalPurchase, OperationalZone, PartRequest,
    Project, ReturnRequest, ServiceCall, SLAPolicy, StaffProfile, TenantMembership,
    VoucherClaim,
)
from .security import any_member, get_current_membership, roles_required, rate_limit_exceeded
from .services.document_loader import UnsafeDocumentError
from .services.excel_io import export_calls_xlsx, import_calls_xlsx
from .services.indexing import index_document
from .services.rag import ask_rag
from .services.ticketing import create_call_from_interaction
from .services.audit import audit_event

STAFF_ROLES = ("admin", "manager", "technician", "store_admin", "store_operator")
TECH_ROLES = ("admin", "manager", "technician")
ADMIN_ROLES = ("admin", "manager")


def _membership(request):
    membership = getattr(request, "servy_membership", None) or get_current_membership(request)
    if membership is None:
        raise PermissionDenied("No active tenant membership is assigned to this account.")
    return membership


def _tenant(request):
    return _membership(request).tenant


@any_member
def dashboard(request):
    membership = _membership(request)
    if membership.role == "customer":
        return redirect("customer_portal")
    if membership.role in {"store_admin", "store_operator"}:
        return redirect("inventory")
    tenant = membership.tenant
    today = timezone.localdate()
    context = {
        "tenant": tenant,
        "asset_count": Asset.objects.filter(tenant=tenant).count(),
        "call_count": ServiceCall.objects.filter(tenant=tenant).count(),
        "open_calls": ServiceCall.objects.filter(tenant=tenant, status__in=["open", "assigned", "in_progress"]).count(),
        "kb_count": KnowledgeDocument.objects.filter(tenant=tenant).count(),
        "interactions_today": CustomerInteraction.objects.filter(tenant=tenant, created_at__date=today).count(),
        "escalations_today": CustomerInteraction.objects.filter(tenant=tenant, created_at__date=today, escalated_call__isnull=False).count(),
        "recent_calls": ServiceCall.objects.filter(tenant=tenant).select_related("customer", "asset", "technician")[:8],
        "recent_interactions": CustomerInteraction.objects.filter(tenant=tenant).select_related("customer", "asset")[:6],
    }
    return render(request, "core/dashboard.html", context)


@login_required
@require_POST
def switch_tenant(request):
    membership = get_object_or_404(
        TenantMembership.objects.select_related("tenant"),
        user=request.user,
        tenant_id=request.POST.get("tenant_id"),
        is_active=True,
        tenant__is_active=True,
    )
    request.session["tenant_id"] = membership.tenant_id
    audit_event(request, "tenant.switch", tenant=membership.tenant, object_type="tenant", object_id=membership.tenant_id)
    next_url = request.POST.get("next") or ""
    if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        return redirect(next_url)
    return redirect("dashboard")


@roles_required(*TECH_ROLES)
def knowledge_base(request):
    tenant = _tenant(request)
    membership = _membership(request)
    q = request.GET.get("q", "").strip()
    docs = KnowledgeDocument.objects.filter(tenant=tenant).select_related("product", "asset", "customer")
    if not membership.can_view_confidential:
        docs = docs.filter(is_confidential=False)
    if q:
        docs = docs.filter(Q(title__icontains=q) | Q(description__icontains=q) | Q(tags__icontains=q))
    paginator = Paginator(docs, 24)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "core/knowledge_base.html", {"page_obj": page, "q": q, "doc_count": docs.count()})




@roles_required(*TECH_ROLES)
def knowledge_download(request, pk):
    """Authenticated, tenant-scoped download for uploaded KB files.

    Confidential files are never exposed by a raw /media URL; access is checked
    against the current membership on every request.
    """
    tenant = _tenant(request)
    membership = _membership(request)
    doc = get_object_or_404(KnowledgeDocument, tenant=tenant, pk=pk)
    if doc.is_confidential and not membership.can_view_confidential:
        raise PermissionDenied("You do not have permission to download this confidential document.")
    if not doc.file:
        raise Http404("No uploaded file is attached to this knowledge item.")
    try:
        handle = doc.file.open("rb")
    except FileNotFoundError as exc:
        raise Http404("Knowledge file is not available on this server.") from exc
    filename = doc.original_filename or Path(doc.file.name).name
    audit_event(request, "knowledge.download", tenant=tenant, object_type="knowledge_document", object_id=doc.id, metadata={"confidential": doc.is_confidential})
    response = FileResponse(handle, as_attachment=True, filename=filename)
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, no-store"
    return response


@roles_required(*ADMIN_ROLES)
def knowledge_upload(request):
    tenant = _tenant(request)
    if request.method == "POST":
        form = KnowledgeDocumentForm(request.POST, request.FILES, tenant=tenant)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.tenant = tenant
            if request.FILES.get("file"):
                doc.original_filename = Path(request.FILES["file"].name).name[:255]
            doc.save()
            try:
                chunks = index_document(doc)
            except UnsafeDocumentError as exc:
                doc.delete()
                messages.error(request, f"Document rejected by local safety checks: {exc}")
                return render(request, "core/knowledge_form.html", {"form": form})
            quarantined = doc.chunks.filter(is_quarantined=True).count()
            audit_event(request, "knowledge.create", tenant=tenant, object_type="knowledge_document", object_id=doc.id, metadata={"chunks": chunks, "quarantined_chunks": quarantined, "confidential": doc.is_confidential})
            note = f"; {quarantined} suspicious chunk(s) quarantined" if quarantined else ""
            messages.success(request, f"Knowledge item saved and indexed into {chunks} RAG chunks{note}.")
            return redirect("knowledge_base")
    else:
        form = KnowledgeDocumentForm(tenant=tenant)
    return render(request, "core/knowledge_form.html", {"form": form})


@roles_required(*ADMIN_ROLES)
@require_POST
def reindex_document(request, pk):
    tenant = _tenant(request)
    doc = get_object_or_404(KnowledgeDocument, tenant=tenant, pk=pk)
    try:
        chunks = index_document(doc)
        audit_event(request, "knowledge.reindex", tenant=tenant, object_type="knowledge_document", object_id=doc.id, metadata={"chunks": chunks})
        messages.success(request, f"Re-indexed {chunks} chunks for {doc.title}.")
    except UnsafeDocumentError as exc:
        messages.error(request, f"Re-index blocked by safety checks: {exc}")
    return redirect("knowledge_base")


@roles_required(*STAFF_ROLES)
def assets(request):
    tenant = _tenant(request)
    q = request.GET.get("q", "").strip()
    qs = Asset.objects.filter(tenant=tenant).select_related("product", "product__brand", "customer", "site")
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(asset_code__icontains=q) | Q(model_number__icontains=q) | Q(product__name__icontains=q))
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    return render(request, "core/assets.html", {"page_obj": page, "q": q})


@roles_required(*STAFF_ROLES)
def inventory(request):
    tenant = _tenant(request)
    q = request.GET.get("q", "").strip()
    qs = InventoryItem.objects.filter(tenant=tenant).select_related("brand", "product", "branch")
    if q:
        qs = qs.filter(Q(spare_name__icontains=q) | Q(category__icontains=q) | Q(ipn__icontains=q) | Q(product__name__icontains=q))
    page = Paginator(qs, 20).get_page(request.GET.get("page"))
    return render(request, "core/inventory.html", {"page_obj": page, "q": q})


@roles_required(*STAFF_ROLES)
def projects(request):
    tenant = _tenant(request)
    qs = Project.objects.filter(tenant=tenant).select_related("customer", "site").prefetch_related("assets")
    page = Paginator(qs, 20).get_page(request.GET.get("page"))
    needs_attention = qs.filter(Q(handover_date__lt=timezone.localdate()) | Q(status="hold")).count()
    return render(request, "core/projects.html", {
        "page_obj": page,
        "total": qs.count(),
        "active": qs.filter(status="active").count(),
        "needs_attention": needs_attention,
    })


@roles_required(*STAFF_ROLES)
def part_requests(request):
    tenant = _tenant(request)
    qs = PartRequest.objects.filter(tenant=tenant).select_related("requester", "approver", "spare")
    page = Paginator(qs, 20).get_page(request.GET.get("page"))
    return render(request, "core/part_requests.html", {"page_obj": page})


@roles_required(*ADMIN_ROLES)
def users(request):
    tenant = _tenant(request)
    q = request.GET.get("q", "").strip()
    role = request.GET.get("role", "").strip()
    qs = StaffProfile.objects.filter(tenant=tenant).select_related("branch")
    if q:
        qs = qs.filter(Q(full_name__icontains=q) | Q(email__icontains=q) | Q(phone__icontains=q))
    if role:
        qs = qs.filter(role=role)
    page = Paginator(qs.order_by("full_name"), 25).get_page(request.GET.get("page"))
    return render(request, "core/users.html", {"page_obj": page, "q": q, "role": role, "roles": StaffProfile.ROLE_CHOICES})


@roles_required(*TECH_ROLES)
def call_register(request):
    tenant = _tenant(request)
    q = request.GET.get("q", "").strip()
    status = request.GET.get("status", "").strip()
    qs = ServiceCall.objects.filter(tenant=tenant).select_related("customer", "site", "project", "asset", "technician")
    if q:
        text_filter = (
            Q(complaint_type__icontains=q) | Q(complaint_text__icontains=q)
            | Q(customer__name__icontains=q) | Q(asset__name__icontains=q)
        )
        if q.isdigit():
            text_filter |= Q(servy_id=int(q))
        qs = qs.filter(text_filter)
    if status:
        qs = qs.filter(status=status)
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    return render(request, "core/calls.html", {
        "page_obj": page, "q": q, "status": status,
        "import_form": CallImportForm(),
        "can_import_export": _membership(request).role in ADMIN_ROLES,
    })


@roles_required(*TECH_ROLES)
def call_detail(request, pk):
    tenant = _tenant(request)
    call = get_object_or_404(
        ServiceCall.objects.select_related("customer", "site", "project", "asset", "asset__product", "technician"),
        tenant=tenant,
        pk=pk,
    )
    result = None
    interaction = None
    form = CallCopilotForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        if rate_limit_exceeded(request, "rag-call-detail"):
            return HttpResponse("Too many AI requests. Please wait briefly and try again.", status=429)
        result = ask_rag(
            tenant=tenant,
            question=form.cleaned_data["question"],
            asset=call.asset,
            customer=call.customer,
            include_confidential=_membership(request).can_view_confidential,
            service_call=call,
            staff_mode=True,
        )
        audit_event(request, "rag.query", tenant=tenant, object_type="service_call", object_id=call.id, metadata={"mode": "call_detail", "references": len(result["references"]), "engine": result["engine"]})
        interaction = CustomerInteraction.objects.create(
            tenant=tenant,
            created_by=request.user,
            customer=call.customer,
            site=call.site,
            asset=call.asset,
            service_call=call,
            question=form.cleaned_data["question"],
            answer=result["answer"],
            source_refs=result["references"],
        )
    return render(request, "core/call_detail.html", {
        "call": call,
        "updates": call.updates.select_related("author").all(),
        "form": form,
        "result": result,
        "interaction": interaction,
    })


@roles_required(*ADMIN_ROLES)
def call_export(request):
    tenant = _tenant(request)
    audit_event(request, "call.export", tenant=tenant, object_type="service_call", metadata={"format": "xlsx"})
    return export_calls_xlsx(tenant)


@roles_required(*ADMIN_ROLES)
@require_POST
def call_import(request):
    tenant = _tenant(request)
    form = CallImportForm(request.POST, request.FILES)
    if form.is_valid():
        try:
            created, skipped = import_calls_xlsx(tenant, form.cleaned_data["file"])
            audit_event(request, "call.import", tenant=tenant, object_type="service_call", metadata={"created": created, "skipped": skipped})
            messages.success(request, f"Imported {created} calls; skipped {skipped} invalid or duplicate rows.")
        except ValueError as exc:
            messages.error(request, str(exc))
    else:
        messages.error(request, "Please upload a valid .xlsx file.")
    return redirect("call_register")


@roles_required(*TECH_ROLES)
def rag_assistant(request):
    tenant = _tenant(request)
    membership = _membership(request)
    result = None
    interaction = None
    form = StaffRAGQueryForm(request.POST or None, tenant=tenant)
    if request.method == "POST" and form.is_valid():
        if rate_limit_exceeded(request, "rag-engineer"):
            return HttpResponse("Too many AI requests. Please wait briefly and try again.", status=429)
        customer = form.cleaned_data["customer"]
        asset = form.cleaned_data["asset"]
        site = form.cleaned_data["site"] or asset.site
        service_call = form.cleaned_data.get("service_call")
        question = form.cleaned_data["question"]
        result = ask_rag(
            tenant=tenant,
            question=question,
            asset=asset,
            customer=customer,
            include_confidential=membership.can_view_confidential,
            service_call=service_call,
            staff_mode=True,
        )
        audit_event(request, "rag.query", tenant=tenant, object_type="asset", object_id=asset.id, metadata={"mode": "engineer", "references": len(result["references"]), "engine": result["engine"], "service_call_id": service_call.id if service_call else None})
        interaction = CustomerInteraction.objects.create(
            tenant=tenant, created_by=request.user, customer=customer, site=site,
            asset=asset, service_call=service_call, question=question,
            answer=result["answer"], source_refs=result["references"],
        )
    return render(request, "core/rag_assistant.html", {"form": form, "result": result, "interaction": interaction, "staff_mode": True})


@any_member
def customer_portal(request):
    membership = _membership(request)
    tenant = membership.tenant
    if membership.role != "customer" and membership.role not in TECH_ROLES:
        raise PermissionDenied("This role cannot use the customer support workflow.")
    result = None
    interaction = None

    active_customer = None
    if membership.role == "customer":
        customer = membership.customer
        if customer is None:
            raise PermissionDenied("Customer account is not linked to a customer record.")
        active_customer = customer
        form = CustomerRAGQueryForm(request.POST or None, tenant=tenant, customer=customer)
        if request.method == "POST" and form.is_valid():
            if rate_limit_exceeded(request, "rag-customer"):
                return HttpResponse("Too many AI requests. Please wait briefly and try again.", status=429)
            asset = form.cleaned_data["asset"]
            site = form.cleaned_data["site"] or asset.site
            question = form.cleaned_data["question"]
            result = ask_rag(tenant=tenant, question=question, asset=asset, customer=customer, include_confidential=False, staff_mode=False)
            audit_event(request, "rag.query", tenant=tenant, object_type="asset", object_id=asset.id, metadata={"mode": "customer", "references": len(result["references"]), "engine": result["engine"]})
            interaction = CustomerInteraction.objects.create(
                tenant=tenant, created_by=request.user, customer=customer, site=site,
                asset=asset, question=question, answer=result["answer"], source_refs=result["references"],
            )
    else:
        form = StaffCustomerSupportForm(request.POST or None, tenant=tenant)
        if request.method == "POST" and form.is_valid():
            if rate_limit_exceeded(request, "rag-support-staff"):
                return HttpResponse("Too many AI requests. Please wait briefly and try again.", status=429)
            customer = form.cleaned_data["customer"]
            active_customer = customer
            asset = form.cleaned_data["asset"]
            site = form.cleaned_data["site"] or asset.site
            question = form.cleaned_data["question"]
            result = ask_rag(tenant=tenant, question=question, asset=asset, customer=customer, include_confidential=False, staff_mode=False)
            audit_event(request, "rag.query", tenant=tenant, object_type="asset", object_id=asset.id, metadata={"mode": "support_staff", "references": len(result["references"]), "engine": result["engine"]})
            interaction = CustomerInteraction.objects.create(
                tenant=tenant, created_by=request.user, customer=customer, site=site,
                asset=asset, question=question, answer=result["answer"], source_refs=result["references"],
            )

    import json
    asset_site_map = {}
    if "asset" in form.fields:
        for a in form.fields["asset"].queryset:
            asset_site_map[str(a.id)] = str(a.site_id) if a.site_id else ""
    asset_site_map_json = json.dumps(asset_site_map)

    context = {
        "form": form,
        "result": result,
        "interaction": interaction,
        "customer": active_customer,
        "is_customer_view": (membership.role == "customer"),
        "asset_site_map_json": asset_site_map_json,
    }
    return render(request, "core/customer_portal.html", context)


def _interaction_for_user(request, interaction_id):
    membership = _membership(request)
    if membership.role != "customer" and membership.role not in TECH_ROLES:
        raise PermissionDenied("This role cannot modify customer-support interactions.")
    qs = CustomerInteraction.objects.filter(tenant=membership.tenant)
    if membership.role == "customer":
        qs = qs.filter(customer=membership.customer, created_by=request.user)
    return get_object_or_404(qs, pk=interaction_id)


@any_member
@require_POST
def mark_resolved(request, interaction_id):
    interaction = _interaction_for_user(request, interaction_id)
    interaction.resolved = True
    interaction.save(update_fields=["resolved"])
    audit_event(request, "interaction.resolved", tenant=interaction.tenant, object_type="customer_interaction", object_id=interaction.id)
    messages.success(request, "Great — the interaction has been marked resolved without creating a service call.")
    return redirect("customer_portal" if _membership(request).role == "customer" else "rag_assistant")


@any_member
@require_POST
def escalate_interaction(request, interaction_id):
    interaction = _interaction_for_user(request, interaction_id)
    if interaction.escalated_call:
        call = interaction.escalated_call
    else:
        call = create_call_from_interaction(interaction)
    audit_event(request, "interaction.escalated", tenant=interaction.tenant, object_type="service_call", object_id=call.id, metadata={"servy_id": call.servy_id})
    tech_name = call.technician.full_name if call.technician else "None (Open for assignment)"
    messages.success(request, f"Service Call Created | Call ID: #{call.servy_id} | Status: {call.get_status_display()} | Assigned Technician: {tech_name}")
    return redirect("customer_portal" if _membership(request).role == "customer" else "call_register")



@roles_required(*TECH_ROLES)
def operations(request):
    tenant = _tenant(request)
    return render(request, "core/operations.html", {
        "sla_policies": SLAPolicy.objects.filter(tenant=tenant),
        "zones": OperationalZone.objects.filter(tenant=tenant),
        "checklists": ChecklistTemplate.objects.filter(tenant=tenant).select_related("product"),
        "local_purchases": LocalPurchase.objects.filter(tenant=tenant).select_related("service_call", "requested_by")[:20],
        "expense_claims": ExpenseClaim.objects.filter(tenant=tenant).select_related("service_call", "claimant")[:20],
        "approvals": ApprovalRequest.objects.filter(tenant=tenant).select_related("service_call", "requested_by", "approver")[:20],
        "voucher_claims": VoucherClaim.objects.filter(tenant=tenant).select_related("service_call", "claimant")[:20],
        "returns": ReturnRequest.objects.filter(tenant=tenant).select_related("service_call", "inventory_item", "requested_by")[:20],
        "corporate_identity": CorporateIdentity.objects.filter(tenant=tenant).first(),
        "business_vocabulary": BusinessVocabulary.objects.filter(tenant=tenant).order_by("key")[:20],
        "customizations": CustomizationSetting.objects.filter(tenant=tenant, is_active=True).order_by("category", "key")[:20],
        "email_rules": EmailNotificationRule.objects.filter(tenant=tenant).order_by("event")[:20],
    })
