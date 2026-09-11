import io
import logging
from django.conf import settings
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.db import transaction
from django.db.models import Q, Count
from django.http import FileResponse, Http404
from django.middleware.csrf import get_token
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie

from rest_framework import permissions, status, throttling
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    Tenant, Customer, Site, Asset, Product, Project, ServiceCall, CallUpdate,
    KnowledgeDocument, KnowledgeChunk, CustomerInteraction, StaffProfile,
    TenantMembership, InventoryItem, PartRequest, OperationalZone
)
from core.security import get_current_membership
from core.services.rag import ask_rag
from core.services.ticketing import create_call_from_interaction
from core.services.indexing import index_document
from core.services.chroma_store import queue_chroma_deletion
from core.services.excel_io import export_calls_xlsx, import_calls_xlsx
from core.forms import MAX_KB_UPLOAD_BYTES

logger = logging.getLogger("servy.api")


# ----------------------------------------------------------------------
# Rate Throttles
# ----------------------------------------------------------------------
class LoginThrottle(throttling.ScopedRateThrottle):
    scope = "login"

class CustomerRAGThrottle(throttling.ScopedRateThrottle):
    scope = "customer_rag"

class EngineerRAGThrottle(throttling.ScopedRateThrottle):
    scope = "engineer_rag"

class UploadThrottle(throttling.ScopedRateThrottle):
    scope = "upload"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_user_context(request_or_user):
    """Derive user's tenant, role, customer (if any), and permissions."""
    user = request_or_user if hasattr(request_or_user, "is_authenticated") and not hasattr(request_or_user, "user") else getattr(request_or_user, "user", request_or_user)
    if not user.is_authenticated:
        return {
            "authenticated": False,
            "role": "anonymous",
            "tenant": None,
            "customer": None,
            "staff_profile": None,
            "membership": None,
            "can_view_confidential": False,
        }

    if hasattr(request_or_user, "session"):
        membership = get_current_membership(request_or_user)
    else:
        membership = TenantMembership.objects.filter(user=user, is_active=True, tenant__is_active=True).select_related("tenant", "customer").first()

    staff_profile = StaffProfile.objects.filter(user=user, is_active=True).select_related("tenant", "branch", "zone").first()

    if user.is_superuser:
        tenant = membership.tenant if membership else (staff_profile.tenant if staff_profile else Tenant.objects.first())
        role = "superuser"
        customer = membership.customer if membership else None
        can_view_confidential = True
    elif membership:
        tenant = membership.tenant
        role = membership.role
        customer = membership.customer
        can_view_confidential = membership.can_view_confidential
    elif staff_profile:
        tenant = staff_profile.tenant
        role = staff_profile.role
        customer = None
        can_view_confidential = False
    else:
        tenant = Tenant.objects.first()
        role = "guest"
        customer = None
        can_view_confidential = False

    return {
        "authenticated": True,
        "role": role,
        "tenant": tenant,
        "customer": customer,
        "staff_profile": staff_profile,
        "membership": membership,
        "can_view_confidential": can_view_confidential,
    }


def get_active_tenant(request):
    """Determine tenant strictly according to user's authorized profile."""
    if request.user.is_superuser:
        tenant_id = request.session.get("tenant_id") or request.session.get("active_tenant_id")
        if tenant_id:
            try:
                return Tenant.objects.get(id=tenant_id)
            except Tenant.DoesNotExist:
                pass
    ctx = get_user_context(request)
    return ctx.get("tenant")


# ----------------------------------------------------------------------
# Authentication & CSRF
# ----------------------------------------------------------------------
class CSRFBootstrapView(APIView):
    """Bootstrap endpoint returning CSRF token for React client."""
    permission_classes = [permissions.AllowAny]

    @method_decorator(ensure_csrf_cookie)
    def get(self, request):
        return Response({
            "detail": "CSRF cookie set",
            "csrfToken": get_token(request),
        })


class LoginView(APIView):
    """Session login endpoint with scoped rate limiting."""
    permission_classes = [permissions.AllowAny]
    throttle_classes = [LoginThrottle]

    @method_decorator(ensure_csrf_cookie)
    def post(self, request):
        username = request.data.get("username", "").strip()
        password = request.data.get("password", "").strip()

        if not username or not password:
            return Response({"detail": "Username and password required."}, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(request, username=username, password=password)
        if user is None:
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        auth_login(request, user)
        ctx = get_user_context(user)
        tenant = ctx["tenant"]

        return Response({
            "detail": "Login successful",
            "csrfToken": get_token(request),
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": ctx["role"],
                "tenant_id": tenant.id if tenant else None,
                "tenant_name": tenant.name if tenant else None,
                "customer_id": ctx["customer"].id if ctx["customer"] else None,
                "customer_name": ctx["customer"].name if ctx["customer"] else None,
            }
        })


class LogoutView(APIView):
    """Session logout endpoint."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        auth_logout(request)
        return Response({"detail": "Logged out successfully"})


class CurrentUserView(APIView):
    """Return profile details for current authenticated session."""
    permission_classes = [permissions.IsAuthenticated]

    @method_decorator(ensure_csrf_cookie)
    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        # Multi-tenant available tenants for superuser
        available_tenants = []
        if request.user.is_superuser:
            available_tenants = list(Tenant.objects.values("id", "name", "slug"))

        return Response({
            "user": {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email,
                "role": ctx["role"],
                "tenant_id": tenant.id if tenant else None,
                "tenant_name": tenant.name if tenant else None,
                "customer_id": ctx["customer"].id if ctx["customer"] else None,
                "customer_name": ctx["customer"].name if ctx["customer"] else None,
            },
            "available_tenants": available_tenants,
            "csrfToken": get_token(request),
        })


class SwitchTenantView(APIView):
    """Switch active tenant - restricted to superusers only."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        if not request.user.is_superuser:
            return Response(
                {"detail": "Tenant switching is only permitted for multi-tenant administrators."},
                status=status.HTTP_403_FORBIDDEN
            )
        tenant_id = request.data.get("tenant_id")
        try:
            target = Tenant.objects.get(id=tenant_id)
            request.session["active_tenant_id"] = target.id
            request.session["tenant_id"] = target.id
            return Response({"detail": f"Switched to tenant {target.name}", "tenant_id": target.id, "tenant_name": target.name})
        except Tenant.DoesNotExist:
            return Response({"detail": "Tenant not found."}, status=status.HTTP_404_NOT_FOUND)


# ----------------------------------------------------------------------
# Dashboard View
# ----------------------------------------------------------------------
class DashboardView(APIView):
    """Dashboard metrics scoped to user role and tenant."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)
        if not tenant:
            return Response({"detail": "Tenant context required."}, status=status.HTTP_400_BAD_REQUEST)

        role = ctx["role"]
        customer = ctx["customer"]

        data = {
            "role": role,
            "tenant_name": tenant.name,
        }

        if role == "customer" and customer:
            calls_qs = ServiceCall.objects.filter(tenant=tenant, customer=customer)
            data["stats"] = {
                "total_calls": calls_qs.count(),
                "open_calls": calls_qs.filter(status__in=["open", "assigned", "in_progress", "pending_parts"]).count(),
                "closed_calls": calls_qs.filter(status="closed").count(),
                "active_assets": Asset.objects.filter(tenant=tenant, customer=customer, status="active").count(),
            }
            data["recent_calls"] = list(calls_qs.order_by("-created_at")[:5].values(
                "id", "servy_id", "complaint_type", "status", "priority", "created_at"
            ))
            data["recent_interactions"] = list(CustomerInteraction.objects.filter(
                tenant=tenant, customer=customer
            ).order_by("-created_at")[:5].values("id", "question", "resolved", "created_at", "escalated_call_id"))

        else:
            # Staff / Technician / Admin
            calls_qs = ServiceCall.objects.filter(tenant=tenant)
            staff = ctx.get("staff_profile")
            my_calls = calls_qs.filter(technician=staff) if staff else calls_qs.none()

            data["stats"] = {
                "total_calls": calls_qs.count(),
                "open_calls": calls_qs.filter(status__in=["open", "assigned", "in_progress"]).count(),
                "assigned_to_me": my_calls.filter(status__in=["assigned", "in_progress"]).count(),
                "kb_documents": KnowledgeDocument.objects.filter(tenant=tenant, is_rag_enabled=True).count(),
            }
            data["recent_calls"] = list(calls_qs.order_by("-created_at")[:8].values(
                "id", "servy_id", "complaint_type", "status", "priority", "created_at",
                "customer__name", "asset__name"
            ))

        return Response(data)


# ----------------------------------------------------------------------
# Workflow 1: Customer AI Support
# ----------------------------------------------------------------------
class CustomerSupportContextView(APIView):
    """Fetch sites and assets for pre-ticket self-service."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)
        customer = ctx["customer"]

        # Staff can pass ?customer_id= to preview or assist
        if not customer and ctx["role"] in {"superuser", "tenant_admin", "branch_manager", "technician"}:
            cust_id = request.query_params.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            else:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer profile not found."}, status=status.HTTP_404_NOT_FOUND)

        sites = list(Site.objects.filter(tenant=tenant, customer=customer).values("id", "name", "city"))
        assets = list(Asset.objects.filter(tenant=tenant, customer=customer, status="active").select_related("product").values(
            "id", "name", "asset_code", "site_id", "model_number", "product__name"
        ))

        return Response({
            "customer": {"id": customer.id, "name": customer.name},
            "sites": sites,
            "assets": assets,
        })


class CustomerSupportQueryView(APIView):
    """Execute customer RAG query before ticket creation."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [CustomerRAGThrottle]

    def post(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)
        customer = ctx["customer"]

        if not customer and ctx["role"] in {"superuser", "tenant_admin", "branch_manager", "technician"}:
            cust_id = request.data.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            else:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer context required."}, status=status.HTTP_400_BAD_REQUEST)

        site_id = request.data.get("site_id")
        asset_id = request.data.get("asset_id")
        question = (request.data.get("question") or "").strip()

        if not site_id or not asset_id or not question:
            return Response({"detail": "Site, asset, and question are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate asset and site belong to customer
        try:
            site = Site.objects.get(tenant=tenant, customer=customer, id=site_id)
        except Site.DoesNotExist:
            return Response({"detail": "Selected site does not belong to this customer."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            asset = Asset.objects.get(tenant=tenant, customer=customer, site=site, id=asset_id)
        except Asset.DoesNotExist:
            return Response({"detail": "Selected asset does not belong to this customer or site."}, status=status.HTTP_400_BAD_REQUEST)

        # Run RAG: Customer mode (no confidential KB, no service call context, customer-scoped)
        rag_res = ask_rag(
            tenant=tenant,
            question=question,
            asset=asset,
            customer=customer,
            include_confidential=False,
            service_call=None,
            staff_mode=False,
        )

        retrieved_list = rag_res.get("retrieved", [])
        retrieval_backend = getattr(retrieved_list, "retrieval_backend", "fallback")

        interaction = CustomerInteraction.objects.create(
            tenant=tenant,
            created_by=request.user,
            customer=customer,
            site=site,
            asset=asset,
            question=question,
            answer=rag_res["answer"],
            source_refs=rag_res["references"],
        )

        return Response({
            "interaction_id": interaction.id,
            "answer": rag_res["answer"],
            "references": rag_res["references"],
            "engine": rag_res["engine"],
            "retrieval_backend": retrieval_backend,
        })


class CustomerSupportResolveView(APIView):
    """Mark customer interaction as resolved."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        try:
            interaction = CustomerInteraction.objects.get(tenant=tenant, id=pk)
        except CustomerInteraction.DoesNotExist:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] == "customer" and interaction.customer_id != ctx["customer"].id:
            return Response({"detail": "Unauthorized interaction ID."}, status=status.HTTP_404_NOT_FOUND)

        interaction.resolved = True
        interaction.save(update_fields=["resolved"])
        return Response({"status": "resolved", "interaction_id": interaction.id})


class CustomerSupportEscalateView(APIView):
    """Escalate unresolved customer interaction to exactly one Service Call."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        try:
            interaction = CustomerInteraction.objects.get(tenant=tenant, id=pk)
        except CustomerInteraction.DoesNotExist:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] == "customer" and interaction.customer_id != ctx["customer"].id:
            return Response({"detail": "Unauthorized interaction ID."}, status=status.HTTP_404_NOT_FOUND)

        # Idempotent ticket creation
        call = create_call_from_interaction(interaction)
        return Response({
            "call_id": call.id,
            "servy_id": call.servy_id,
            "complaint_type": call.complaint_type,
            "status": call.status,
            "priority": call.priority,
            "technician": call.technician.full_name if call.technician else "Unassigned",
            "message": f"Service Call #{call.servy_id} created successfully.",
        })


# ----------------------------------------------------------------------
# Workflow 2: Engineer Copilot
# ----------------------------------------------------------------------
class EngineerCopilotQueryView(APIView):
    """Technician Copilot scoped to an existing Service Call."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [EngineerRAGThrottle]

    def post(self, request):
        ctx = get_user_context(request)
        if ctx["role"] == "customer":
            return Response({"detail": "Engineer Copilot is restricted to service staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = get_active_tenant(request)
        call_id = request.data.get("call_id")
        question = (request.data.get("question") or "").strip()

        if not call_id or not question:
            return Response({"detail": "Service call ID and question are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            call = ServiceCall.objects.select_related("asset", "customer", "site").get(tenant=tenant, id=call_id)
        except ServiceCall.DoesNotExist:
            return Response({"detail": "Service call not found in this tenant."}, status=status.HTTP_404_NOT_FOUND)

        # Run RAG in staff mode: includes confidential manuals and previous resolutions
        rag_res = ask_rag(
            tenant=tenant,
            question=question,
            asset=call.asset,
            customer=call.customer,
            include_confidential=True,
            service_call=call,
            staff_mode=True,
        )

        retrieved_list = rag_res.get("retrieved", [])
        retrieval_backend = getattr(retrieved_list, "retrieval_backend", "fallback")

        return Response({
            "answer": rag_res["answer"],
            "references": rag_res["references"],
            "engine": rag_res["engine"],
            "retrieval_backend": retrieval_backend,
        })


# ----------------------------------------------------------------------
# Sites & Assets Endpoints
# ----------------------------------------------------------------------
class SitesAssetsView(APIView):
    """Retrieve assets for a specific site."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, site_id):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        site_qs = Site.objects.filter(tenant=tenant, id=site_id)
        if ctx["role"] == "customer" and ctx["customer"]:
            site_qs = site_qs.filter(customer=ctx["customer"])

        site = site_qs.first()
        if not site:
            return Response({"detail": "Site not found or access denied."}, status=status.HTTP_404_NOT_FOUND)

        assets = list(Asset.objects.filter(
            tenant=tenant, site=site, status="active"
        ).select_related("product").values("id", "name", "asset_code", "model_number", "product__name"))

        return Response(assets)


class AssetsListView(APIView):
    """List assets scoped by tenant and customer."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        qs = Asset.objects.filter(tenant=tenant).select_related("customer", "site", "product")
        if ctx["role"] == "customer" and ctx["customer"]:
            qs = qs.filter(customer=ctx["customer"])

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(asset_code__icontains=search) | Q(model_number__icontains=search))

        items = list(qs[:100].values(
            "id", "name", "asset_code", "model_number", "serial_number",
            "status", "customer__name", "site__name", "product__name"
        ))
        return Response(items)


# ----------------------------------------------------------------------
# Call Register Endpoints
# ----------------------------------------------------------------------
class CallRegisterView(APIView):
    """List service calls with search, filter, and tenant isolation."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        qs = ServiceCall.objects.filter(tenant=tenant).select_related(
            "customer", "site", "asset", "technician"
        )
        if ctx["role"] == "customer" and ctx["customer"]:
            qs = qs.filter(customer=ctx["customer"])

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(
                Q(complaint_text__icontains=search) |
                Q(complaint_type__icontains=search) |
                Q(servy_id__icontains=search) |
                Q(customer__name__icontains=search)
            )

        items = list(qs.order_by("-created_at")[:100].values(
            "id", "servy_id", "complaint_type", "status", "priority",
            "contact_name", "created_at", "customer__name", "site__name",
            "asset__name", "technician__full_name"
        ))
        return Response(items)


class CallDetailView(APIView):
    """Retrieve full service call details with updates and part requests."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        try:
            call = ServiceCall.objects.select_related(
                "customer", "site", "asset", "technician", "zone", "sla_policy"
            ).get(tenant=tenant, id=pk)
        except ServiceCall.DoesNotExist:
            return Response({"detail": "Call not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] == "customer" and call.customer_id != ctx["customer"].id:
            return Response({"detail": "Call not found."}, status=status.HTTP_404_NOT_FOUND)

        updates = list(call.updates.select_related("author").order_by("created_at").values(
            "id", "status", "note", "created_at", "author__full_name"
        ))
        parts = list(call.part_requests.order_by("-date").values(
            "id", "indent_id", "spare_description", "manager_status", "store_status", "date"
        ))

        return Response({
            "call": {
                "id": call.id,
                "servy_id": call.servy_id,
                "complaint_type": call.complaint_type,
                "complaint_text": call.complaint_text,
                "technician_notes": call.technician_notes,
                "resolution_text": call.resolution_text,
                "status": call.status,
                "priority": call.priority,
                "contact_name": call.contact_name,
                "contact_phone": call.contact_phone,
                "contact_email": call.contact_email,
                "created_at": call.created_at,
                "customer": {"id": call.customer.id, "name": call.customer.name} if call.customer else None,
                "site": {"id": call.site.id, "name": call.site.name} if call.site else None,
                "asset": {"id": call.asset.id, "name": call.asset.name, "model_number": call.asset.model_number} if call.asset else None,
                "technician": {"id": call.technician.id, "name": call.technician.full_name} if call.technician else None,
            },
            "updates": updates,
            "part_requests": parts,
        })


class CallExportView(APIView):
    """Export service calls to sanitized Excel format."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        if ctx["role"] == "customer":
            return Response({"detail": "Export is restricted to staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = get_active_tenant(request)
        return export_calls_xlsx(tenant)


class CallImportView(APIView):
    """Import service calls from Excel with safety bounds."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        ctx = get_user_context(request)
        if ctx["role"] not in {"superuser", "admin", "manager"}:
            return Response({"detail": "Import is restricted to managers."}, status=status.HTTP_403_FORBIDDEN)

        tenant = get_active_tenant(request)
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response({"detail": "File required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            created, skipped = import_calls_xlsx(tenant, uploaded)
            return Response({
                "detail": f"Successfully imported {created} calls (skipped {skipped}).",
                "created": created,
                "skipped": skipped,
            })
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.exception("Call import error")
            return Response({"detail": "Failed to import file."}, status=status.HTTP_400_BAD_REQUEST)


# ----------------------------------------------------------------------
# Knowledge Base Endpoints
# ----------------------------------------------------------------------
class KnowledgeListView(APIView):
    """List knowledge articles scoped to role and customer."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        qs = KnowledgeDocument.objects.filter(tenant=tenant).select_related("product", "asset", "customer")
        if ctx["role"] == "customer":
            # Strict non-confidential customer scoping
            qs = qs.filter(is_confidential=False)
            if ctx["customer"]:
                qs = qs.filter(Q(customer__isnull=True) | Q(customer=ctx["customer"]))

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(Q(title__icontains=search) | Q(tags__icontains=search) | Q(description__icontains=search))

        items = list(qs[:100].values(
            "id", "title", "doc_type", "is_confidential", "is_rag_enabled",
            "index_status", "index_version", "updated_at", "product__name", "asset__name"
        ))
        return Response(items)


class KnowledgeDetailView(APIView):
    """Retrieve details of a knowledge document."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        try:
            doc = KnowledgeDocument.objects.select_related("product", "asset", "customer").get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] == "customer" and doc.is_confidential:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        chunks = list(doc.chunks.order_by("chunk_index").values("id", "chunk_index", "heading", "text", "is_quarantined"))

        return Response({
            "document": {
                "id": doc.id,
                "title": doc.title,
                "description": doc.description,
                "doc_type": doc.doc_type,
                "tags": doc.tags,
                "content_text": doc.content_text,
                "has_file": bool(doc.file),
                "original_filename": doc.original_filename,
                "is_confidential": doc.is_confidential,
                "is_rag_enabled": doc.is_rag_enabled,
                "index_status": doc.index_status,
                "index_version": doc.index_version,
                "indexed_at": doc.indexed_at,
                "index_error": doc.index_error,
                "product_name": doc.product.name if doc.product else None,
                "asset_name": doc.asset.name if doc.asset else None,
            },
            "chunks": chunks,
        })


class KnowledgeIndexStatusView(APIView):
    """Check indexing status of an uploaded document."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        tenant = get_active_tenant(request)
        try:
            doc = KnowledgeDocument.objects.get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            "id": doc.id,
            "status": doc.index_status,
            "version": doc.index_version,
            "indexed_at": doc.indexed_at,
            "error": doc.index_error,
        })


class KnowledgeUploadView(APIView):
    """Upload and index Knowledge Base documents."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [UploadThrottle]

    def post(self, request):
        ctx = get_user_context(request)
        if ctx["role"] == "customer":
            return Response({"detail": "KB uploads are restricted to staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = get_active_tenant(request)
        title = (request.data.get("title") or "").strip()
        doc_type = request.data.get("doc_type", "reference")
        is_confidential = str(request.data.get("is_confidential", "")).lower() in {"1", "true", "yes"}
        content_text = (request.data.get("content_text") or "").strip()
        uploaded_file = request.FILES.get("file")

        if not title:
            return Response({"detail": "Title is required."}, status=status.HTTP_400_BAD_REQUEST)

        if not content_text and not uploaded_file:
            return Response({"detail": "Provide either text content or a file."}, status=status.HTTP_400_BAD_REQUEST)

        # File validation
        if uploaded_file:
            if uploaded_file.size > MAX_KB_UPLOAD_BYTES:
                return Response({"detail": f"File exceeds max size of {MAX_KB_UPLOAD_BYTES // (1024*1024)}MB."}, status=status.HTTP_400_BAD_REQUEST)
            head = uploaded_file.read(8)
            uploaded_file.seek(0)
            name_lower = uploaded_file.name.lower()
            if name_lower.endswith(".pdf") and not head.startswith(b"%PDF-"):
                return Response({"detail": "File header does not match PDF specification."}, status=status.HTTP_400_BAD_REQUEST)
            if name_lower.endswith(".docx") and not head.startswith(b"PK"):
                return Response({"detail": "File header does not match DOCX specification."}, status=status.HTTP_400_BAD_REQUEST)

        doc = KnowledgeDocument.objects.create(
            tenant=tenant,
            title=title,
            doc_type=doc_type,
            is_confidential=is_confidential,
            content_text=content_text,
            file=uploaded_file,
            original_filename=uploaded_file.name if uploaded_file else "",
            is_rag_enabled=True,
            index_status="INDEXING",
        )

        # Process and index
        chunks_indexed = index_document(doc)

        return Response({
            "id": doc.id,
            "title": doc.title,
            "chunks_count": chunks_indexed,
            "index_status": doc.index_status,
            "message": f"Document created and indexed ({chunks_indexed} chunks).",
        }, status=status.HTTP_201_CREATED)


class KnowledgeDownloadView(APIView):
    """Download protected KB attachment."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = get_user_context(request)
        tenant = get_active_tenant(request)

        try:
            doc = KnowledgeDocument.objects.get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            raise Http404("Document not found.")

        if ctx["role"] == "customer" and doc.is_confidential:
            raise Http404("Document not found.")

        if not doc.file:
            return Response({"detail": "No file attached to this document."}, status=status.HTTP_404_NOT_FOUND)

        return FileResponse(doc.file.open("rb"), as_attachment=True, filename=doc.original_filename or doc.file.name)


class KnowledgeDeleteView(APIView):
    """Delete KB document from SQLite and purge vector index from Chroma."""
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, pk):
        ctx = get_user_context(request)
        if ctx["role"] not in {"superuser", "admin", "manager"}:
            return Response({"detail": "Deletion is restricted to administrators."}, status=status.HTTP_403_FORBIDDEN)

        tenant = get_active_tenant(request)
        try:
            doc = KnowledgeDocument.objects.get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        doc_id = doc.id
        with transaction.atomic():
            doc.delete()
            queue_chroma_deletion(doc_id)

        return Response({"status": "deleted", "document_id": doc_id})


# ----------------------------------------------------------------------
# Read-Only Supporting Endpoints
# ----------------------------------------------------------------------
class ProjectsListView(APIView):
    """Read-only projects list."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        tenant = get_active_tenant(request)
        ctx = get_user_context(request)
        qs = Project.objects.filter(tenant=tenant).select_related("customer", "site")
        if ctx["role"] == "customer" and ctx["customer"]:
            qs = qs.filter(customer=ctx["customer"])
        return Response(list(qs[:100].values("id", "code", "name", "status", "customer__name", "site__name")))


class InventoryListView(APIView):
    """Read-only inventory list."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        tenant = get_active_tenant(request)
        qs = InventoryItem.objects.filter(tenant=tenant).select_related("category")
        return Response(list(qs[:100].values("id", "item_code", "name", "quantity_on_hand", "reorder_level", "category__name")))


class PartRequestsListView(APIView):
    """Read-only part requests list."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        tenant = get_active_tenant(request)
        qs = PartRequest.objects.filter(tenant=tenant).select_related("service_call", "requester")
        return Response(list(qs[:100].values(
            "id", "indent_id", "spare_description", "manager_status", "store_status", "date",
            "service_call__servy_id", "requester__full_name"
        )))


class OperationsListView(APIView):
    """Read-only zones and operational structure."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        tenant = get_active_tenant(request)
        zones = list(OperationalZone.objects.filter(tenant=tenant).select_related("branch").values("id", "name", "code", "branch__name"))
        return Response({"zones": zones})


class UsersListView(APIView):
    """Read-only staff directory."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = get_user_context(request)
        if ctx["role"] == "customer":
            return Response({"detail": "Access restricted."}, status=status.HTTP_403_FORBIDDEN)
        tenant = get_active_tenant(request)
        staff = list(StaffProfile.objects.filter(tenant=tenant).select_related("branch", "zone").values(
            "id", "full_name", "role", "email", "phone", "branch__name", "zone__name", "is_active"
        ))
        return Response(staff)
