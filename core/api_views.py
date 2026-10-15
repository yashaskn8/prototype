import os
import logging
from django.conf import settings
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.core.exceptions import ValidationError, PermissionDenied
from django.db import transaction
from django.db.models import Q, Count, Max
from django.http import FileResponse, Http404
from django.middleware.csrf import get_token
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie

from rest_framework import permissions, status, throttling, exceptions
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    Tenant, Customer, Site, Asset, Product, Project, ServiceCall, CallUpdate,
    KnowledgeDocument, KnowledgeChunk, CustomerInteraction, StaffProfile,
    TenantMembership, InventoryItem, PartRequest, OperationalZone,
    ProductCategory, ProductDomain, Brand, DiagnosticSession, DiagnosticEvent,
    CustomerConversation, CustomerMessage, SourceProvenanceRecord, ProcedureApproval
)
from core.security import (
    require_api_context, authorized_knowledge_queryset,
    STAFF_ROLES, TECH_ROLES, ADMIN_ROLES, CUSTOMER_ROLES, ALL_ROLES
)
from core.services.rag import ask_rag
from core.services.customer_diagnostic import check_for_safety_hazard
from core.services.ticketing import create_call_from_interaction
from core.services.indexing import index_document
from core.services.document_loader import UnsafeDocumentError
from core.services.chroma_store import queue_chroma_deletion
from core.services.excel_io import export_calls_xlsx, import_calls_xlsx
from core.services.starlly_api import (
    get_starlly_client,
    get_authorized_org_for_tenant,
    StarllyConfigurationError,
    StarllyAuthenticationError,
    StarllyTimeoutError,
    StarllyConnectionError,
    StarllyNotFoundError,
    StarllyApiError,
    StarllyTenantIsolationError,
)
from core.services.customer_diagnostic import (
    build_structured_diagnostic_contract,
    check_for_safety_hazard,
    check_for_vague_complaint,
    get_or_create_customer_session,
    record_customer_action_presentation,
    resolve_customer_session,
)
from core.diagnostics.schemas import (
    EVENT_CLARIFICATION_RECORDED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_SESSION_ESCALATED,
)
from core.services.customer_isolation import (
    get_customer_data_mode,
    is_real_tenant,
    is_synthetic_asset,
    is_synthetic_customer,
    validate_customer_serving_policy,
)
from rest_framework.exceptions import PermissionDenied
from core.forms import MAX_KB_UPLOAD_BYTES

logger = logging.getLogger("servy.api")

ALLOWED_KB_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


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

        tenant = None
        role = "guest"
        customer_id = None
        customer_name = None

        try:
            ctx = require_api_context(request)
            tenant = ctx["tenant"]
            role = ctx["role"]
            if ctx["customer"]:
                customer_id = ctx["customer"].id
                customer_name = ctx["customer"].name
        except (exceptions.ValidationError, exceptions.ParseError):
            # Superuser with multiple active tenants and none selected yet
            if user.is_superuser:
                role = "superuser"
            else:
                raise

        available_tenants = []
        if user.is_superuser:
            available_tenants = list(Tenant.objects.filter(is_active=True).values("id", "name", "slug"))

        return Response({
            "detail": "Login successful",
            "csrfToken": get_token(request),
            "available_tenants": available_tenants,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": role,
                "tenant_id": tenant.id if tenant else None,
                "tenant_name": tenant.name if tenant else None,
                "customer_id": customer_id,
                "customer_name": customer_name,
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
        tenant = None
        role = "guest"
        customer_id = None
        customer_name = None

        try:
            ctx = require_api_context(request)
            tenant = ctx["tenant"]
            role = ctx["role"]
            if ctx["customer"]:
                customer_id = ctx["customer"].id
                customer_name = ctx["customer"].name
        except (exceptions.ValidationError, exceptions.ParseError):
            if request.user.is_superuser:
                role = "superuser"
            else:
                raise

        available_tenants = []
        if request.user.is_superuser:
            available_tenants = list(Tenant.objects.filter(is_active=True).values("id", "name", "slug"))

        return Response({
            "user": {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email,
                "role": role,
                "tenant_id": tenant.id if tenant else None,
                "tenant_name": tenant.name if tenant else None,
                "customer_id": customer_id,
                "customer_name": customer_name,
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
            target = Tenant.objects.get(id=tenant_id, is_active=True)
            request.session["active_tenant_id"] = target.id
            request.session["tenant_id"] = target.id
            return Response({"detail": f"Switched to tenant {target.name}", "tenant_id": target.id, "tenant_name": target.name})
        except Tenant.DoesNotExist:
            return Response({"detail": "Active tenant not found."}, status=status.HTTP_404_NOT_FOUND)


# ----------------------------------------------------------------------
# Dashboard View
# ----------------------------------------------------------------------
class DashboardView(APIView):
    """Dashboard metrics scoped to user role and tenant."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        data = {
            "role": role,
            "tenant_name": tenant.name,
            "tenant": {"id": tenant.id, "name": tenant.name, "slug": tenant.slug},
        }

        if role in CUSTOMER_ROLES and customer:
            calls_qs = ServiceCall.objects.filter(tenant=tenant, customer=customer)
            interactions_qs = CustomerInteraction.objects.filter(tenant=tenant, customer=customer)
            stats = {
                "total_calls": calls_qs.count(),
                "service_calls": calls_qs.count(),
                "open_calls": calls_qs.filter(status__in=["open", "assigned", "in_progress", "pending_parts"]).count(),
                "closed_calls": calls_qs.filter(status="closed").count(),
                "active_assets": Asset.objects.filter(tenant=tenant, customer=customer, status="active").count(),
                "assets": Asset.objects.filter(tenant=tenant, customer=customer, status="active").count(),
                "knowledge_documents": KnowledgeDocument.objects.filter(tenant=tenant, is_rag_enabled=True).count(),
                "interactions_today": interactions_qs.filter(created_at__date=timezone.now().date()).count(),
                "escalations_today": interactions_qs.filter(created_at__date=timezone.now().date(), escalated_call__isnull=False).count(),
            }
            data["stats"] = stats
            data["metrics"] = stats
            data["recent_calls"] = list(calls_qs.order_by("-created_at")[:5].values(
                "id", "servy_id", "complaint_type", "status", "priority", "created_at"
            ))
            recent_interactions_qs = interactions_qs.select_related(
                "customer", "asset"
            ).order_by("-created_at")[:5]
            data["recent_interactions"] = [
                {
                    "id": ix.id,
                    "question": ix.question,
                    "resolved": ix.resolved,
                    "created_at": ix.created_at,
                    "escalated_call_id": ix.escalated_call_id,
                    "customer__name": ix.customer.name if ix.customer else "",
                    "asset__name": ix.asset.name if ix.asset else "",
                    "escalated_call": ix.escalated_call_id is not None,
                    "resolved_without_call": ix.resolved and ix.escalated_call_id is None,
                }
                for ix in recent_interactions_qs
            ]

        else:
            # Staff / Technician / Admin / Superuser
            calls_qs = ServiceCall.objects.filter(tenant=tenant)
            interactions_qs = CustomerInteraction.objects.filter(tenant=tenant)
            staff = ctx.get("staff_profile")
            my_calls = calls_qs.filter(technician=staff) if staff else calls_qs.none()

            stats = {
                "total_calls": calls_qs.count(),
                "service_calls": calls_qs.count(),
                "open_calls": calls_qs.filter(status__in=["open", "assigned", "in_progress"]).count(),
                "assigned_to_me": my_calls.filter(status__in=["assigned", "in_progress"]).count(),
                "kb_documents": KnowledgeDocument.objects.filter(tenant=tenant, is_rag_enabled=True).count(),
                "knowledge_documents": KnowledgeDocument.objects.filter(tenant=tenant, is_rag_enabled=True).count(),
                "assets": Asset.objects.filter(tenant=tenant, status="active").count(),
                "inventory_items": InventoryItem.objects.filter(tenant=tenant).count(),
                "part_requests": PartRequest.objects.filter(tenant=tenant).count(),
                "interactions_today": interactions_qs.filter(created_at__date=timezone.now().date()).count(),
                "escalations_today": interactions_qs.filter(created_at__date=timezone.now().date(), escalated_call__isnull=False).count(),
            }
            data["stats"] = stats
            data["metrics"] = stats
            data["recent_calls"] = list(calls_qs.order_by("-created_at")[:8].values(
                "id", "servy_id", "complaint_type", "status", "priority", "created_at",
                "customer__name", "asset__name"
            ))
            recent_interactions_qs = interactions_qs.select_related(
                "customer", "asset"
            ).order_by("-created_at")[:5]
            data["recent_interactions"] = [
                {
                    "id": ix.id,
                    "question": ix.question,
                    "resolved": ix.resolved,
                    "created_at": ix.created_at,
                    "escalated_call_id": ix.escalated_call_id,
                    "customer__name": ix.customer.name if ix.customer else "",
                    "asset__name": ix.asset.name if ix.asset else "",
                    "escalated_call": ix.escalated_call_id is not None,
                    "resolved_without_call": ix.resolved and ix.escalated_call_id is None,
                }
                for ix in recent_interactions_qs
            ]

        return Response(data)


# ----------------------------------------------------------------------
# Workflow 1: Customer AI Support (Pre-ticket)
# ----------------------------------------------------------------------
class CustomerSupportContextView(APIView):
    """Fetch sites and assets for pre-ticket self-service."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Customer AI Support is not available for store roles."}, status=status.HTTP_403_FORBIDDEN)

        # Staff can pass ?customer_id= to preview or assist
        if not customer and role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            cust_id = request.query_params.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            else:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer profile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Enforce server-side real customer data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=customer)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        sites = list(Site.objects.filter(tenant=tenant, customer=customer).values("id", "name", "city"))
        assets_qs = Asset.objects.filter(tenant=tenant, customer=customer, status="active").select_related("product")
        if get_customer_data_mode() == "verified_real_only":
            assets = [
                {
                    "id": a.id,
                    "name": a.name,
                    "asset_code": a.asset_code,
                    "site_id": a.site_id,
                    "model_number": a.model_number,
                    "product__name": a.product.name if a.product else None,
                }
                for a in assets_qs
                if not is_synthetic_asset(a)
            ]
        else:
            assets = list(assets_qs.values(
                "id", "name", "asset_code", "site_id", "model_number", "product__name"
            ))

        # Check for active resumable diagnostic session
        active_session_data = None
        asset_id_param = request.query_params.get("asset_id")
        session_qs = DiagnosticSession.objects.filter(
            tenant=tenant,
            customer=customer,
            status__in=["ACTIVE", "WAITING_INPUT", "WAITING_VERIFY", "NO_PLAYBOOK_AVAILABLE"],
        ).order_by("-updated_at")

        if asset_id_param:
            session_qs = session_qs.filter(asset_id=asset_id_param)

        active_session = session_qs.first()
        if active_session:
            if get_customer_data_mode() == "verified_real_only" and is_synthetic_asset(active_session.asset):
                active_session_data = None
            else:
                latest_interaction = CustomerInteraction.objects.filter(
                    tenant=tenant,
                    customer=customer,
                    asset=active_session.asset,
                ).order_by("-created_at").first()

                contract_data = None
                if latest_interaction:
                    contract_data = build_structured_diagnostic_contract(
                        interaction=latest_interaction,
                        session=active_session,
                        raw_answer=latest_interaction.answer or "",
                        retrieved_references=latest_interaction.source_refs or [],
                        is_hazard=("IMMEDIATE SAFETY RISK" in (latest_interaction.answer or "")),
                        hazard_warning=(latest_interaction.answer if "IMMEDIATE SAFETY RISK" in (latest_interaction.answer or "") else ""),
                    )

                active_session_data = {
                    "session_id": str(active_session.session_id),
                    "asset_id": active_session.asset_id,
                    "status": active_session.status,
                    "version": active_session.version,
                    "complaint": active_session.complaint,
                    "contract": contract_data,
                }

        return Response({
            "customer": {"id": customer.id, "name": customer.name},
            "sites": sites,
            "assets": assets,
            "active_session": active_session_data,
        })


def _clean_input(val, max_len=1000):
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    return val.strip()[:max_len]


class CustomerSupportQueryView(APIView):
    """Execute customer RAG query before ticket creation with multi-input hazard evaluation."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [CustomerRAGThrottle]

    def post(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Customer AI Support is not available for store roles."}, status=status.HTTP_403_FORBIDDEN)

        if not customer and role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            cust_id = request.data.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            if not customer:
                target_site_id = request.data.get("site_id")
                if target_site_id:
                    matched_site = Site.objects.filter(tenant=tenant, id=target_site_id).select_related("customer").first()
                    if matched_site and matched_site.customer:
                        customer = matched_site.customer
            if not customer:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer context required."}, status=status.HTTP_400_BAD_REQUEST)

        # Enforce server-side real customer data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=customer)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        site_id = request.data.get("site_id")
        asset_id = request.data.get("asset_id")
        question = _clean_input(request.data.get("question") or request.data.get("query"))
        clarification_response = _clean_input(request.data.get("clarification_response"))
        user_observation = _clean_input(request.data.get("observation"))

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

        # Enforce asset-level real data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=customer, asset=asset)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        conversation_uuid = request.data.get("conversation_uuid")
        session_id = request.data.get("session_id")
        idempotency_key = request.headers.get("X-Idempotency-Key") or request.data.get("idempotency_key")

        try:
            from core.services.customer_conversation import process_customer_diagnostic_turn
            turn_result = process_customer_diagnostic_turn(
                tenant=tenant,
                customer=customer,
                user=request.user,
                site=site,
                asset=asset,
                question=question,
                clarification_response=clarification_response,
                user_observation=user_observation,
                conversation_uuid=conversation_uuid,
                session_id=session_id,
                idempotency_key=idempotency_key,
            )
            return Response(turn_result["contract"])
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)
        except ValidationError as ve:
            detail_msg = ve.message if hasattr(ve, "message") else str(ve)
            return Response({"detail": detail_msg}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.exception("Error processing customer diagnostic query: %s", exc)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class CustomerSupportResolveView(APIView):
    """Mark customer interaction as resolved with clear distinction between customer-reported and verified repair."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            return Response(
                {"detail": "Customer AI Support is not available for store roles."},
                status=status.HTTP_403_FORBIDDEN,
            )
        tenant = ctx["tenant"]

        try:
            interaction = CustomerInteraction.objects.get(tenant=tenant, id=pk)
        except CustomerInteraction.DoesNotExist:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] in CUSTOMER_ROLES and interaction.customer_id != ctx["customer"].id:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        # Enforce server-side real customer data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=interaction.customer, asset=interaction.asset)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        # Safety Guard: Check if interaction or session has an active hazard or escalation
        is_hazard, _ = check_for_safety_hazard(interaction.question or "")
        if is_hazard or "IMMEDIATE SAFETY RISK" in (interaction.answer or ""):
            return Response(
                {"detail": "Cannot resolve an interaction with an active safety hazard. Certified technician dispatch required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if interaction.escalated_call_id:
            return Response(
                {"detail": "Cannot resolve an interaction that has already been escalated to service call."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        session = None
        if interaction.asset_id:
            session = (
                DiagnosticSession.objects.filter(
                    tenant=tenant,
                    customer=interaction.customer,
                    asset=interaction.asset,
                )
                .exclude(status="RESOLVED")
                .order_by("-updated_at")
                .first()
            )

            if session:
                facts = session.facts_snapshot or {}
                if facts.get("hazard") or session.status in ("ESCALATED", "ABANDONED"):
                    return Response(
                        {"detail": f"Cannot resolve session with safety hazard or in terminal status '{session.status}'. Certified technician service required."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

        resolve_customer_session(
            tenant=tenant,
            customer=interaction.customer,
            session=session,
            interaction=interaction,
            verified=False,
        )

        if interaction.conversation:
            interaction.conversation.status = "RESOLVED"
            interaction.conversation.closed_at = timezone.now()
            interaction.conversation.save(update_fields=["status", "closed_at", "updated_at"])
            CustomerMessage.objects.create(
                tenant=tenant,
                conversation=interaction.conversation,
                sender_role="system",
                message_text="Issue marked as resolved by customer.",
            )

        return Response({
            "status": "resolved",
            "interaction_id": interaction.id,
            "session_id": str(session.session_id) if session else None,
            "resolution_type": "customer_reported",
            "detail": "Marked as resolved by customer. Thank you!",
        })


class CustomerSupportEscalateView(APIView):
    """Escalate unresolved customer interaction to exactly one Service Call."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            return Response(
                {"detail": "Customer AI Support is not available for store roles."},
                status=status.HTTP_403_FORBIDDEN,
            )
        tenant = ctx["tenant"]

        try:
            interaction = CustomerInteraction.objects.get(tenant=tenant, id=pk)
        except CustomerInteraction.DoesNotExist:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        if ctx["role"] in CUSTOMER_ROLES and interaction.customer_id != ctx["customer"].id:
            return Response({"detail": "Interaction not found."}, status=status.HTTP_404_NOT_FOUND)

        # Enforce server-side real customer data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=interaction.customer, asset=interaction.asset)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        note = _clean_input(request.data.get("note") or request.data.get("customer_note"))
        session_id = _clean_input(request.data.get("session_id"))

        # Idempotent ticket creation with OneToOne constraint enforcement and customer observation preservation
        call = create_call_from_interaction(interaction, customer_note=note, session_id=session_id)

        if interaction.conversation:
            interaction.conversation.status = "ESCALATED"
            interaction.conversation.linked_service_call = call
            interaction.conversation.save(update_fields=["status", "linked_service_call", "updated_at"])
            CustomerMessage.objects.create(
                tenant=tenant,
                conversation=interaction.conversation,
                sender_role="system",
                message_text=f"Service Call #{call.servy_id} created successfully.",
                structured_contract={"call_id": call.id, "servy_id": call.servy_id},
            )

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
# Workflow 2: Engineer Copilot (Post-ticket)
# ----------------------------------------------------------------------
class EngineerCopilotQueryView(APIView):
    """Technician Copilot scoped to an existing Service Call."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [EngineerRAGThrottle]

    def post(self, request):
        ctx = require_api_context(request)
        role = ctx["role"]
        if role not in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Engineer Copilot is restricted to service staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        call_id = request.data.get("call_id") or request.data.get("service_call_id")
        question = (request.data.get("question") or request.data.get("query") or "").strip()

        if not call_id or not question:
            return Response({"detail": "Service call ID and question are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            call = ServiceCall.objects.select_related("asset", "customer", "site").get(tenant=tenant, id=call_id)
        except ServiceCall.DoesNotExist:
            return Response({"detail": "Service call not found in this tenant."}, status=status.HTTP_404_NOT_FOUND)

        # Respect can_view_confidential permission flag
        include_confidential = ctx["can_view_confidential"]

        rag_res = ask_rag(
            tenant=tenant,
            question=question,
            asset=call.asset,
            customer=call.customer,
            include_confidential=include_confidential,
            service_call=call,
            staff_mode=True,
        )

        retrieved_list = rag_res.get("retrieved", [])
        retrieval_backend = getattr(retrieved_list, "retrieval_backend", "fallback")

        # Enrich past resolution references with full ServiceCall details
        past_res_call_ids = [
            r.get("service_call_id")
            for r in rag_res.get("references", [])
            if r.get("source_kind") == "past_resolution" and r.get("service_call_id")
        ]
        sc_map = {
            sc.id: sc
            for sc in ServiceCall.objects.filter(tenant=tenant, id__in=past_res_call_ids)
        }
        past_resolutions = []
        for r in rag_res.get("references", []):
            if r.get("source_kind") == "past_resolution":
                sc = sc_map.get(r.get("service_call_id"))
                servy_id = sc.servy_id if sc else r.get("service_call_id")
                complaint_type = sc.complaint_type if sc and sc.complaint_type else (r.get("heading") or "Past Service Call")
                resolution_text = (sc.resolution_text or sc.technician_notes or "Resolved according to standard technical procedures.") if sc else "Resolved according to standard technical procedures."
                past_resolutions.append({
                    "service_call_id": r.get("service_call_id"),
                    "servy_id": servy_id,
                    "complaint_type": complaint_type,
                    "resolution_text": resolution_text,
                    "title": r.get("title"),
                    "reference": r.get("reference"),
                    "doc_type": r.get("doc_type"),
                    "score": r.get("score"),
                })

        recovery_passport = None
        if hasattr(call, "recovery_passport") and call.recovery_passport:
            rp = call.recovery_passport
            recovery_passport = {
                "id": rp.id,
                "structured_data": rp.structured_data,
                "do_not_repeat_items": rp.do_not_repeat_items,
                "evidence_completeness": rp.evidence_completeness,
                "created_at": rp.created_at,
            }

        return Response({
            "answer": rag_res["answer"],
            "references": rag_res["references"],
            "past_resolutions": past_resolutions,
            "engine": rag_res["engine"],
            "retrieval_backend": retrieval_backend,
            "recovery_passport": recovery_passport,
        })


# ----------------------------------------------------------------------
# Sites & Assets Endpoints
# ----------------------------------------------------------------------
class SitesAssetsView(APIView):
    """Retrieve assets for a specific site."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, site_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        site_qs = Site.objects.filter(tenant=tenant, id=site_id)
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"detail": "Site not found or access denied."}, status=status.HTTP_404_NOT_FOUND)
            try:
                validate_customer_serving_policy(tenant=tenant, customer=customer)
            except PermissionDenied as pe:
                return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)
            site_qs = site_qs.filter(customer=customer)

        site = site_qs.first()
        if not site:
            return Response({"detail": "Site not found or access denied."}, status=status.HTTP_404_NOT_FOUND)

        asset_filter = {
            "tenant": tenant,
            "site": site,
            "status": "active",
        }
        if role in CUSTOMER_ROLES:
            asset_filter["customer"] = customer
        elif site.customer_id:
            asset_filter["customer"] = site.customer

        assets_qs = Asset.objects.filter(**asset_filter).select_related("product")
        if role in CUSTOMER_ROLES and get_customer_data_mode() == "verified_real_only":
            assets = [
                {
                    "id": a.id,
                    "name": a.name,
                    "asset_code": a.asset_code,
                    "model_number": a.model_number,
                    "product__name": a.product.name if a.product else None,
                }
                for a in assets_qs
                if not is_synthetic_asset(a)
            ]
        else:
            assets = list(assets_qs.values("id", "name", "asset_code", "model_number", "product__name"))

        return Response({"assets": assets})


class AssetsListView(APIView):
    """List assets scoped by tenant and customer."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        if role in CUSTOMER_ROLES and get_customer_data_mode() == "verified_real_only":
            try:
                validate_customer_serving_policy(tenant=tenant, customer=customer)
            except PermissionDenied:
                return Response({"assets": []})

        qs = Asset.objects.filter(tenant=tenant).select_related("customer", "site", "product")
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"assets": []})
            qs = qs.filter(customer=customer)

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(asset_code__icontains=search) | Q(model_number__icontains=search))

        items = list(qs[:100].values(
            "id", "name", "asset_code", "model_number", "serial_number",
            "status", "customer__name", "site__name", "product__name"
        ))
        if role in CUSTOMER_ROLES and get_customer_data_mode() == "verified_real_only":
            items = [a for a in items if not is_synthetic_asset(a)]
        return Response({"assets": items})


class AssetDetailView(APIView):
    """Retrieve full asset details, warranty countdown, and associated operational counts."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = Asset.objects.filter(tenant=tenant).select_related(
            "customer", "site", "product", "product__brand", "product__category", "product__category__domain"
        )
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)
            qs = qs.filter(customer=customer)

        asset = qs.filter(id=pk).first()
        if not asset:
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        # Warranty calculations
        today = timezone.now().date()
        warranty_days_remaining = None
        is_warranty_expired = None
        if asset.warranty_until:
            delta = (asset.warranty_until - today).days
            warranty_days_remaining = max(0, delta)
            is_warranty_expired = delta < 0

        # Associated metrics
        open_calls_count = ServiceCall.objects.filter(
            tenant=tenant, asset=asset, status__in=["open", "assigned", "in_progress"]
        ).count()

        from core.services.knowledge_scope import get_applicable_knowledge_queryset
        doc_count = get_applicable_knowledge_queryset(asset, ctx).count()

        return Response({
            "asset": {
                "id": asset.id,
                "name": asset.name,
                "asset_code": asset.asset_code,
                "model_number": asset.model_number,
                "serial_number": asset.serial_number,
                "status": asset.status,
                "warranty_until": asset.warranty_until,
                "warranty_days_remaining": warranty_days_remaining,
                "is_warranty_expired": is_warranty_expired,
                "customer": {"id": asset.customer.id, "name": asset.customer.name} if asset.customer else None,
                "site": {
                    "id": asset.site.id,
                    "name": asset.site.name,
                    "address": asset.site.address,
                    "city": asset.site.city,
                } if asset.site else None,
                "product": {
                    "id": asset.product.id,
                    "name": asset.product.name,
                    "description": asset.product.description,
                    "brand_name": asset.product.brand.name if asset.product and asset.product.brand else None,
                    "category_name": asset.product.category.name if asset.product and asset.product.category else None,
                    "domain_name": asset.product.category.domain.name if asset.product and asset.product.category and asset.product.category.domain else None,
                } if asset.product else None,
                "open_calls_count": open_calls_count,
                "documents_count": doc_count,
            }
        })


class AssetDocumentsView(APIView):
    """Retrieve all KnowledgeDocuments applicable to an asset (Asset -> Product -> Category -> General)."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = Asset.objects.filter(tenant=tenant).select_related("product")
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)
            qs = qs.filter(customer=customer)

        asset = qs.filter(id=pk).first()
        if not asset:
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        from core.services.knowledge_scope import get_applicable_knowledge_queryset
        doc_qs = get_applicable_knowledge_queryset(asset, ctx)

        results = []
        for d in doc_qs[:200]:
            scope_level = "general"
            if d.asset_id == asset.id:
                scope_level = "asset"
            elif asset.product_id and d.product_id == asset.product_id:
                scope_level = "product"
            elif asset.product and asset.product.category_id and d.category_id == asset.product.category_id:
                scope_level = "category"

            results.append({
                "id": d.id,
                "title": d.title,
                "description": d.description,
                "doc_type": d.doc_type,
                "doc_type_display": d.get_doc_type_display(),
                "version": d.version,
                "is_rag_enabled": d.is_rag_enabled,
                "index_status": d.index_status,
                "index_error": d.index_error,
                "has_file": bool(d.file),
                "original_filename": d.original_filename or (os.path.basename(d.file.name) if d.file else ""),
                "scope_level": scope_level,
                "is_confidential": d.is_confidential,
                "published_at": d.published_at,
                "updated_at": d.updated_at,
            })

        return Response({"documents": results})


class AssetDocumentUploadView(APIView):
    """Upload a document attached directly to an asset.
    Customer uploads are strictly created with is_rag_enabled=False to prevent prompt injection.
    Staff uploads can trigger immediate indexing."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [UploadThrottle]

    def post(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = Asset.objects.filter(tenant=tenant).select_related("customer", "product")
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)
            qs = qs.filter(customer=customer)

        asset = qs.filter(id=pk).first()
        if not asset:
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        title = (request.data.get("title") or "").strip()
        description = (request.data.get("description") or "").strip()
        doc_type = request.data.get("doc_type", "reference")
        content_text = (request.data.get("content_text") or "").strip()
        uploaded_file = request.FILES.get("file")

        if not title:
            return Response({"detail": "Title is required."}, status=status.HTTP_400_BAD_REQUEST)

        if not content_text and not uploaded_file:
            return Response({"detail": "Provide text content or a file attachment."}, status=status.HTTP_400_BAD_REQUEST)

        # File validation
        if uploaded_file:
            name_lower = uploaded_file.name.lower()
            ext = os.path.splitext(name_lower)[1]
            if ext not in ALLOWED_KB_EXTENSIONS:
                return Response(
                    {"detail": f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_KB_EXTENSIONS))}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if uploaded_file.size > MAX_KB_UPLOAD_BYTES:
                return Response(
                    {"detail": f"File exceeds max size of {MAX_KB_UPLOAD_BYTES // (1024*1024)}MB."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            head = uploaded_file.read(8)
            uploaded_file.seek(0)
            if ext == ".pdf" and not head.startswith(b"%PDF-"):
                return Response({"detail": "File header does not match PDF specification."}, status=status.HTTP_400_BAD_REQUEST)
            if ext == ".docx" and not head.startswith(b"PK"):
                return Response({"detail": "File header does not match DOCX specification."}, status=status.HTTP_400_BAD_REQUEST)
            if ext in {".txt", ".md", ".csv"}:
                sample = uploaded_file.read(4096)
                uploaded_file.seek(0)
                if b"\x00" in sample:
                    return Response({"detail": "Binary data or null bytes detected in text file."}, status=status.HTTP_400_BAD_REQUEST)

        is_customer = role in CUSTOMER_ROLES
        is_rag_enabled = False if is_customer else str(request.data.get("is_rag_enabled", "true")).lower() in {"1", "true", "yes"}

        doc = None
        chunks_indexed = 0
        try:
            with transaction.atomic():
                doc = KnowledgeDocument.objects.create(
                    tenant=tenant,
                    customer=asset.customer,
                    asset=asset,
                    product=asset.product,
                    title=title,
                    description=description,
                    doc_type=doc_type,
                    source_type="file" if uploaded_file else "text",
                    is_confidential=False,
                    is_rag_enabled=is_rag_enabled,
                    content_text=content_text,
                    file=uploaded_file,
                    original_filename=uploaded_file.name if uploaded_file else "",
                    index_status="INDEXING" if is_rag_enabled else "NOT_INDEXED",
                )
                if is_rag_enabled and (content_text or uploaded_file):
                    chunks_indexed = index_document(doc)
        except UnsafeDocumentError as exc:
            if doc and doc.file:
                try:
                    doc.file.delete(save=False)
                except Exception:
                    pass
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "id": doc.id,
            "title": doc.title,
            "chunks_count": chunks_indexed,
            "index_status": doc.index_status,
            "is_rag_enabled": doc.is_rag_enabled,
            "message": "Document uploaded successfully. It is pending staff approval before AI RAG ingestion." if is_customer else f"Document indexed ({chunks_indexed} chunks)."
        }, status=status.HTTP_201_CREATED)


class AssetCallsView(APIView):
    """Retrieve service calls scoped to a single asset, with open vs history filtering.
    Customer callers receive sanitized payloads omitting internal technician_notes."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = Asset.objects.filter(tenant=tenant)
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)
            qs = qs.filter(customer=customer)

        asset = qs.filter(id=pk).first()
        if not asset:
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        calls_qs = ServiceCall.objects.filter(tenant=tenant, asset=asset).select_related(
            "customer", "site", "technician"
        )
        if role in CUSTOMER_ROLES:
            calls_qs = calls_qs.filter(customer=customer)

        scope_param = request.query_params.get("scope", "all").lower()
        if scope_param == "open":
            calls_qs = calls_qs.filter(status__in=["open", "assigned", "in_progress"])
        elif scope_param == "history":
            calls_qs = calls_qs.filter(status__in=["marked_for_closure", "closed"])

        is_customer = role in CUSTOMER_ROLES
        items = []
        for c in calls_qs.order_by("-created_at")[:100]:
            items.append({
                "id": c.id,
                "servy_id": c.servy_id,
                "complaint_type": c.complaint_type,
                "complaint_text": c.complaint_text,
                "status": c.status,
                "priority": c.priority,
                "contact_name": c.contact_name,
                "created_at": c.created_at,
                "closed_at": c.closed_at,
                "resolution_text": c.resolution_text,
                "technician_notes": "" if is_customer else c.technician_notes,
                "technician_name": c.technician.full_name if (c.technician and not is_customer) else None,
                "site_name": c.site.name if c.site else None,
            })

        return Response({"calls": items})


class KnowledgeApproveRagView(APIView):
    """Idempotently approve a document for AI RAG ingestion. Restricted to Staff/Admins."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Permission denied. Only administrators and managers can approve documents for RAG."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        try:
            doc = KnowledgeDocument.objects.get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        if doc.is_rag_enabled and doc.index_status == "INDEXED":
            return Response({
                "id": doc.id,
                "is_rag_enabled": True,
                "index_status": doc.index_status,
                "message": "Document is already approved and indexed for RAG."
            })

        try:
            doc.is_rag_enabled = True
            doc.save(update_fields=["is_rag_enabled"])
            chunks_indexed = index_document(doc)
        except Exception as exc:
            doc.is_rag_enabled = False
            doc.index_status = "FAILED"
            doc.index_error = str(exc)
            doc.save(update_fields=["is_rag_enabled", "index_status", "index_error"])
            return Response({"detail": f"Failed to index document: {str(exc)}"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "id": doc.id,
            "is_rag_enabled": True,
            "index_status": doc.index_status,
            "chunks_count": chunks_indexed,
            "message": f"Document successfully approved and indexed ({chunks_indexed} chunks)."
        })


class KnowledgeRemoveRagView(APIView):
    """Idempotently remove a document from AI RAG. Restricted to Staff/Admins."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Permission denied. Only administrators and managers can remove documents from RAG."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        try:
            doc = KnowledgeDocument.objects.get(tenant=tenant, id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        if not doc.is_rag_enabled and doc.index_status == "NOT_INDEXED":
            return Response({
                "id": doc.id,
                "is_rag_enabled": False,
                "index_status": doc.index_status,
                "message": "Document is already disabled for RAG."
            })

        doc.is_rag_enabled = False
        doc.index_status = "NOT_INDEXED"
        doc.save(update_fields=["is_rag_enabled", "index_status"])
        queue_chroma_deletion(doc.id)

        return Response({
            "id": doc.id,
            "is_rag_enabled": False,
            "index_status": "NOT_INDEXED",
            "message": "Document removed from AI RAG."
        })


# ----------------------------------------------------------------------
# Call Register Endpoints
# ----------------------------------------------------------------------
class CallRegisterView(APIView):
    """List service calls with search, filter, and tenant isolation."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = ServiceCall.objects.filter(tenant=tenant).select_related(
            "customer", "site", "asset", "technician"
        )
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"calls": []})
            qs = qs.filter(customer=customer)

        # Filter: assigned_to_me — scopes to the logged-in technician's StaffProfile
        assigned_to_me = request.query_params.get("assigned_to_me", "").lower()
        if assigned_to_me in ("true", "1"):
            staff_profile = ctx.get("staff_profile")
            if staff_profile:
                qs = qs.filter(technician=staff_profile)
            else:
                # User has no linked StaffProfile — return empty set
                return Response({"calls": []})

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        priority_filter = request.query_params.get("priority")
        if priority_filter:
            qs = qs.filter(priority=priority_filter)

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(
                Q(complaint_text__icontains=search) |
                Q(complaint_type__icontains=search) |
                Q(servy_id__icontains=search) |
                Q(customer__name__icontains=search) |
                Q(technician__full_name__icontains=search)
            )

        items = list(qs.order_by("-created_at")[:100].values(
            "id", "servy_id", "complaint_type", "complaint_text", "status", "priority",
            "contact_name", "created_at", "customer__name", "site__name",
            "asset__name", "technician__full_name"
        ))
        return Response({"calls": items})


class CallDetailView(APIView):
    """Retrieve full service call details with updates and part requests."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]

        try:
            call = ServiceCall.objects.select_related(
                "customer", "site", "asset", "technician", "zone", "sla_policy"
            ).get(tenant=tenant, id=pk)
        except ServiceCall.DoesNotExist:
            return Response({"detail": "Call not found."}, status=status.HTTP_404_NOT_FOUND)

        is_customer = ctx["role"] in CUSTOMER_ROLES
        if is_customer and (not ctx["customer"] or call.customer_id != ctx["customer"].id):
            return Response({"detail": "Call not found."}, status=status.HTTP_404_NOT_FOUND)

        updates_raw = list(call.updates.select_related("author").order_by("created_at").values(
            "id", "status", "note", "created_at", "author__full_name"
        ))
        # Customer callers must NOT see internal notes or staff identity in call updates
        if is_customer:
            updates = [
                {"id": u["id"], "status": u["status"], "note": "", "created_at": u["created_at"], "author__full_name": None}
                for u in updates_raw
            ]
        else:
            updates = updates_raw

        parts = list(call.part_requests.order_by("-date").values(
            "id", "indent_id", "spare_description", "manager_status", "store_status", "date"
        ))

        rp = getattr(call, "recovery_passport", None)
        recovery_passport = None
        if rp:
            recovery_passport = {
                "id": rp.id,
                "structured_data": rp.structured_data,
                "do_not_repeat_items": rp.do_not_repeat_items,
                "evidence_completeness": rp.evidence_completeness,
                "created_at": rp.created_at,
            }

        return Response({
            "call": {
                "id": call.id,
                "servy_id": call.servy_id,
                "complaint_type": call.complaint_type,
                "complaint_text": call.complaint_text,
                "technician_notes": "" if is_customer else call.technician_notes,
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
                "technician": ({"id": call.technician.id, "name": call.technician.full_name} if call.technician else None) if not is_customer else None,
            },
            "updates": updates,
            "part_requests": parts if not is_customer else [],
            "recovery_passport": recovery_passport,
        })


class CallExportView(APIView):
    """Export service calls to sanitized Excel format."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Export is restricted to administrators and managers."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        return export_calls_xlsx(tenant)


class CallImportView(APIView):
    """Import service calls from Excel with safety bounds."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Import is restricted to administrators and managers."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
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
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            return Response({"detail": "Knowledge base access is not permitted for store roles."}, status=status.HTTP_403_FORBIDDEN)

        qs = authorized_knowledge_queryset(ctx).select_related("product", "asset", "customer").annotate(chunks_count=Count("chunks"))

        search = request.query_params.get("q", "").strip()
        if search:
            qs = qs.filter(Q(title__icontains=search) | Q(tags__icontains=search) | Q(description__icontains=search))

        doc_type = request.query_params.get("doc_type", "").strip()
        if doc_type:
            qs = qs.filter(doc_type=doc_type)

        source_type = request.query_params.get("source_type", "").strip()
        if source_type:
            qs = qs.filter(source_type=source_type)

        product_id = request.query_params.get("product_id")
        if product_id:
            qs = qs.filter(product_id=product_id)

        asset_id = request.query_params.get("asset_id")
        if asset_id:
            qs = qs.filter(asset_id=asset_id)

        index_status = request.query_params.get("index_status", "").strip()
        if index_status:
            qs = qs.filter(index_status=index_status)

        rag_enabled = request.query_params.get("is_rag_enabled")
        if rag_enabled is not None and rag_enabled != "":
            qs = qs.filter(is_rag_enabled=rag_enabled.lower() in ("true", "1"))

        # Confidential filter only for authorized staff
        if ctx.get("can_view_confidential"):
            confidential_param = request.query_params.get("is_confidential")
            if confidential_param is not None and confidential_param != "":
                qs = qs.filter(is_confidential=confidential_param.lower() in ("true", "1"))

        total_count = qs.count()

        items = list(qs[:100].values(
            "id", "title", "description", "doc_type", "source_type", "source_url", "tags",
            "file", "is_confidential", "disable_sharing", "is_rag_enabled",
            "index_status", "index_version", "updated_at", "product__name", "asset__name",
            "customer__name", "chunks_count"
        ))
        for item in items:
            item["has_file"] = bool(item.pop("file", None))

        return Response({
            "count": total_count,
            "documents": items,
        })


class KnowledgeDetailView(APIView):
    """Retrieve details of a knowledge document."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            return Response({"detail": "Knowledge base access is not permitted for store roles."}, status=status.HTTP_403_FORBIDDEN)

        qs = authorized_knowledge_queryset(ctx)
        try:
            doc = qs.select_related("product", "asset", "customer").get(id=pk)
        except KnowledgeDocument.DoesNotExist:
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
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        qs = authorized_knowledge_queryset(ctx)
        try:
            doc = qs.get(id=pk)
        except KnowledgeDocument.DoesNotExist:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            "id": doc.id,
            "status": doc.index_status,
            "version": doc.index_version,
            "indexed_at": doc.indexed_at,
            "error": doc.index_error,
        })


class KnowledgeMetadataOptionsView(APIView):
    """Return available metadata taxonomy options for KB upload and classification."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Access denied."}, status=status.HTTP_403_FORBIDDEN)

        customers = list(Customer.objects.filter(tenant=tenant).values("id", "name"))
        domains = list(ProductDomain.objects.filter(tenant=tenant).values("id", "name"))
        categories = list(ProductCategory.objects.filter(tenant=tenant).values("id", "name", "domain_id"))
        brands = list(Brand.objects.filter(tenant=tenant).values("id", "name"))
        products = list(Product.objects.filter(tenant=tenant).values("id", "name", "category_id", "brand_id"))
        assets = list(Asset.objects.filter(tenant=tenant, status="active").values("id", "name", "asset_code", "model_number", "product_id", "customer_id"))

        doc_types = [{"key": k, "label": str(v)} for k, v in KnowledgeDocument.DOC_TYPES]
        source_types = [{"key": k, "label": str(v)} for k, v in KnowledgeDocument.SOURCE_TYPES]

        return Response({
            "customers": customers,
            "domains": domains,
            "categories": categories,
            "brands": brands,
            "products": products,
            "assets": assets,
            "doc_types": doc_types,
            "source_types": source_types,
        })


class KnowledgeUploadView(APIView):
    """Upload and index Knowledge Base documents with generic content & RAG classification."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [UploadThrottle]

    def post(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "KB uploads are restricted to managers and administrators."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        title = (request.data.get("title") or "").strip()
        description = (request.data.get("description") or "").strip()
        doc_type = request.data.get("doc_type", "reference")
        source_type = (request.data.get("source_type") or ("file" if request.FILES.get("file") else "text")).strip().lower()
        source_url = (request.data.get("source_url") or "").strip()
        tags = (request.data.get("tags") or "").strip()
        is_confidential = str(request.data.get("is_confidential", "")).lower() in {"1", "true", "yes"}
        disable_sharing = str(request.data.get("disable_sharing", "")).lower() in {"1", "true", "yes"}
        rag_enabled_raw = request.data.get("is_rag_enabled")
        is_rag_enabled = True if rag_enabled_raw is None or rag_enabled_raw == "" else str(rag_enabled_raw).lower() in {"1", "true", "yes"}

        customer_id = request.data.get("customer_id")
        domain_id = request.data.get("domain_id")
        category_id = request.data.get("category_id")
        product_id = request.data.get("product_id")
        asset_id = request.data.get("asset_id")

        customer = Customer.objects.filter(tenant=tenant, id=customer_id).first() if customer_id else None
        domain = ProductDomain.objects.filter(tenant=tenant, id=domain_id).first() if domain_id else None
        category = ProductCategory.objects.filter(tenant=tenant, id=category_id).first() if category_id else None
        product = Product.objects.filter(tenant=tenant, id=product_id).first() if product_id else None
        asset = Asset.objects.filter(tenant=tenant, id=asset_id).first() if asset_id else None

        content_text = (request.data.get("content_text") or "").strip()
        uploaded_file = request.FILES.get("file")

        if not title:
            return Response({"detail": "Title is required."}, status=status.HTTP_400_BAD_REQUEST)

        if not content_text and not uploaded_file and not source_url:
            return Response({"detail": "Provide text content, an attachment file, or a source hyperlink."}, status=status.HTTP_400_BAD_REQUEST)

        # File validation
        if uploaded_file:
            name_lower = uploaded_file.name.lower()
            ext = os.path.splitext(name_lower)[1]
            if ext not in ALLOWED_KB_EXTENSIONS:
                return Response(
                    {"detail": f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_KB_EXTENSIONS))}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if uploaded_file.size > MAX_KB_UPLOAD_BYTES:
                return Response(
                    {"detail": f"File exceeds max size of {MAX_KB_UPLOAD_BYTES // (1024*1024)}MB."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            head = uploaded_file.read(8)
            uploaded_file.seek(0)

            if ext == ".pdf" and not head.startswith(b"%PDF-"):
                return Response({"detail": "File header does not match PDF specification."}, status=status.HTTP_400_BAD_REQUEST)
            if ext == ".docx" and not head.startswith(b"PK"):
                return Response({"detail": "File header does not match DOCX specification."}, status=status.HTTP_400_BAD_REQUEST)
            if ext in {".txt", ".md", ".csv"}:
                sample = uploaded_file.read(4096)
                uploaded_file.seek(0)
                if b"\x00" in sample:
                    return Response({"detail": "Binary data or null bytes detected in text file."}, status=status.HTTP_400_BAD_REQUEST)

        doc = None
        chunks_indexed = 0
        try:
            with transaction.atomic():
                doc = KnowledgeDocument.objects.create(
                    tenant=tenant,
                    customer=customer,
                    domain=domain,
                    category=category,
                    product=product,
                    asset=asset,
                    title=title,
                    description=description,
                    doc_type=doc_type,
                    source_type=source_type,
                    source_url=source_url,
                    tags=tags,
                    is_confidential=is_confidential,
                    disable_sharing=disable_sharing,
                    is_rag_enabled=is_rag_enabled,
                    content_text=content_text,
                    file=uploaded_file,
                    original_filename=uploaded_file.name if uploaded_file else "",
                    index_status="INDEXING" if is_rag_enabled else "NOT_INDEXED",
                )
                if is_rag_enabled and (content_text or uploaded_file):
                    chunks_indexed = index_document(doc)
        except UnsafeDocumentError as exc:
            if doc and doc.file:
                try:
                    doc.file.delete(save=False)
                except Exception:
                    pass
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({
            "id": doc.id,
            "title": doc.title,
            "chunks_count": chunks_indexed,
            "index_status": doc.index_status,
            "is_rag_enabled": doc.is_rag_enabled,
            "message": f"Document created ({chunks_indexed} chunks indexed)." if is_rag_enabled else "Document saved to Knowledge Base.",
        }, status=status.HTTP_201_CREATED)


class KnowledgeDownloadView(APIView):
    """Download protected KB attachment."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] in {"store_admin", "store_operator"}:
            raise Http404("Document not found.")

        qs = authorized_knowledge_queryset(ctx)
        try:
            doc = qs.get(id=pk)
        except KnowledgeDocument.DoesNotExist:
            raise Http404("Document not found.")

        if not doc.file:
            return Response({"detail": "No file attached to this document."}, status=status.HTTP_404_NOT_FOUND)

        return FileResponse(doc.file.open("rb"), as_attachment=True, filename=doc.original_filename or doc.file.name)


class KnowledgeDeleteView(APIView):
    """Delete KB document from relational database (MySQL) and purge vector index from Chroma."""
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, pk):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Deletion is restricted to administrators and managers."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
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
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        role = ctx["role"]
        customer = ctx["customer"]

        qs = Project.objects.filter(tenant=tenant).select_related("customer", "site")
        if role in CUSTOMER_ROLES:
            if not customer:
                return Response({"projects": []})
            qs = qs.filter(customer=customer)

        items = list(qs[:100].values("id", "code", "name", "status", "customer__name", "site__name"))
        return Response({"projects": items})


class InventoryListView(APIView):
    """Read-only inventory list."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (STAFF_ROLES | {"superuser"}):
            return Response({"detail": "Inventory access is restricted to internal staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        qs = InventoryItem.objects.filter(tenant=tenant).select_related("branch", "brand", "product")

        items = []
        for it in qs[:100]:
            items.append({
                "id": it.id,
                "name": it.spare_name,
                "spare_name": it.spare_name,
                "sku": it.ipn,
                "ipn": it.ipn,
                "category": it.category,
                "brand": it.brand.name if it.brand else "",
                "product": it.product.name if it.product else "",
                "quantity": it.quantity,
                "quantity_on_hand": it.quantity,
                "unit": it.unit,
                "reorder_level": 5,
                "branch__name": it.branch.name if it.branch else "Central Warehouse",
            })

        return Response({"items": items})


class PartRequestsListView(APIView):
    """Read-only part requests list."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (STAFF_ROLES | {"superuser"}):
            return Response({"detail": "Part requests access is restricted to internal staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        qs = PartRequest.objects.filter(tenant=tenant).select_related("service_call", "requester")
        items = list(qs[:100].values(
            "id", "indent_id", "spare_description", "manager_status", "store_status", "date",
            "service_call__servy_id", "requester__full_name"
        ))
        return Response({"part_requests": items})


class OperationsListView(APIView):
    """Read-only zones and operational structure."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (STAFF_ROLES | {"superuser"}):
            return Response({"detail": "Operations access is restricted to internal staff."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        zones = list(OperationalZone.objects.filter(tenant=tenant).select_related("branch").values(
            "id", "name", "description", "branch__name", "is_active"
        ))
        for z in zones:
            z["code"] = z["name"]
        return Response({"zones": zones})


class UsersListView(APIView):
    """Read-only staff directory."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (ADMIN_ROLES | {"superuser"}):
            return Response({"detail": "Access to users directory is restricted to administrators and managers."}, status=status.HTTP_403_FORBIDDEN)

        tenant = ctx["tenant"]
        staff = list(StaffProfile.objects.filter(tenant=tenant).select_related("branch", "zone").values(
            "id", "full_name", "role", "email", "phone", "branch__name", "zone__name", "is_active"
        ))
        return Response({"users": staff})


# ----------------------------------------------------------------------
# External Integrations: Starlly API
# ----------------------------------------------------------------------
class ExternalAssetsListView(APIView):
    """Fetch and view external asset records from Starlly API.

    Adheres strictly to tenant isolation and does NOT overwrite or mutate
    local Servy assets without explicit approval.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        if ctx["role"] not in (STAFF_ROLES | {"superuser"}):
            return Response(
                {"detail": "External asset integration is restricted to internal staff."},
                status=status.HTTP_403_FORBIDDEN,
            )

        tenant = ctx["tenant"]

        # Reject unmapped tenants BEFORE making any external network request
        authorized_org = get_authorized_org_for_tenant(tenant)
        if not authorized_org:
            return Response(
                {
                    "error": "Tenant Access Denied",
                    "detail": f"Tenant '{tenant.slug}' is not authorized for Starlly external integrations.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # Do not allow staff to override external organisation IDs through request parameters
        query_org = request.query_params.get("org_id") or request.query_params.get("organisation_id")
        if query_org is not None and str(query_org).strip() != str(authorized_org).strip():
            return Response(
                {
                    "error": "Forbidden Override",
                    "detail": "Organisational override via request parameters is strictly forbidden.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # Parse query parameters safely
        limit_param = request.query_params.get("limit")
        offset_param = request.query_params.get("offset")
        search_param = request.query_params.get("search") or request.query_params.get("q")

        limit = None
        offset = None
        if limit_param is not None:
            try:
                limit = max(1, min(100, int(limit_param)))
            except (ValueError, TypeError):
                return Response({"detail": "Invalid limit parameter."}, status=status.HTTP_400_BAD_REQUEST)

        if offset_param is not None:
            try:
                offset = max(0, int(offset_param))
            except (ValueError, TypeError):
                return Response({"detail": "Invalid offset parameter."}, status=status.HTTP_400_BAD_REQUEST)

        client = get_starlly_client(tenant=tenant)

        try:
            result = client.fetch_assets(limit=limit, offset=offset, search=search_param)
        except StarllyTenantIsolationError as exc:
            return Response(
                {
                    "error": "Tenant Access Denied",
                    "detail": str(exc),
                },
                status=status.HTTP_403_FORBIDDEN,
            )
        except StarllyConfigurationError as exc:
            return Response(
                {
                    "error": "Configuration Blocker",
                    "detail": str(exc),
                    "is_blocker": True,
                    "is_configured": False,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except StarllyAuthenticationError as exc:
            return Response(
                {
                    "error": "Starlly Authentication Failed",
                    "detail": str(exc),
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except StarllyTimeoutError as exc:
            return Response(
                {
                    "error": "Starlly API Timeout",
                    "detail": str(exc),
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except StarllyConnectionError as exc:
            return Response(
                {
                    "error": "Starlly API Unavailable",
                    "detail": str(exc),
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except StarllyApiError as exc:
            return Response(
                {
                    "error": "Starlly API Error",
                    "detail": str(exc),
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # Cross-reference with existing local tenant assets (READ-ONLY check, zero DB writes)
        raw_assets = result.get("assets", [])
        incoming_codes = [a.get("code") for a in raw_assets if a.get("code")]
        local_assets_by_code = {
            a.asset_code: a.id
            for a in Asset.objects.filter(tenant=tenant, asset_code__in=incoming_codes)
        }

        enriched_assets = []
        for asset in raw_assets:
            code = asset.get("code")
            local_id = local_assets_by_code.get(code)
            enriched_asset = dict(asset)
            enriched_asset["is_synced"] = bool(local_id)
            enriched_asset["local_asset_id"] = local_id
            enriched_asset["tenant_id"] = tenant.id
            enriched_asset["tenant_name"] = tenant.name
            enriched_assets.append(enriched_asset)

        return Response({
            "assets": enriched_assets,
            "count": result.get("count", len(enriched_assets)),
            "total_returned": len(enriched_assets),
            "organisation_id": result.get("organisation_id"),
            "tenant": {
                "id": tenant.id,
                "name": tenant.name,
                "slug": tenant.slug,
            },
            "limit": limit,
            "offset": offset,
            "sync_status": "read_only",
        })


# ----------------------------------------------------------------------
# Customer Interaction Platform: Conversations & Service Tracking
# ----------------------------------------------------------------------

class CustomerConversationsListView(APIView):
    """List or create persistent customer support conversations."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Customer AI Support is not available for store roles."}, status=status.HTTP_403_FORBIDDEN)

        if not customer and role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            cust_id = request.query_params.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            else:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer profile not found."}, status=status.HTTP_404_NOT_FOUND)

        qs = CustomerConversation.objects.filter(tenant=tenant, customer=customer).select_related(
            "site", "asset", "current_diagnostic_session", "linked_service_call"
        ).order_by("-updated_at")

        asset_id = request.query_params.get("asset_id")
        if asset_id:
            qs = qs.filter(asset_id=asset_id)

        site_id = request.query_params.get("site_id")
        if site_id:
            qs = qs.filter(site_id=site_id)

        conv_status = request.query_params.get("status")
        if conv_status:
            qs = qs.filter(status=conv_status.upper())

        try:
            limit = min(int(request.query_params.get("limit", 20)), 50)
            offset = max(int(request.query_params.get("offset", 0)), 0)
        except ValueError:
            limit, offset = 20, 0

        total_count = qs.count()
        paged = qs[offset:offset + limit]

        results = []
        for c in paged:
            results.append({
                "id": c.id,
                "conversation_uuid": str(c.conversation_uuid),
                "title": c.title or (f"Troubleshooting {c.asset.name}" if c.asset else "Support Conversation"),
                "status": c.status,
                "site": {"id": c.site.id, "name": c.site.name} if c.site else None,
                "asset": {
                    "id": c.asset.id,
                    "name": c.asset.name,
                    "asset_code": c.asset.asset_code,
                    "model_number": c.asset.model_number,
                } if c.asset else None,
                "session_id": str(c.current_diagnostic_session.session_id) if c.current_diagnostic_session else None,
                "session_status": c.current_diagnostic_session.status if c.current_diagnostic_session else None,
                "linked_service_call": {
                    "id": c.linked_service_call.id,
                    "servy_id": c.linked_service_call.servy_id,
                    "status": c.linked_service_call.status,
                    "priority": c.linked_service_call.priority,
                } if c.linked_service_call else None,
                "message_count": c.messages.count(),
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "closed_at": c.closed_at,
            })

        return Response({
            "count": total_count,
            "conversations": results,
            "limit": limit,
            "offset": offset,
        })

    def post(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Customer AI Support is not available for store roles."}, status=status.HTTP_403_FORBIDDEN)

        if not customer and role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            cust_id = request.data.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            if not customer:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer context required."}, status=status.HTTP_400_BAD_REQUEST)

        site_id = request.data.get("site_id")
        asset_id = request.data.get("asset_id")
        title = _clean_input(request.data.get("title") or "")
        complaint = _clean_input(request.data.get("complaint") or request.data.get("initial_message") or "")

        site = None
        if site_id:
            try:
                site = Site.objects.get(tenant=tenant, customer=customer, id=site_id)
            except Site.DoesNotExist:
                return Response({"detail": "Site not found for this customer."}, status=status.HTTP_400_BAD_REQUEST)

        asset = None
        if asset_id:
            try:
                asset = Asset.objects.get(tenant=tenant, customer=customer, id=asset_id)
            except Asset.DoesNotExist:
                return Response({"detail": "Asset not found for this customer."}, status=status.HTTP_400_BAD_REQUEST)
            if site and asset.site_id and asset.site_id != site.id:
                return Response({"detail": "Asset does not belong to selected site."}, status=status.HTTP_400_BAD_REQUEST)
            if not site and asset.site:
                site = asset.site

            # P0-5: Enforce asset-level real data isolation
            try:
                validate_customer_serving_policy(tenant=tenant, customer=customer, asset=asset)
            except PermissionDenied as pe:
                return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        session = None
        if asset:
            session = DiagnosticSession.objects.create(
                tenant=tenant,
                customer=customer,
                site=site,
                asset=asset,
                complaint=complaint or "Customer conversation started",
                status="ACTIVE",
                current_node_id="intake",
                version=1,
                created_by=request.user,
            )
            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=session,
                seq_num=1,
                event_type="SESSION_STARTED",
                actor_role="customer",
                payload={"start_node_id": "intake", "asset_id": asset.id, "channel": "conversation"},
            )

        conv = CustomerConversation.objects.create(
            tenant=tenant,
            customer=customer,
            site=site,
            asset=asset,
            created_by=request.user,
            title=title or (f"Inquiry for {asset.name}" if asset else "Support Inquiry"),
            status="ACTIVE",
            current_diagnostic_session=session,
        )

        if complaint:
            CustomerMessage.objects.create(
                tenant=tenant,
                conversation=conv,
                sender_role="customer",
                message_text=complaint,
            )

        return Response({
            "id": conv.id,
            "conversation_uuid": str(conv.conversation_uuid),
            "title": conv.title,
            "status": conv.status,
            "session_id": str(session.session_id) if session else None,
            "site": {"id": site.id, "name": site.name} if site else None,
            "asset": {
                "id": asset.id,
                "name": asset.name,
                "asset_code": asset.asset_code,
                "model_number": asset.model_number,
            } if asset else None,
            "created_at": conv.created_at,
        }, status=status.HTTP_201_CREATED)


class CustomerConversationDetailView(APIView):
    """Fetch complete authorized conversation details with equipment context and linked ticket."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, conversation_uuid):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        try:
            conv = CustomerConversation.objects.select_related(
                "customer", "site", "asset", "current_diagnostic_session", "linked_service_call"
            ).get(tenant=tenant, conversation_uuid=conversation_uuid)
        except CustomerConversation.DoesNotExist:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and conv.customer_id != customer.id:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        messages_qs = conv.messages.all().order_by("created_at")
        messages = [
            {
                "id": m.id,
                "sender_role": m.sender_role,
                "message_text": m.message_text,
                "structured_contract": m.structured_contract,
                "source_refs": m.source_refs,
                "created_at": m.created_at,
            }
            for m in messages_qs
        ]

        ticket_data = None
        if conv.linked_service_call:
            sc = conv.linked_service_call
            ticket_data = {
                "id": sc.id,
                "servy_id": sc.servy_id,
                "status": sc.status,
                "priority": sc.priority,
                "complaint_type": sc.complaint_type,
                "complaint_text": sc.complaint_text,
                "created_at": sc.created_at,
                "response_due_at": sc.response_due_at,
                "resolution_due_at": sc.resolution_due_at,
                "technician_assigned": bool(sc.technician_id),
                "technician_name": sc.technician.full_name if sc.technician else None,
            }

        return Response({
            "id": conv.id,
            "conversation_uuid": str(conv.conversation_uuid),
            "title": conv.title,
            "status": conv.status,
            "customer": {"id": conv.customer.id, "name": conv.customer.name},
            "site": {"id": conv.site.id, "name": conv.site.name} if conv.site else None,
            "asset": {
                "id": conv.asset.id,
                "name": conv.asset.name,
                "asset_code": conv.asset.asset_code,
                "model_number": conv.asset.model_number,
                "status": conv.asset.status,
            } if conv.asset else None,
            "session_id": str(conv.current_diagnostic_session.session_id) if conv.current_diagnostic_session else None,
            "session_status": conv.current_diagnostic_session.status if conv.current_diagnostic_session else None,
            "ticket": ticket_data,
            "messages": messages,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "closed_at": conv.closed_at,
        })


class CustomerConversationMessagesView(APIView):
    """Retrieve immutable messages or post a new customer message within a conversation."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [CustomerRAGThrottle]

    def get(self, request, conversation_uuid):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        try:
            conv = CustomerConversation.objects.get(tenant=tenant, conversation_uuid=conversation_uuid)
        except CustomerConversation.DoesNotExist:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and conv.customer_id != customer.id:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        messages = [
            {
                "id": m.id,
                "sender_role": m.sender_role,
                "message_text": m.message_text,
                "structured_contract": m.structured_contract,
                "source_refs": m.source_refs,
                "created_at": m.created_at,
            }
            for m in conv.messages.all().order_by("created_at")
        ]
        return Response({"messages": messages})

    def post(self, request, conversation_uuid):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        try:
            conv = CustomerConversation.objects.select_related("asset", "site", "customer", "current_diagnostic_session").get(
                tenant=tenant, conversation_uuid=conversation_uuid
            )
        except CustomerConversation.DoesNotExist:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and conv.customer_id != customer.id:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if conv.status in ("RESOLVED", "CLOSED"):
            return Response({"detail": "This conversation is closed. Start a new conversation for new issues."}, status=status.HTTP_400_BAD_REQUEST)

        message_text = _clean_input(request.data.get("message") or request.data.get("text") or request.data.get("question"))
        clarification_response = _clean_input(request.data.get("clarification_response"))
        observation = _clean_input(request.data.get("observation"))

        if not message_text and not clarification_response and not observation:
            return Response({"detail": "Message text is required."}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = request.headers.get("X-Idempotency-Key") or request.data.get("idempotency_key")

        try:
            from core.services.customer_conversation import process_customer_diagnostic_turn
            turn_result = process_customer_diagnostic_turn(
                tenant=tenant,
                customer=conv.customer,
                user=request.user,
                site=conv.site,
                asset=conv.asset,
                question=message_text,
                clarification_response=clarification_response,
                user_observation=observation,
                conversation_uuid=str(conv.conversation_uuid),
                session_id=str(conv.current_diagnostic_session.session_id) if conv.current_diagnostic_session else None,
                idempotency_key=idempotency_key,
            )
            return Response({
                "user_message": turn_result["user_message"],
                "assistant_message": turn_result["assistant_message"],
                "contract": turn_result["contract"],
            })
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)
        except ValidationError as ve:
            detail_msg = ve.message if hasattr(ve, "message") else str(ve)
            return Response({"detail": detail_msg}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.exception("Error processing customer conversation message turn: %s", exc)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class CustomerConversationTicketView(APIView):
    """Fetch or escalate linked ticket for this conversation."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, conversation_uuid):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        try:
            conv = CustomerConversation.objects.select_related("linked_service_call").get(
                tenant=tenant, conversation_uuid=conversation_uuid
            )
        except CustomerConversation.DoesNotExist:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and conv.customer_id != customer.id:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if not conv.linked_service_call:
            return Response({"ticket": None, "detail": "No ticket has been created for this conversation yet."})

        sc = conv.linked_service_call
        return Response({
            "ticket": {
                "id": sc.id,
                "servy_id": sc.servy_id,
                "status": sc.status,
                "priority": sc.priority,
                "complaint_type": sc.complaint_type,
                "complaint_text": sc.complaint_text,
                "created_at": sc.created_at,
                "response_due_at": sc.response_due_at,
                "resolution_due_at": sc.resolution_due_at,
                "technician_assigned": bool(sc.technician_id),
                "technician_name": sc.technician.full_name if sc.technician else None,
            }
        })

    def post(self, request, conversation_uuid):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Customer AI Support is not available for store roles."}, status=status.HTTP_403_FORBIDDEN)

        try:
            conv = CustomerConversation.objects.select_related("linked_service_call", "asset", "site", "customer").get(
                tenant=tenant, conversation_uuid=conversation_uuid
            )
        except CustomerConversation.DoesNotExist:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and conv.customer_id != customer.id:
            return Response({"detail": "Conversation not found."}, status=status.HTTP_404_NOT_FOUND)

        # Enforce server-side real customer data isolation
        try:
            validate_customer_serving_policy(tenant=tenant, customer=conv.customer, asset=conv.asset)
        except PermissionDenied as pe:
            return Response({"detail": str(pe)}, status=status.HTTP_403_FORBIDDEN)

        # If already linked, return existing ticket idempotently
        if conv.linked_service_call:
            sc = conv.linked_service_call
            return Response({
                "ticket": {
                    "id": sc.id,
                    "servy_id": sc.servy_id,
                    "status": sc.status,
                    "priority": sc.priority,
                    "complaint_type": sc.complaint_type,
                    "complaint_text": sc.complaint_text,
                    "created_at": sc.created_at,
                    "technician_assigned": bool(sc.technician_id),
                    "technician_name": sc.technician.full_name if sc.technician else None,
                },
                "detail": f"Service Call #{sc.servy_id} already exists for this conversation."
            })

        # Find latest interaction or create one for escalation
        interaction = CustomerInteraction.objects.filter(conversation=conv).order_by("-id").first()
        if not interaction:
            interaction = CustomerInteraction.objects.create(
                tenant=tenant,
                customer=conv.customer,
                site=conv.site,
                asset=conv.asset,
                conversation=conv,
                diagnostic_session=conv.current_diagnostic_session,
                customer_symptom=conv.title or "Customer requested service from conversation",
                status="active",
            )

        note = _clean_input(request.data.get("user_observation") or request.data.get("note") or request.data.get("customer_note") or "")
        session_id = conv.current_diagnostic_session_id

        call = create_call_from_interaction(interaction, customer_note=note, session_id=session_id)
        conv.status = "TICKETED"
        conv.linked_service_call = call
        conv.save(update_fields=["status", "linked_service_call", "updated_at"])

        CustomerMessage.objects.create(
            tenant=tenant,
            conversation=conv,
            sender_role="system",
            message_text=f"Service Call #{call.servy_id} created successfully.",
            structured_contract={"call_id": call.id, "servy_id": call.servy_id},
        )

        return Response({
            "ticket": {
                "id": call.id,
                "servy_id": call.servy_id,
                "status": call.status,
                "priority": call.priority,
                "complaint_type": call.complaint_type,
                "complaint_text": call.complaint_text,
                "created_at": call.created_at,
                "technician_assigned": bool(call.technician_id),
                "technician_name": call.technician.full_name if call.technician else None,
            },
            "detail": f"Service Call #{call.servy_id} created successfully."
        })


class CustomerTicketsListView(APIView):
    """List service tickets for the authenticated customer."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        if role in {"store_admin", "store_operator"}:
            return Response({"detail": "Access denied."}, status=status.HTTP_403_FORBIDDEN)

        if not customer and role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
            cust_id = request.query_params.get("customer_id")
            if cust_id:
                customer = Customer.objects.filter(tenant=tenant, id=cust_id).first()
            else:
                customer = Customer.objects.filter(tenant=tenant).first()

        if not customer:
            return Response({"detail": "Customer context required."}, status=status.HTTP_400_BAD_REQUEST)

        qs = ServiceCall.objects.filter(tenant=tenant, customer=customer).select_related(
            "asset", "site", "technician"
        ).order_by("-created_at")

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter.lower())

        asset_id = request.query_params.get("asset_id")
        if asset_id:
            qs = qs.filter(asset_id=asset_id)

        try:
            limit = min(int(request.query_params.get("limit", 20)), 50)
            offset = max(int(request.query_params.get("offset", 0)), 0)
        except ValueError:
            limit, offset = 20, 0

        total_count = qs.count()
        calls = qs[offset:offset + limit]

        results = [
            {
                "id": c.id,
                "servy_id": c.servy_id,
                "status": c.status,
                "priority": c.priority,
                "complaint_type": c.complaint_type,
                "complaint_text": c.complaint_text,
                "created_at": c.created_at,
                "asset": {"id": c.asset.id, "name": c.asset.name, "asset_code": c.asset.asset_code} if c.asset else None,
                "site": {"id": c.site.id, "name": c.site.name} if c.site else None,
                "technician_assigned": bool(c.technician_id),
                "technician_name": c.technician.full_name if c.technician else None,
            }
            for c in calls
        ]

        return Response({
            "count": total_count,
            "tickets": results,
            "limit": limit,
            "offset": offset,
        })


class CustomerTicketDetailView(APIView):
    """Retrieve details of a customer's service ticket without leaking internal staff notes."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        role = ctx["role"]

        try:
            call = ServiceCall.objects.select_related("asset", "site", "technician", "customer").get(
                tenant=tenant, id=pk
            )
        except ServiceCall.DoesNotExist:
            return Response({"detail": "Ticket not found."}, status=status.HTTP_404_NOT_FOUND)

        if role in CUSTOMER_ROLES and call.customer_id != customer.id:
            return Response({"detail": "Ticket not found."}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            "id": call.id,
            "servy_id": call.servy_id,
            "status": call.status,
            "priority": call.priority,
            "complaint_type": call.complaint_type,
            "complaint_text": call.complaint_text,
            "created_at": call.created_at,
            "response_due_at": call.response_due_at,
            "resolution_due_at": call.resolution_due_at,
            "customer": {"id": call.customer.id, "name": call.customer.name},
            "asset": {
                "id": call.asset.id,
                "name": call.asset.name,
                "asset_code": call.asset.asset_code,
                "model_number": call.asset.model_number,
            } if call.asset else None,
            "site": {"id": call.site.id, "name": call.site.name} if call.site else None,
            "technician_assigned": bool(call.technician_id),
            "technician_name": call.technician.full_name if call.technician else None,
            "resolution_text": call.resolution_text if call.status in ("resolved", "closed") else None,
        })
