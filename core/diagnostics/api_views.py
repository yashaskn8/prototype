"""
REST API Views for Servy Zero-Repeat Diagnostic Recovery.

All endpoints enforce:
  - Active tenant scoping
  - Customer ownership and IDOR prevention
  - Optimistic concurrency control (expected_version => 409 Conflict)
  - Idempotency via DiagnosticCommand lifecycle (IN_PROGRESS -> COMPLETED / FAILED)
  - Concurrency conflict handling (IN_PROGRESS conflict => HTTP 409)
  - Sanitized customer-facing responses
"""

from django.core.exceptions import ValidationError
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Asset, DiagnosticSession
from core.security import require_api_context

from .orchestrator import (
    ConcurrencyConflictError,
    CurrentNodeMismatchError,
    IdempotencyPayloadConflictError,
    PresentationRequiredError,
    SessionLimitExceededError,
    TerminalSessionMutationError,
    clarify_diagnostic_session,
    complete_idempotency,
    confirm_safe_action,
    escalate_diagnostic_session,
    fail_idempotency,
    get_session_view_data,
    reserve_idempotency_key,
    resolve_contradiction,
    resolve_diagnostic_session,
    start_diagnostic_session,
    submit_diagnostic_answer,
)


class AssetDiagnosticsStartView(APIView):
    """POST /api/assets/{asset_id}/diagnostics/ - Initialize a new diagnostic recovery session."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, asset_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            asset = Asset.objects.select_related("product", "product__category", "site").get(tenant=tenant, id=asset_id)
        except Asset.DoesNotExist:
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or asset.customer_id != customer.id):
            return Response({"detail": "Asset not found."}, status=status.HTTP_404_NOT_FOUND)

        target_customer = customer if is_customer else asset.customer
        if not target_customer:
            return Response({"detail": "Asset is not assigned to a customer."}, status=status.HTTP_400_BAD_REQUEST)

        complaint = request.data.get("complaint", "").strip()
        if not complaint:
            return Response({"detail": "A description of the symptom or complaint is required."}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="START_SESSION",
                    payload={"asset_id": asset.id, "complaint": complaint}
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            session, view_data = start_diagnostic_session(
                tenant=tenant,
                customer=target_customer,
                asset=asset,
                complaint=complaint,
                user=ctx["user"],
                idempotency_key=idempotency_key
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_201_CREATED, view_data)
            return Response(view_data, status=status.HTTP_201_CREATED)

        except SessionLimitExceededError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_429_TOO_MANY_REQUESTS)
            return Response({"detail": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)
        except ValidationError as e:
            msg = e.message if hasattr(e, "message") else str(e)
            fail_idempotency(cmd, msg, response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": msg}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticSessionDetailView(APIView):
    """GET /api/diagnostics/{session_id}/ - Retrieve current diagnostic session state and node."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, session_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook", "escalated_call"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        view_data = get_session_view_data(session, customer)
        return Response(view_data)


class DiagnosticAnswerView(APIView):
    """POST /api/diagnostics/{session_id}/answers/ - Submit observation or verification response."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        node_id = request.data.get("node_id") or session.current_node_id
        value = request.data.get("value")
        expected_version = request.data.get("expected_version")

        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required for optimistic concurrency."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="SUBMIT_ANSWER",
                    payload={"session_id": str(session.session_id), "node_id": node_id, "value": value},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, view_data = submit_diagnostic_answer(
                session=session,
                node_id=node_id,
                value=value,
                expected_version=expected_version,
                customer=customer,
                actor_role="customer" if is_customer else "staff"
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except CurrentNodeMismatchError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "expected_node": e.expected_node, "actual_node": e.actual_node},
                status=status.HTTP_409_CONFLICT
            )
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticActionCompleteView(APIView):
    """POST /api/diagnostics/{session_id}/actions/{node_id}/complete/ - Confirm SAFE_ACTION completed."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id, node_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        expected_version = request.data.get("expected_version")
        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        presentation_token = request.data.get("presentation_token")

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="COMPLETE_ACTION",
                    payload={"session_id": str(session.session_id), "node_id": node_id, "presentation_token": presentation_token},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, view_data = confirm_safe_action(
                session=session,
                node_id=node_id,
                expected_version=expected_version,
                presentation_token=presentation_token,
                customer=customer
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except CurrentNodeMismatchError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "expected_node": e.expected_node, "actual_node": e.actual_node},
                status=status.HTTP_409_CONFLICT
            )
        except PresentationRequiredError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticResolveView(APIView):
    """POST /api/diagnostics/{session_id}/resolve/ - Mark diagnostic incident as recovered."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        expected_version = request.data.get("expected_version")
        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        summary = request.data.get("summary", "Resolved by customer.")

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="RESOLVE_SESSION",
                    payload={"session_id": str(session.session_id), "summary": summary},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, view_data = resolve_diagnostic_session(
                session=session,
                expected_version=expected_version,
                customer=customer,
                summary=summary
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticClarifyView(APIView):
    """POST /api/diagnostics/{session_id}/clarify/ - Clarify symptoms for a NO_PLAYBOOK_AVAILABLE session."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        expected_version = request.data.get("expected_version")
        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        clarification = request.data.get("clarification", "").strip()
        if not clarification:
            return Response({"detail": "Clarification text is required."}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="CLARIFY_SESSION",
                    payload={"session_id": str(session.session_id), "clarification": clarification},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, view_data = clarify_diagnostic_session(
                session=session,
                clarification=clarification,
                expected_version=expected_version,
                customer=customer
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticContradictionResolveView(APIView):
    """POST /api/diagnostics/{session_id}/contradictions/{fact_key}/resolve/ - Resolve fact contradiction."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id, fact_key):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        expected_version = request.data.get("expected_version")
        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        value = request.data.get("value")
        if value is None:
            return Response({"detail": "Field 'value' is required to resolve contradiction."}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = reserve_idempotency_key(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="RESOLVE_CONTRADICTION",
                    payload={"session_id": str(session.session_id), "fact_key": fact_key, "value": value},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, view_data = resolve_contradiction(
                session=session,
                fact_key=fact_key,
                value=value,
                expected_version=expected_version,
                customer=customer
            )
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise


class DiagnosticEscalateView(APIView):
    """POST /api/diagnostics/{session_id}/escalate/ - Escalate to ServiceCall with Recovery Passport."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id):
        ctx = require_api_context(request)
        tenant = ctx["tenant"]
        customer = ctx["customer"]
        is_customer = ctx["role"] == "customer"

        try:
            session = DiagnosticSession.objects.select_related(
                "asset", "customer", "playbook", "escalated_call"
            ).get(tenant=tenant, session_id=session_id)
        except DiagnosticSession.DoesNotExist:
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        if is_customer and (not customer or session.customer_id != customer.id):
            return Response({"detail": "Diagnostic session not found."}, status=status.HTTP_404_NOT_FOUND)

        expected_version = request.data.get("expected_version")
        if expected_version is None:
            return Response({"detail": "Field 'expected_version' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expected_version = int(expected_version)
        except (ValueError, TypeError):
            return Response({"detail": "Invalid 'expected_version' integer."}, status=status.HTTP_400_BAD_REQUEST)

        reason = request.data.get("reason", "Customer requested escalation to engineering service.").strip()[:500]

        idempotency_key = request.headers.get("Idempotency-Key") or request.data.get("idempotency_key", "")
        cmd = None
        if idempotency_key:
            try:
                cmd, is_replay = check_or_record_idempotency(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type="ESCALATE_SESSION",
                    payload={"session_id": str(session.session_id), "reason": reason},
                    session=session
                )
                if is_replay:
                    return Response(cmd.response_payload, status=cmd.response_status)
            except IdempotencyPayloadConflictError as e:
                return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
            except ConcurrencyConflictError as e:
                return Response({"detail": str(e), "current_version": getattr(e, "current_version", 0)}, status=status.HTTP_409_CONFLICT)

        try:
            updated_session, call, passport = escalate_diagnostic_session(
                session=session,
                reason=reason,
                expected_version=expected_version,
                customer=customer
            )
            view_data = get_session_view_data(updated_session, customer)
            if cmd:
                complete_idempotency(cmd, status.HTTP_200_OK, view_data)
            return Response(view_data)

        except ConcurrencyConflictError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_409_CONFLICT)
            return Response(
                {"detail": str(e), "current_version": e.current_version},
                status=status.HTTP_409_CONFLICT
            )
        except TerminalSessionMutationError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_400_BAD_REQUEST)
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            fail_idempotency(cmd, str(e), response_status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise
