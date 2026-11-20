"""URLs for the access-code demo gateways (selected by SERVY_DEMO_GATEWAY).

customer   -> /demo-api/...  (Customer RAG demo, port 8002)
technician -> /tech-api/...  (Technician demo, port 8003)
Each gateway serves only its own prefix. The normal application uses servy_rag.urls.
"""
from django.conf import settings
from django.urls import path
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from core.api_views import (
    CustomerConversationDetailView, CustomerConversationMessagesStreamView, CustomerConversationMessagesView,
    CustomerConversationsListView)
from core.customer_sources import CustomerSourceView
from core.demo_gateway import (
    CustomerAccessCodeAuthentication, CustomerAttachmentView, CustomerConversationTicketGatewayView,
    CustomerInventoryView, TechnicianAccessCodeAuthentication, TechnicianAttachmentView,
    TechnicianTicketDetailGatewayView, bind)
from core.technician_inbox import TechnicianInboxView, TechnicianTicketAcknowledgeView, TechnicianTicketUpdateView


def _health(authentication_class, gateway):
    class Health(APIView):
        authentication_classes = [authentication_class]
        permission_classes = [permissions.IsAuthenticated]

        def get(self, request):
            # Readiness for the single-user prototype: model warm-up state, whether
            # a RAG request is running, and this process's memory (no tenant data).
            from core.ml import preload
            from core.services import rag_admission
            import psutil
            memory = psutil.Process().memory_info()
            models = dict(preload.status)
            return Response({
                "status": "ok", "gateway": gateway, "database": settings.SERVY_DEMO_EXPECTED_DATABASE,
                "models": models,
                "models_ready": all(v in ("ready", "disabled") for v in models.values()),
                "rag_busy": rag_admission.busy(),
                "v5_runtime": __import__("core.ml.reranker", fromlist=["active_runtime"]).active_runtime(),
                "process_memory_mb": {"working_set": round(memory.rss / 2**20),
                                      "private": round(getattr(memory, "private", 0) / 2**20)},
            })
    return Health.as_view()


C = CustomerAccessCodeAuthentication
T = TechnicianAccessCodeAuthentication
customer_patterns = [
    path("demo-api/health/", _health(C, "customer")),
    path("demo-api/inventory/", CustomerInventoryView.as_view()),
    path("demo-api/customer-support/source/", bind(CustomerSourceView, C)),
    path("demo-api/customer-support/conversations/", bind(CustomerConversationsListView, C)),
    path("demo-api/customer-support/conversations/<uuid:conversation_uuid>/", bind(CustomerConversationDetailView, C)),
    path("demo-api/customer-support/conversations/<uuid:conversation_uuid>/messages/",
         bind(CustomerConversationMessagesView, C)),
    path("demo-api/customer-support/conversations/<uuid:conversation_uuid>/messages-stream/",
         bind(CustomerConversationMessagesStreamView, C)),
    path("demo-api/customer-support/conversations/<uuid:conversation_uuid>/ticket/",
         CustomerConversationTicketGatewayView.as_view()),
    path("demo-api/tickets/<int:call_id>/attachments/<int:attachment_id>/", CustomerAttachmentView.as_view()),
]
technician_patterns = [
    path("tech-api/health/", _health(T, "technician")),
    path("tech-api/technician/inbox/", bind(TechnicianInboxView, T)),
    path("tech-api/technician/inbox/<int:pk>/", TechnicianTicketDetailGatewayView.as_view()),
    path("tech-api/technician/inbox/<int:pk>/acknowledge/", bind(TechnicianTicketAcknowledgeView, T)),
    path("tech-api/technician/inbox/<int:pk>/update/", bind(TechnicianTicketUpdateView, T)),
    path("tech-api/technician/inbox/<int:pk>/attachments/<int:attachment_id>/", TechnicianAttachmentView.as_view()),
]

urlpatterns = []   # use servy_rag.demo_customer_urls or servy_rag.demo_technician_urls
