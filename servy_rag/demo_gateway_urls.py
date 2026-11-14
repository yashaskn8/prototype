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
            return Response({"status": "ok", "gateway": gateway, "database": settings.SERVY_DEMO_EXPECTED_DATABASE})
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
