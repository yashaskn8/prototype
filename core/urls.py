from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("tenant/switch/", views.switch_tenant, name="switch_tenant"),
    path("knowledge/", views.knowledge_base, name="knowledge_base"),
    path("knowledge/new/", views.knowledge_upload, name="knowledge_upload"),
    path("knowledge/<int:pk>/download/", views.knowledge_download, name="knowledge_download"),
    path("knowledge/<int:pk>/reindex/", views.reindex_document, name="reindex_document"),
    path("assets/", views.assets, name="assets"),
    path("inventory/", views.inventory, name="inventory"),
    path("projects/", views.projects, name="projects"),
    path("part-requests/", views.part_requests, name="part_requests"),
    path("operations/", views.operations, name="operations"),
    path("users/", views.users, name="users"),
    path("calls/", views.call_register, name="call_register"),
    path("calls/<int:pk>/", views.call_detail, name="call_detail"),
    path("calls/export/", views.call_export, name="call_export"),
    path("calls/import/", views.call_import, name="call_import"),
    path("ai/assistant/", views.rag_assistant, name="rag_assistant"),
    path("customer-support/", views.customer_portal, name="customer_portal"),
    path("interaction/<int:interaction_id>/resolved/", views.mark_resolved, name="mark_resolved"),
    path("interaction/<int:interaction_id>/escalate/", views.escalate_interaction, name="escalate_interaction"),
]
