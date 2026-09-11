from django.urls import path
from . import api_views

urlpatterns = [
    # Auth & Session
    path("auth/csrf/", api_views.CSRFBootstrapView.as_view(), name="api_auth_csrf"),
    path("auth/login/", api_views.LoginView.as_view(), name="api_auth_login"),
    path("auth/logout/", api_views.LogoutView.as_view(), name="api_auth_logout"),
    path("auth/me/", api_views.CurrentUserView.as_view(), name="api_auth_me"),
    path("auth/switch-tenant/", api_views.SwitchTenantView.as_view(), name="api_auth_switch_tenant"),

    # Dashboard
    path("dashboard/", api_views.DashboardView.as_view(), name="api_dashboard"),

    # Workflow 1: Customer AI Support (Pre-ticket)
    path("customer-support/context/", api_views.CustomerSupportContextView.as_view(), name="api_customer_support_context"),
    path("customer-support/query/", api_views.CustomerSupportQueryView.as_view(), name="api_customer_support_query"),
    path("customer-support/interactions/<int:pk>/resolve/", api_views.CustomerSupportResolveView.as_view(), name="api_customer_support_resolve"),
    path("customer-support/interactions/<int:pk>/escalate/", api_views.CustomerSupportEscalateView.as_view(), name="api_customer_support_escalate"),

    # Workflow 2: Engineer Copilot (Post-ticket)
    path("engineer-copilot/query/", api_views.EngineerCopilotQueryView.as_view(), name="api_engineer_copilot_query"),

    # Sites & Assets
    path("sites/<int:site_id>/assets/", api_views.SitesAssetsView.as_view(), name="api_sites_assets"),
    path("assets/", api_views.AssetsListView.as_view(), name="api_assets"),

    # Call Register & Detail
    path("calls/", api_views.CallRegisterView.as_view(), name="api_calls"),
    path("calls/<int:pk>/", api_views.CallDetailView.as_view(), name="api_call_detail"),
    path("calls/export/", api_views.CallExportView.as_view(), name="api_calls_export"),
    path("calls/import/", api_views.CallImportView.as_view(), name="api_calls_import"),

    # Knowledge Base
    path("knowledge/", api_views.KnowledgeListView.as_view(), name="api_knowledge"),
    path("knowledge/metadata-options/", api_views.KnowledgeMetadataOptionsView.as_view(), name="api_knowledge_metadata_options"),
    path("knowledge/<int:pk>/", api_views.KnowledgeDetailView.as_view(), name="api_knowledge_detail"),
    path("knowledge/<int:pk>/index-status/", api_views.KnowledgeIndexStatusView.as_view(), name="api_knowledge_index_status"),
    path("knowledge/<int:pk>/download/", api_views.KnowledgeDownloadView.as_view(), name="api_knowledge_download"),
    path("knowledge/upload/", api_views.KnowledgeUploadView.as_view(), name="api_knowledge_upload"),
    path("knowledge/<int:pk>/delete/", api_views.KnowledgeDeleteView.as_view(), name="api_knowledge_delete"),

    # Read-Only Supporting Modules
    path("projects/", api_views.ProjectsListView.as_view(), name="api_projects"),
    path("inventory/", api_views.InventoryListView.as_view(), name="api_inventory"),
    path("part-requests/", api_views.PartRequestsListView.as_view(), name="api_part_requests"),
    path("operations/", api_views.OperationsListView.as_view(), name="api_operations"),
    path("users/", api_views.UsersListView.as_view(), name="api_users"),
]
