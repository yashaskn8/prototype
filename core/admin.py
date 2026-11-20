from django.contrib import admin
from .models import *

for model in [
    Tenant, TenantMembership, Branch, OperationalZone, StaffProfile, Customer, Site,
    ProductDomain, ProductCategory, Brand, Product, Asset, Project, SLAPolicy,
    ChecklistTemplate, KnowledgeDocument, KnowledgeChunk, InventoryItem, ServiceCall,
    CallUpdate, ServiceResolutionIndex, PartRequest, LocalPurchase, ExpenseClaim,
    CustomerInteraction, AuditEvent,
]:
    try:
        admin.site.register(model)
    except admin.sites.AlreadyRegistered:
        pass


@admin.register(AnswerFeedback)
class AnswerFeedbackAdmin(admin.ModelAdmin):
    """Review workflow: read the answer's outcome, mark reviewed, write a de-identified
    question, approve for export. Nothing here changes documents, prompts or models."""
    list_display = ("id", "rating", "reason", "response_mode", "asset", "review_status", "export_approved", "created_at")
    list_filter = ("rating", "reason", "review_status", "export_approved", "response_mode")
    readonly_fields = ("tenant", "customer", "conversation", "assistant_message", "asset", "rating", "reason", "comment",
                       "response_status", "response_mode", "model_versions", "cited_document_versions",
                       "created_at", "updated_at")
    fields = readonly_fields + ("review_status", "reviewer_note", "eval_question", "export_approved", "reviewed_at")
    actions = ["export_regression_cases"]

    @admin.action(description="Export approved negative feedback as regression cases (JSON)")
    def export_regression_cases(self, request, queryset):
        import json
        from django.http import HttpResponse
        from core.services.answer_feedback import regression_cases
        body = json.dumps(regression_cases(queryset), indent=1)
        response = HttpResponse(body, content_type="application/json")
        response["Content-Disposition"] = 'attachment; filename="servy_feedback_regression_cases.json"'
        return response
