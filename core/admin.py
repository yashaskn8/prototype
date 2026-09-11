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
