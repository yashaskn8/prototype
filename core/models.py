import uuid
from pathlib import Path

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


def knowledge_upload_path(instance, filename):
    suffix = Path(filename).suffix.lower()[:12]
    return f"knowledge/{instance.tenant_id}/{timezone.now():%Y/%m}/{uuid.uuid4().hex}{suffix}"


class Tenant(models.Model):
    name = models.CharField(max_length=160)
    slug = models.SlugField(unique=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class TenantOwnedModel(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)

    class Meta:
        abstract = True

    def _validate_tenant_relations(self):
        """Reject accidental cross-tenant foreign-key assignments.

        View/query filtering is still required for authorization; this is a second
        integrity layer so programmatic writes cannot silently link Tenant A rows
        to Tenant B objects.
        """
        from django.core.exceptions import ValidationError

        if not self.tenant_id:
            return
        errors = {}
        for field in self._meta.fields:
            if field.name == "tenant" or not getattr(field, "many_to_one", False):
                continue
            related_id = getattr(self, field.attname, None)
            if not related_id:
                continue
            related = getattr(self, field.name, None)
            related_tenant_id = getattr(related, "tenant_id", None)
            if related_tenant_id is not None and related_tenant_id != self.tenant_id:
                errors[field.name] = "Related object belongs to a different tenant."
        if errors:
            raise ValidationError(errors)

    def clean(self):
        super().clean()
        self._validate_tenant_relations()

    def save(self, *args, **kwargs):
        # Django does not call Model.clean() automatically on save. Calling it
        # here keeps critical tenant/customer relationship checks active for
        # programmatic writes as well as ModelForms.
        self.clean()
        return super().save(*args, **kwargs)


class Customer(TenantOwnedModel):
    name = models.CharField(max_length=160)
    contact_name = models.CharField(max_length=150, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    email = models.EmailField(blank=True)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class TenantMembership(models.Model):
    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("manager", "Manager"),
        ("technician", "Technician"),
        ("store_admin", "Store Admin"),
        ("store_operator", "Store Operator"),
        ("customer", "Customer"),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tenant_memberships")
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="memberships")
    role = models.CharField(max_length=30, choices=ROLE_CHOICES)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, null=True, blank=True, related_name="portal_memberships")
    can_view_confidential = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ("user", "tenant")

    def clean(self):
        from django.core.exceptions import ValidationError
        if self.role == "customer" and not self.customer_id:
            raise ValidationError("Customer memberships must be linked to a customer.")
        if self.role != "customer" and self.customer_id:
            raise ValidationError("Only customer-role memberships may be linked to a customer record.")
        if self.role == "customer" and self.can_view_confidential:
            raise ValidationError("Customer memberships cannot be granted confidential knowledge access.")
        if self.customer_id and self.customer.tenant_id != self.tenant_id:
            raise ValidationError("Customer and membership tenant must match.")

    def save(self, *args, **kwargs):
        self.clean()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} / {self.tenant.slug} / {self.role}"


class Branch(TenantOwnedModel):
    name = models.CharField(max_length=120)
    city = models.CharField(max_length=120, blank=True)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class OperationalZone(TenantOwnedModel):
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    branch = models.ForeignKey(Branch, on_delete=models.SET_NULL, null=True, blank=True, related_name="zones")
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class StaffProfile(TenantOwnedModel):
    ROLE_CHOICES = [
        ("admin", "Admin"), ("technician", "Technician"),
        ("store_admin", "Store Admin"), ("store_operator", "Store Operator"),
        ("manager", "Manager"),
    ]
    user = models.OneToOneField(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="staff_profile")
    full_name = models.CharField(max_length=150)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=30, blank=True)
    branch = models.ForeignKey(Branch, on_delete=models.SET_NULL, null=True, blank=True)
    zone = models.ForeignKey(OperationalZone, on_delete=models.SET_NULL, null=True, blank=True, related_name="staff")
    role = models.CharField(max_length=30, choices=ROLE_CHOICES, default="technician")
    last_login_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.full_name


class Site(TenantOwnedModel):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="sites")
    zone = models.ForeignKey(OperationalZone, on_delete=models.SET_NULL, null=True, blank=True, related_name="sites")
    name = models.CharField(max_length=150)
    address = models.TextField(blank=True)
    city = models.CharField(max_length=120, blank=True)

    class Meta:
        unique_together = ("tenant", "customer", "name")

    def __str__(self):
        return f"{self.customer.name} / {self.name}"


class ProductDomain(TenantOwnedModel):
    name = models.CharField(max_length=140)
    description = models.TextField(blank=True)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class ProductCategory(TenantOwnedModel):
    domain = models.ForeignKey(ProductDomain, on_delete=models.CASCADE, related_name="categories")
    name = models.CharField(max_length=140)

    class Meta:
        unique_together = ("tenant", "domain", "name")

    def __str__(self):
        return self.name


class Brand(TenantOwnedModel):
    name = models.CharField(max_length=140)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class Product(TenantOwnedModel):
    category = models.ForeignKey(ProductCategory, on_delete=models.SET_NULL, null=True, blank=True, related_name="products")
    brand = models.ForeignKey(Brand, on_delete=models.SET_NULL, null=True, blank=True, related_name="products")
    name = models.CharField(max_length=160)
    description = models.TextField(blank=True)

    class Meta:
        unique_together = ("tenant", "brand", "name")

    def __str__(self):
        return self.name


class Asset(TenantOwnedModel):
    STATUS_CHOICES = [("active", "Active"), ("inactive", "Inactive"), ("retired", "Retired")]
    customer = models.ForeignKey(Customer, on_delete=models.SET_NULL, null=True, blank=True, related_name="assets")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="assets")
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, related_name="assets")
    name = models.CharField(max_length=160)
    asset_code = models.CharField(max_length=60)
    model_number = models.CharField(max_length=100, blank=True)
    serial_number = models.CharField(max_length=100, blank=True)
    warranty_until = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")

    class Meta:
        unique_together = ("tenant", "asset_code")

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.site_id and self.customer_id and self.site.customer_id != self.customer_id:
            errors["site"] = "Site does not belong to the selected customer."
        if errors:
            raise ValidationError(errors)

    @property
    def brand(self):
        return self.product.brand if self.product else None

    def __str__(self):
        return f"{self.name} ({self.asset_code})"


class Project(TenantOwnedModel):
    STATUS_CHOICES = [("active", "Active"), ("completed", "Completed"), ("hold", "On Hold")]
    code = models.CharField(max_length=80)
    name = models.CharField(max_length=180)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="projects")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="projects")
    primary_poc = models.CharField(max_length=150, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    handover_date = models.DateField(null=True, blank=True)
    assets = models.ManyToManyField(Asset, blank=True, related_name="projects")

    class Meta:
        unique_together = ("tenant", "code")

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        if self.site_id and self.customer_id and self.site.customer_id != self.customer_id:
            raise ValidationError({"site": "Project site must belong to the project customer."})

    def __str__(self):
        return f"{self.code} — {self.name}"


class SLAPolicy(TenantOwnedModel):
    name = models.CharField(max_length=140)
    priority = models.CharField(max_length=20, default="normal")
    response_minutes = models.PositiveIntegerField(default=120)
    resolution_minutes = models.PositiveIntegerField(default=720)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ("tenant", "name")

    def __str__(self):
        return self.name


class ChecklistTemplate(TenantOwnedModel):
    name = models.CharField(max_length=160)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, related_name="checklists")
    description = models.TextField(blank=True)
    items = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class KnowledgeDocument(TenantOwnedModel):
    DOC_TYPES = [
        ("installation", "Installation Guide"), ("troubleshooting", "Troubleshooting"),
        ("maintenance", "Maintenance Guide"), ("cleaning", "Cleaning Procedure"),
        ("user_guide", "User Guide"), ("service_manual", "Service Manual"),
        ("faq", "FAQ"), ("reference", "Reference"),
    ]
    SOURCE_TYPES = [("file", "File"), ("text", "Text"), ("link", "Hyperlink"), ("video", "Video")]
    customer = models.ForeignKey(Customer, on_delete=models.SET_NULL, null=True, blank=True, related_name="knowledge_documents")
    domain = models.ForeignKey(ProductDomain, on_delete=models.SET_NULL, null=True, blank=True)
    category = models.ForeignKey(ProductCategory, on_delete=models.SET_NULL, null=True, blank=True)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    asset = models.ForeignKey(Asset, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=220)
    description = models.TextField(blank=True)
    doc_type = models.CharField(max_length=30, choices=DOC_TYPES, default="reference")
    source_type = models.CharField(max_length=20, choices=SOURCE_TYPES, default="text")
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated search tags")
    content_text = models.TextField(blank=True)
    file = models.FileField(upload_to=knowledge_upload_path, blank=True)
    original_filename = models.CharField(max_length=255, blank=True)
    source_url = models.URLField(blank=True)
    version = models.CharField(max_length=40, default="1.0")
    language = models.CharField(max_length=20, default="en")
    checksum_sha256 = models.CharField(max_length=64, blank=True)
    is_confidential = models.BooleanField(default=False)
    disable_sharing = models.BooleanField(default=False)
    is_rag_enabled = models.BooleanField(default=True)
    INDEX_STATUS_CHOICES = [
        ("NOT_INDEXED", "Not Indexed"),
        ("INDEXING", "Indexing"),
        ("INDEXED", "Indexed"),
        ("FAILED", "Failed"),
    ]
    index_status = models.CharField(max_length=20, choices=INDEX_STATUS_CHOICES, default="NOT_INDEXED")
    index_version = models.PositiveIntegerField(default=1)
    indexed_at = models.DateTimeField(null=True, blank=True)
    index_error = models.TextField(blank=True, default="")
    published_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "title"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.asset_id:
            if self.customer_id and self.asset.customer_id and self.asset.customer_id != self.customer_id:
                errors["asset"] = "Asset does not belong to the selected customer."
            if self.product_id and self.asset.product_id and self.asset.product_id != self.product_id:
                errors["asset"] = "Asset does not belong to the selected product."
        if self.product_id and self.category_id and self.product.category_id and self.product.category_id != self.category_id:
            errors["product"] = "Product does not belong to the selected category."
        if self.category_id and self.domain_id and self.category.domain_id != self.domain_id:
            errors["category"] = "Category does not belong to the selected domain."
        if errors:
            raise ValidationError(errors)

    def __str__(self):
        return self.title


class KnowledgeChunk(TenantOwnedModel):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField(default=0)
    heading = models.CharField(max_length=220, blank=True)
    text = models.TextField()
    embedding = models.JSONField(default=dict, blank=True, help_text="Metadata-aware local vector stored as {index:value}")
    content_embedding = models.JSONField(default=dict, blank=True, help_text="Content-only local vector used to verify answer relevance")
    embedding_model = models.CharField(max_length=80, default="hashing-v1")
    safety_flags = models.JSONField(default=list, blank=True)
    is_quarantined = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("document", "chunk_index")
        ordering = ["document_id", "chunk_index"]

    def __str__(self):
        return f"{self.document.title} #{self.chunk_index}"


class InventoryItem(TenantOwnedModel):
    branch = models.ForeignKey(Branch, on_delete=models.SET_NULL, null=True, blank=True)
    spare_name = models.CharField(max_length=180)
    category = models.CharField(max_length=160, blank=True)
    brand = models.ForeignKey(Brand, on_delete=models.SET_NULL, null=True, blank=True)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    ipn = models.CharField(max_length=100, blank=True)
    quantity = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    unit = models.CharField(max_length=40, default="Units")

    def __str__(self):
        return self.spare_name


class ServiceCall(TenantOwnedModel):
    STATUS_CHOICES = [
        ("open", "Open"), ("assigned", "Assigned"), ("in_progress", "In Progress"),
        ("marked_for_closure", "Marked For Closure"), ("closed", "Closed")
    ]
    PRIORITIES = [("low", "Low"), ("normal", "Normal"), ("high", "High"), ("critical", "Critical")]
    servy_id = models.PositiveIntegerField()
    call_type = models.CharField(max_length=80, default="Service")
    complaint_type = models.CharField(max_length=180)
    complaint_text = models.TextField(blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="service_calls")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True)
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, null=True, blank=True)
    asset = models.ForeignKey(Asset, on_delete=models.SET_NULL, null=True, blank=True)
    zone = models.ForeignKey(OperationalZone, on_delete=models.SET_NULL, null=True, blank=True, related_name="service_calls")
    sla_policy = models.ForeignKey(SLAPolicy, on_delete=models.SET_NULL, null=True, blank=True, related_name="service_calls")
    technician = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="service_calls")
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="open")
    priority = models.CharField(max_length=20, choices=PRIORITIES, default="normal")
    contact_name = models.CharField(max_length=150, blank=True)
    contact_phone = models.CharField(max_length=30, blank=True)
    contact_email = models.EmailField(blank=True)
    response_due_at = models.DateTimeField(null=True, blank=True)
    resolution_due_at = models.DateTimeField(null=True, blank=True)
    resolution_text = models.TextField(blank=True)
    technician_notes = models.TextField(blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("tenant", "servy_id")
        ordering = ["-created_at"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.site_id and self.site.customer_id != self.customer_id:
            errors["site"] = "Call site must belong to the selected customer."
        if self.asset_id and self.asset.customer_id and self.asset.customer_id != self.customer_id:
            errors["asset"] = "Call asset must belong to the selected customer."
        if self.project_id and self.project.customer_id != self.customer_id:
            errors["project"] = "Call project must belong to the selected customer."
        if self.zone_id and self.site_id and self.site.zone_id and self.zone_id != self.site.zone_id:
            errors["zone"] = "Call zone does not match the selected site."
        if errors:
            raise ValidationError(errors)

    def __str__(self):
        return f"{self.servy_id} - {self.complaint_type}"


class CallUpdate(TenantOwnedModel):
    service_call = models.ForeignKey(ServiceCall, on_delete=models.CASCADE, related_name="updates")
    author = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=30, blank=True)
    note = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]


class ServiceResolutionIndex(TenantOwnedModel):
    service_call = models.OneToOneField(ServiceCall, on_delete=models.CASCADE, related_name="resolution_index")
    text = models.TextField()
    embedding = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class PartRequest(TenantOwnedModel):
    STATUS_CHOICES = [("pending", "Pending"), ("approved", "Approved"), ("partial", "Partial"), ("out_of_stock", "Out Of Stock"), ("allotted", "Part Allotted"), ("disparity", "Disparity"), ("na", "NA")]
    indent_id = models.PositiveIntegerField()
    date = models.DateTimeField(default=timezone.now)
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="part_requests")
    spare = models.ForeignKey(InventoryItem, on_delete=models.SET_NULL, null=True, blank=True)
    spare_description = models.CharField(max_length=240)
    requester = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="requested_parts")
    approver = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="approved_parts")
    manager_status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="pending")
    store_status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="pending")
    technician_status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="na")

    class Meta:
        unique_together = ("tenant", "indent_id")
        ordering = ["-date"]


class LocalPurchase(TenantOwnedModel):
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="local_purchases")
    requested_by = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True)
    part_description = models.CharField(max_length=240)
    vendor_name = models.CharField(max_length=180, blank=True)
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    status = models.CharField(max_length=30, default="pending")
    receipt_reference = models.CharField(max_length=180, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ExpenseClaim(TenantOwnedModel):
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="expenses")
    claimant = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True)
    category = models.CharField(max_length=80, default="Travel")
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    notes = models.TextField(blank=True)
    status = models.CharField(max_length=30, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)


class ApprovalRequest(TenantOwnedModel):
    STATUS_CHOICES = [("pending", "Pending"), ("approved", "Approved"), ("rejected", "Rejected")]
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="approval_requests")
    module = models.CharField(max_length=80, default="service")
    requested_by = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="approval_requests")
    approver = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="approval_decisions")
    reason = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    decided_at = models.DateTimeField(null=True, blank=True)


class VoucherClaim(TenantOwnedModel):
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="voucher_claims")
    claimant = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="voucher_claims")
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    purpose = models.CharField(max_length=220, blank=True)
    reference = models.CharField(max_length=120, blank=True)
    status = models.CharField(max_length=30, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)


class ReturnRequest(TenantOwnedModel):
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="return_requests")
    inventory_item = models.ForeignKey(InventoryItem, on_delete=models.SET_NULL, null=True, blank=True, related_name="return_requests")
    requested_by = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="return_requests")
    quantity = models.DecimalField(max_digits=12, decimal_places=2, default=1)
    reason = models.CharField(max_length=240, blank=True)
    status = models.CharField(max_length=30, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)


class CorporateIdentity(TenantOwnedModel):
    company_name = models.CharField(max_length=180)
    support_email = models.EmailField(blank=True)
    support_phone = models.CharField(max_length=30, blank=True)
    address = models.TextField(blank=True)
    primary_color = models.CharField(max_length=20, default="#6f42c1")
    logo_text = models.CharField(max_length=120, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Corporate identities"


class BusinessVocabulary(TenantOwnedModel):
    key = models.CharField(max_length=80)
    display_value = models.CharField(max_length=120)
    description = models.CharField(max_length=240, blank=True)

    class Meta:
        unique_together = ("tenant", "key")


class CustomizationSetting(TenantOwnedModel):
    category = models.CharField(max_length=80, default="general")
    key = models.CharField(max_length=120)
    value = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ("tenant", "category", "key")


class EmailNotificationRule(TenantOwnedModel):
    event = models.CharField(max_length=120)
    recipient_role = models.CharField(max_length=40, blank=True)
    subject_template = models.CharField(max_length=220, blank=True)
    is_enabled = models.BooleanField(default=True)

    class Meta:
        unique_together = ("tenant", "event", "recipient_role")


class CustomerInteraction(TenantOwnedModel):
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="servy_interactions")
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="rag_interactions")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True)
    asset = models.ForeignKey(Asset, on_delete=models.SET_NULL, null=True, blank=True)
    service_call = models.ForeignKey(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="rag_interactions")
    question = models.TextField()
    answer = models.TextField(blank=True)
    source_refs = models.JSONField(default=list, blank=True)
    resolved = models.BooleanField(null=True, blank=True)
    escalated_call = models.OneToOneField(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="source_interactions")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.site_id and self.site.customer_id != self.customer_id:
            errors["site"] = "Interaction site does not belong to the selected customer."
        if self.asset_id and self.asset.customer_id and self.asset.customer_id != self.customer_id:
            errors["asset"] = "Interaction asset does not belong to the selected customer."
        if self.service_call_id and self.service_call.customer_id != self.customer_id:
            errors["service_call"] = "Interaction service call belongs to another customer."
        if self.escalated_call_id and self.escalated_call.customer_id != self.customer_id:
            errors["escalated_call"] = "Escalated call belongs to another customer."
        if errors:
            raise ValidationError(errors)

    def __str__(self):
        return f"{self.customer} - {self.question[:60]}"


class AuditEvent(TenantOwnedModel):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="servy_audit_events")
    action = models.CharField(max_length=120)
    object_type = models.CharField(max_length=80, blank=True)
    object_id = models.CharField(max_length=80, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["tenant", "action", "created_at"]),
            models.Index(fields=["tenant", "object_type", "object_id"]),
        ]

    def __str__(self):
        return f"{self.action} {self.object_type}:{self.object_id}".strip()


class DiagnosticPlaybook(TenantOwnedModel):
    STATUS_CHOICES = [
        ("DRAFT", "Draft"),
        ("PUBLISHED", "Published"),
        ("RETIRED", "Retired"),
    ]
    name = models.CharField(max_length=200)
    version = models.PositiveIntegerField(default=1)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="DRAFT")
    domain = models.ForeignKey(ProductDomain, on_delete=models.SET_NULL, null=True, blank=True, related_name="diagnostic_playbooks")
    category = models.ForeignKey(ProductCategory, on_delete=models.SET_NULL, null=True, blank=True, related_name="diagnostic_playbooks")
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, related_name="diagnostic_playbooks")
    applicability_tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated symptom or model tags")
    definition = models.JSONField(default=dict, help_text="Structured graph containing nodes, transitions, and validation rules")
    published_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("tenant", "name", "version")
        ordering = ["-updated_at", "name"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.product_id and self.category_id and self.product.category_id and self.product.category_id != self.category_id:
            errors["product"] = "Product does not belong to the selected category."
        if self.category_id and self.domain_id and self.category.domain_id != self.domain_id:
            errors["category"] = "Category does not belong to the selected domain."
        if errors:
            raise ValidationError(errors)

    def save(self, *args, **kwargs):
        if self.pk:
            old = DiagnosticPlaybook.objects.filter(pk=self.pk).values("status", "definition", "version").first()
            if old and old["status"] == "PUBLISHED":
                update_fields = kwargs.get("update_fields")
                if update_fields is None or "definition" in update_fields:
                    if self.definition != old["definition"]:
                        from django.core.exceptions import ValidationError
                        raise ValidationError("Published DiagnosticPlaybook version is immutable. Create a new version instead.")
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} v{self.version} ({self.status})"


class DiagnosticSession(TenantOwnedModel):
    STATUS_CHOICES = [
        ("ACTIVE", "Active"),
        ("WAITING_INPUT", "Waiting Input"),
        ("WAITING_VERIFY", "Waiting Verify"),
        ("RESOLVED", "Resolved"),
        ("ESCALATED", "Escalated"),
        ("ABANDONED", "Abandoned"),
    ]
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="diagnostic_sessions")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="diagnostic_sessions")
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="diagnostic_sessions")
    playbook = models.ForeignKey(DiagnosticPlaybook, on_delete=models.SET_NULL, null=True, blank=True, related_name="sessions")
    playbook_version = models.PositiveIntegerField(default=1)
    complaint = models.TextField(help_text="Original complaint or symptom reported by customer")
    current_node_id = models.CharField(max_length=100, default="start")
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="ACTIVE")
    version = models.PositiveIntegerField(default=1, help_text="Monotonically increasing version for optimistic concurrency")
    facts_snapshot = models.JSONField(default=dict, blank=True)
    evidence_completeness = models.FloatField(default=0.0)
    escalated_call = models.OneToOneField(ServiceCall, on_delete=models.SET_NULL, null=True, blank=True, related_name="diagnostic_session")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="created_diagnostic_sessions")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.site_id and self.customer_id and self.site.customer_id != self.customer_id:
            errors["site"] = "Session site does not belong to the selected customer."
        if self.asset_id and self.customer_id and self.asset.customer_id != self.customer_id:
            errors["asset"] = "Session asset does not belong to the selected customer."
        if self.escalated_call_id and self.customer_id and self.escalated_call.customer_id != self.customer_id:
            errors["escalated_call"] = "Escalated call belongs to another customer."
        if errors:
            raise ValidationError(errors)

    def __str__(self):
        return f"DiagSession {self.session_id} - {self.asset.name} ({self.status})"


class DiagnosticEvent(TenantOwnedModel):
    session = models.ForeignKey(DiagnosticSession, on_delete=models.CASCADE, related_name="events")
    seq_num = models.PositiveIntegerField()
    event_type = models.CharField(max_length=60)
    actor_role = models.CharField(max_length=30, default="customer")
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("session", "seq_num")
        ordering = ["session", "seq_num"]

    def save(self, *args, **kwargs):
        if self.pk:
            from django.core.exceptions import ValidationError
            raise ValidationError("DiagnosticEvent records are strictly immutable and cannot be updated.")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        from django.core.exceptions import ValidationError
        raise ValidationError("DiagnosticEvent records are strictly immutable and cannot be deleted.")

    def __str__(self):
        return f"{self.session_id} #{self.seq_num}: {self.event_type}"


class DiagnosticCommand(TenantOwnedModel):
    """Idempotency record for state-changing diagnostic operations."""
    session = models.ForeignKey(DiagnosticSession, on_delete=models.CASCADE, null=True, blank=True, related_name="commands")
    idempotency_key = models.CharField(max_length=120)
    command_type = models.CharField(max_length=60)
    request_hash = models.CharField(max_length=64)
    response_status = models.PositiveIntegerField(default=200)
    response_payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("tenant", "idempotency_key")
        indexes = [
            models.Index(fields=["tenant", "idempotency_key"]),
        ]

    def __str__(self):
        return f"Cmd {self.command_type} ({self.idempotency_key})"


class RecoveryPassport(TenantOwnedModel):
    session = models.OneToOneField(DiagnosticSession, on_delete=models.CASCADE, related_name="recovery_passport")
    service_call = models.OneToOneField(ServiceCall, on_delete=models.CASCADE, null=True, blank=True, related_name="recovery_passport")
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="recovery_passports")
    complaint = models.TextField()
    structured_data = models.JSONField(default=dict, help_text="Deterministic diagnostic evidence, observations, and outcomes")
    do_not_repeat_items = models.JSONField(default=list, help_text="Verified actions and checks that should not be repeated")
    evidence_completeness = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def clean(self):
        super().clean()
        from django.core.exceptions import ValidationError
        errors = {}
        if self.service_call_id and self.service_call.customer_id != self.session.customer_id:
            errors["service_call"] = "Service call customer does not match diagnostic session customer."
        if self.asset_id != self.session.asset_id:
            errors["asset"] = "Passport asset must match session asset."
        if errors:
            raise ValidationError(errors)

    def __str__(self):
        return f"RecoveryPassport for Asset {self.asset_id} (Session {self.session_id})"
