from pathlib import Path
from django import forms
from django.conf import settings

from .models import KnowledgeDocument, Asset, Customer, Site, ServiceCall

ALLOWED_KB_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_KB_UPLOAD_BYTES = getattr(settings, "SERVY_MAX_KB_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_IMPORT_BYTES = getattr(settings, "SERVY_MAX_IMPORT_BYTES", 5 * 1024 * 1024)
MAX_TEXT_CHARS = getattr(settings, "SERVY_MAX_DOCUMENT_TEXT_CHARS", 500_000)


class KnowledgeDocumentForm(forms.ModelForm):
    class Meta:
        model = KnowledgeDocument
        fields = [
            "customer", "domain", "category", "product", "asset", "title", "description",
            "doc_type", "source_type", "tags", "content_text", "file", "source_url",
            "version", "language", "is_confidential", "disable_sharing", "is_rag_enabled",
        ]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
            "content_text": forms.Textarea(attrs={"rows": 8, "placeholder": "Paste document text here when source type is Text."}),
            "tags": forms.TextInput(attrs={"placeholder": "troubleshooting, maintenance, model-MA100"}),
        }

    def __init__(self, *args, tenant=None, **kwargs):
        self.tenant = tenant
        super().__init__(*args, **kwargs)
        for name in ["customer", "domain", "category", "product", "asset"]:
            field = self.fields.get(name)
            if field and tenant:
                field.queryset = field.queryset.filter(tenant=tenant)
        for field in self.fields.values():
            cls = field.widget.attrs.get("class", "")
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = f"form-check-input {cls}".strip()
            else:
                field.widget.attrs["class"] = f"form-control {cls}".strip()

    def clean_file(self):
        f = self.cleaned_data.get("file")
        if not f:
            return f
        if f.size > MAX_KB_UPLOAD_BYTES:
            raise forms.ValidationError(f"File is too large. Maximum allowed size is {MAX_KB_UPLOAD_BYTES // (1024*1024)} MB.")
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_KB_EXTENSIONS:
            raise forms.ValidationError("Unsupported file type. Allowed: PDF, DOCX, TXT, MD, CSV.")
        # Lightweight signature checks stop obvious extension spoofing before the
        # file is persisted. Deep bounded parsing still happens in document_loader.
        head = f.read(16)
        f.seek(0)
        if ext == ".pdf" and not head.startswith(b"%PDF-"):
            raise forms.ValidationError("The uploaded file does not look like a valid PDF.")
        if ext == ".docx" and not head.startswith(b"PK"):
            raise forms.ValidationError("The uploaded file does not look like a valid DOCX archive.")
        if ext in {".txt", ".md", ".csv"} and b"\x00" in head:
            raise forms.ValidationError("Binary content is not accepted as a text/CSV knowledge document.")
        return f

    def clean(self):
        cleaned = super().clean()
        tenant = self.tenant
        for field_name in ["customer", "domain", "category", "product", "asset"]:
            obj = cleaned.get(field_name)
            if obj and tenant and obj.tenant_id != tenant.id:
                self.add_error(field_name, "Selected item belongs to another tenant.")
        customer = cleaned.get("customer")
        asset = cleaned.get("asset")
        product = cleaned.get("product")
        if asset:
            if customer and asset.customer_id and asset.customer_id != customer.id:
                self.add_error("asset", "Asset does not belong to the selected customer.")
            if product and asset.product_id and asset.product_id != product.id:
                self.add_error("asset", "Asset does not belong to the selected product.")
        source_type = cleaned.get("source_type")
        content_text = (cleaned.get("content_text") or "").strip()
        if len(content_text) > MAX_TEXT_CHARS:
            self.add_error("content_text", f"Text content is limited to {MAX_TEXT_CHARS:,} characters.")
        if source_type == "file" and not cleaned.get("file") and not self.instance.file:
            self.add_error("file", "Upload a document for File source type.")
        if source_type == "text" and not content_text:
            self.add_error("content_text", "Enter text content for Text source type.")
        if source_type in {"link", "video"} and not cleaned.get("source_url"):
            self.add_error("source_url", "Enter the source URL for Link/Video knowledge items.")
        return cleaned


class StaffRAGQueryForm(forms.Form):
    customer = forms.ModelChoiceField(queryset=Customer.objects.none(), required=True)
    site = forms.ModelChoiceField(queryset=Site.objects.none(), required=False)
    asset = forms.ModelChoiceField(queryset=Asset.objects.none(), required=True)
    service_call = forms.ModelChoiceField(queryset=ServiceCall.objects.none(), required=False, help_text="Optional: ground the answer in an existing Call Register entry.")
    question = forms.CharField(widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Describe the issue or ask a product question..."}), max_length=4000)

    def __init__(self, *args, tenant=None, **kwargs):
        self.tenant = tenant
        super().__init__(*args, **kwargs)
        if tenant:
            self.fields["customer"].queryset = Customer.objects.filter(tenant=tenant).order_by("name")
            self.fields["site"].queryset = Site.objects.filter(tenant=tenant).select_related("customer").order_by("customer__name", "name")
            self.fields["asset"].queryset = Asset.objects.filter(tenant=tenant).select_related("product", "customer").order_by("name")
            self.fields["service_call"].queryset = ServiceCall.objects.filter(tenant=tenant).select_related("customer", "asset").order_by("-created_at")[:500]
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"

    def clean(self):
        cleaned = super().clean()
        customer = cleaned.get("customer")
        site = cleaned.get("site")
        asset = cleaned.get("asset")
        call = cleaned.get("service_call")
        if customer and asset and asset.customer_id and asset.customer_id != customer.id:
            self.add_error("asset", "This asset does not belong to the selected customer.")
        if customer and site and site.customer_id != customer.id:
            self.add_error("site", "This site does not belong to the selected customer.")
        if call:
            if customer and call.customer_id != customer.id:
                self.add_error("service_call", "The selected call belongs to a different customer.")
            if asset and call.asset_id and call.asset_id != asset.id:
                self.add_error("service_call", "The selected call belongs to a different asset.")
        return cleaned


class StaffCustomerSupportForm(forms.Form):
    """Customer AI Support form used when staff acts on behalf of a customer before ticket creation."""
    customer = forms.ModelChoiceField(queryset=Customer.objects.none(), required=True)
    site = forms.ModelChoiceField(queryset=Site.objects.none(), required=False)
    asset = forms.ModelChoiceField(queryset=Asset.objects.none(), required=True)
    question = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Describe the problem or ask a question..."}),
        max_length=4000,
        required=True,
    )

    def __init__(self, *args, tenant=None, **kwargs):
        self.tenant = tenant
        super().__init__(*args, **kwargs)
        if tenant:
            self.fields["customer"].queryset = Customer.objects.filter(tenant=tenant).order_by("name")
            self.fields["site"].queryset = Site.objects.filter(tenant=tenant).select_related("customer").order_by("name")
            self.fields["asset"].queryset = Asset.objects.filter(tenant=tenant).select_related("product", "site", "customer").order_by("name")
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"

    def clean(self):
        cleaned = super().clean()
        customer = cleaned.get("customer")
        site = cleaned.get("site")
        asset = cleaned.get("asset")
        if customer and site and site.customer_id != customer.id:
            self.add_error("site", "Selected site does not belong to the chosen customer.")
        if customer and asset and asset.customer_id != customer.id:
            self.add_error("asset", "Selected asset does not belong to the chosen customer.")
        if site and asset and asset.site_id and asset.site_id != site.id:
            self.add_error("asset", "Selected asset does not belong to the chosen site.")
        cleaned["question"] = (cleaned.get("question") or "").strip()
        return cleaned


class CustomerRAGQueryForm(forms.Form):
    """Customer self-service AI support form used BEFORE ticket creation."""
    site = forms.ModelChoiceField(queryset=Site.objects.none(), required=False)
    asset = forms.ModelChoiceField(queryset=Asset.objects.none(), required=True)
    question = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Describe the problem or ask a question..."}),
        max_length=4000,
        required=True,
    )

    def __init__(self, *args, tenant=None, customer=None, **kwargs):
        self.tenant = tenant
        self.customer = customer
        super().__init__(*args, **kwargs)
        if tenant and customer:
            self.fields["site"].queryset = Site.objects.filter(tenant=tenant, customer=customer).order_by("name")
            self.fields["asset"].queryset = Asset.objects.filter(tenant=tenant, customer=customer).select_related("product", "site").order_by("name")
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"

    def clean(self):
        cleaned = super().clean()
        site = cleaned.get("site")
        asset = cleaned.get("asset")
        if not self.customer:
            self.add_error("asset", "No customer account identified.")
            return cleaned
        if site and site.customer_id != self.customer.id:
            self.add_error("site", "Invalid site for this customer account.")
        if asset and asset.customer_id != self.customer.id:
            self.add_error("asset", "Invalid asset for this customer account.")
        if site and asset and asset.site_id and asset.site_id != site.id:
            self.add_error("asset", "Selected asset does not belong to the chosen site.")
        cleaned["question"] = (cleaned.get("question") or "").strip()
        return cleaned



class CallCopilotForm(forms.Form):
    question = forms.CharField(
        max_length=4000,
        widget=forms.Textarea(attrs={"rows": 3, "class": "form-control", "placeholder": "Ask about this call, similar past cases, or relevant troubleshooting..."}),
    )


class CallImportForm(forms.Form):
    file = forms.FileField(help_text="Upload .xlsx file. Column names can match the included template.")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if f.size > MAX_IMPORT_BYTES:
            raise forms.ValidationError(f"Import file is too large. Maximum is {MAX_IMPORT_BYTES // (1024*1024)} MB.")
        if Path(f.name).suffix.lower() != ".xlsx":
            raise forms.ValidationError("Only .xlsx files are accepted.")
        return f
