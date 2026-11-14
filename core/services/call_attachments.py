"""ServiceCall attachments: validation, storage and authorized lookup.

Limits match the previous local ticket workspace: at most 10 files, 25 MB each
and 25 MB in total. Only images, PDF and plain text are accepted; each file's
declared type, extension and leading bytes must agree. Files are stored under
MEDIA_ROOT (an isolated folder for servy_dev); metadata lives in MySQL.
"""
import hashlib
from pathlib import Path

from django.core.files.base import ContentFile
from django.db import transaction

MAX_ATTACHMENTS = 10
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024

# content type -> (allowed extensions, leading-bytes check)
ALLOWED_TYPES = {
    "image/jpeg": ({".jpg", ".jpeg"}, lambda b: b.startswith(b"\xff\xd8\xff")),
    "image/png": ({".png"}, lambda b: b.startswith(b"\x89PNG\r\n\x1a\n")),
    "image/gif": ({".gif"}, lambda b: b[:6] in (b"GIF87a", b"GIF89a")),
    "image/webp": ({".webp"}, lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP"),
    "application/pdf": ({".pdf"}, lambda b: b.startswith(b"%PDF-")),
    "text/plain": ({".txt"}, None),
}


class AttachmentError(ValueError):
    """An attachment set is rejected; .detail is safe to show the customer."""

    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


def _is_text(content):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return "\x00" not in text


def safe_filename(name):
    clean = Path(str(name or "attachment").replace("\\", "/")).name.strip() or "attachment"
    return clean[:200]


def validate_attachments(files):
    """files: iterable of objects with .name, .content_type and .read() (or dicts).

    Returns [(name, content_type, bytes)] or raises AttachmentError.
    """
    items = []
    for upload in files:
        if isinstance(upload, dict):
            name, content_type, content = upload.get("name"), upload.get("content_type"), upload.get("content")
        else:
            name, content_type = upload.name, getattr(upload, "content_type", "")
            if getattr(upload, "size", 0) and upload.size > MAX_ATTACHMENT_BYTES:
                raise AttachmentError(f"{safe_filename(name)} is larger than 25 MB.")
            content = upload.read()
        items.append((safe_filename(name), str(content_type or "").split(";")[0].strip().lower(), content or b""))
    if len(items) > MAX_ATTACHMENTS:
        raise AttachmentError(f"Attach at most {MAX_ATTACHMENTS} files.")
    total = 0
    for name, content_type, content in items:
        if not content:
            raise AttachmentError(f"{name} is empty.")
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise AttachmentError(f"{name} is larger than 25 MB.")
        total += len(content)
        rule = ALLOWED_TYPES.get(content_type)
        if rule is None:
            raise AttachmentError(f"{name}: only images (JPEG, PNG, GIF, WebP), PDF and plain text files are accepted.")
        extensions, signature = rule
        if Path(name).suffix.lower() not in extensions:
            raise AttachmentError(f"{name}: the file extension does not match its type.")
        if (signature and not signature(content)) or (content_type == "text/plain" and not _is_text(content)):
            raise AttachmentError(f"{name}: the file content does not match its declared type.")
    if total > MAX_TOTAL_BYTES:
        raise AttachmentError("Attachments exceed the 25 MB total limit.")
    return items


def store_attachments(service_call, items, uploaded_by=None):
    """Save validated items for a ServiceCall. All-or-nothing: files are removed on failure."""
    from core.models import ServiceCallAttachment

    saved = []
    try:
        with transaction.atomic():
            for name, content_type, content in items:
                attachment = ServiceCallAttachment(
                    tenant_id=service_call.tenant_id, service_call=service_call, original_name=name,
                    content_type=content_type, size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(), uploaded_by=uploaded_by)
                attachment.file.save(name, ContentFile(content), save=False)
                saved.append(attachment)
                attachment.save()
    except Exception:
        for attachment in saved:
            if attachment.file and attachment.file.name:
                attachment.file.storage.delete(attachment.file.name)
        raise
    return saved


def serialize_attachment(attachment):
    return {"id": str(attachment.id), "name": attachment.original_name,
            "type": attachment.content_type, "size": attachment.size}


def authorized_attachment(service_calls_queryset, call_id, attachment_id):
    """Return the attachment only if its ServiceCall is in the caller's authorized queryset."""
    from core.models import ServiceCallAttachment

    if not str(attachment_id).isdigit() or not str(call_id).isdigit():
        return None
    call = service_calls_queryset.filter(pk=int(call_id)).first()
    if call is None:
        return None
    return ServiceCallAttachment.objects.filter(
        pk=int(attachment_id), service_call=call, tenant_id=call.tenant_id).first()
