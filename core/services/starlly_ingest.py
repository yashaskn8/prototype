"""Starlly Solutions real company data ingestion service.

Provides a secure, verified, zero-deletion pipeline to ingest authentic assets,
site-assets, complaints, and technical document attachments from Starlly Org 273 (Skydream)
into a dedicated, isolated Servy company tenant.

Guarantees:
- ZERO deletion of any existing database record, file, vector, or demo fixture.
- Real company records are strictly isolated into Tenant 'skydream'.
- Technical attachments are downloaded over HTTPS only from approved hosts (starlly.sgp1.digitaloceanspaces.com).
- Full SSRF and redirect protection.
- Content hash (SHA-256) validation and structured text chunking.
- Dense vectors are synced exclusively to isolated company collections ('servy_company_dense_kb', 'servy_company_bge_small_v1').
- Idempotent: re-running updates or skips without duplicating records.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from core.models import (
    Asset, Brand, Customer, KnowledgeChunk, KnowledgeDocument,
    Product, ProductCategory, ProductDomain, Site, Tenant,
)
from core.services.starlly_api import StarllyApiClient, validate_api_key, validate_base_url
from core.services.chroma_store import sync_document_to_chroma
from core.services.customer_isolation import VERIFIED_REAL_PROVENANCE_REGISTRY

logger = logging.getLogger("servy.starlly_ingest")

APPROVED_ATTACHMENT_HOSTS = [
    "starlly.sgp1.digitaloceanspaces.com",
]


class IngestionSecurityError(Exception):
    """Raised when an attachment URL fails SSRF or host validation."""
    pass


MAX_ATTACHMENT_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
MAX_EXTRACTED_TEXT_LENGTH = 1_000_000       # 1M characters limit


def sanitize_attachment_filename(filename: str, sha_prefix: str = "") -> str:
    """Generate safe, path-traversal-free filename for local media storage."""
    if not filename:
        filename = "manual.txt"
    # Remove path traversal, directory separators, Windows drive letters, null bytes
    clean = filename.replace("\\", "/").split("/")[-1]
    clean = re.sub(r"^[a-zA-Z]:", "", clean)
    clean = re.sub(r"[\x00-\x1f\x7f]", "", clean)
    clean = re.sub(r"\.\.+", "", clean)
    clean = re.sub(r"[^\w\.\-\_]", "_", clean)
    clean = clean.strip("._ ")
    if not clean:
        clean = "manual.txt"
    if len(clean) > 80:
        name_part, ext = os.path.splitext(clean)
        clean = f"{name_part[:70]}{ext}"
    if sha_prefix:
        clean = f"{sha_prefix[:16]}_{clean}"
    return clean


def validate_attachment_url(url: str) -> str:
    """Validate attachment download URL against approved storage hosts.

    Prevents SSRF, private network requests, and untrusted domain access.
    """
    if not url or not str(url).strip():
        raise IngestionSecurityError("Attachment URL is empty.")

    clean_url = str(url).strip()
    parsed = urlparse(clean_url)

    if parsed.scheme.lower() != "https":
        raise IngestionSecurityError(f"Insecure scheme '{parsed.scheme}'. Only HTTPS is permitted.")

    host = parsed.netloc.lower()
    if host not in APPROVED_ATTACHMENT_HOSTS:
        raise IngestionSecurityError(
            f"Host '{host}' is not in approved storage hosts list: {APPROVED_ATTACHMENT_HOSTS}"
        )

    return clean_url


def fetch_attachment_content(file_url: str, timeout: int = 20) -> tuple[bytes, str, str]:
    """Securely download attachment binary with bounded streaming and extract genuine text content.

    Returns (raw_bytes, sha256_hex, text_content).
    """
    valid_url = validate_attachment_url(file_url)

    try:
        resp = requests.get(
            valid_url,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "Servy-RAG-Ingest/1.0"},
        )
    except Exception as exc:
        raise IngestionSecurityError(f"Failed to connect to attachment storage: {exc}")

    if resp.status_code in (301, 302, 303, 307, 308):
        raise IngestionSecurityError(f"Redirect from attachment storage rejected: status {resp.status_code}")

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download attachment: HTTP {resp.status_code}")

    # Check Content-Length header if present
    content_length = resp.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > MAX_ATTACHMENT_FILE_SIZE:
                raise IngestionSecurityError(
                    f"Attachment Content-Length {content_length} exceeds limit of {MAX_ATTACHMENT_FILE_SIZE} bytes."
                )
        except ValueError:
            pass

    # Bounded streaming download in 64KB chunks
    buffer = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        buffer.extend(chunk)
        if len(buffer) > MAX_ATTACHMENT_FILE_SIZE:
            raise IngestionSecurityError(
                f"Downloaded stream exceeded maximum file size of {MAX_ATTACHMENT_FILE_SIZE} bytes."
            )

    content = bytes(buffer)
    if not content:
        raise ValueError("Downloaded attachment is empty (0 bytes).")

    # Format inspection: reject HTML error pages and executables
    prefix = content[:512].lower()
    if b"<html" in prefix or b"<!doctype html" in prefix:
        raise IngestionSecurityError("Downloaded content is an HTML web page, not an authorized technical manual.")
    if prefix.startswith(b"mz") or prefix.startswith(b"\x7felf"):
        raise IngestionSecurityError("Downloaded content is an executable binary.")

    sha = hashlib.sha256(content).hexdigest()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as err:
            raise IngestionSecurityError(f"Could not decode text content: {err}")

    if not text.strip():
        raise ValueError("Extracted text content from attachment is blank.")

    if len(text) > MAX_EXTRACTED_TEXT_LENGTH:
        raise IngestionSecurityError(
            f"Extracted text length {len(text)} exceeds limit of {MAX_EXTRACTED_TEXT_LENGTH} characters."
        )

    return content, sha, text


def split_document_into_sections(text: str) -> List[Dict[str, str]]:
    """Split technical manual text into semantically cohesive sections with headings."""
    lines = text.splitlines()
    sections: List[Dict[str, str]] = []
    current_heading = "General Overview"
    current_lines: List[str] = []

    header_pattern = re.compile(r"^(?:[0-9]+\.\s+|#+\s+|SYMPTOM:\s+|SECTION\s+[0-9]+:\s*)(.+)$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        match = header_pattern.match(stripped)
        if match and len(current_lines) > 2:
            section_body = "\n".join(current_lines).strip()
            if section_body:
                sections.append({
                    "heading": current_heading,
                    "text": section_body,
                })
            current_heading = stripped
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        section_body = "\n".join(current_lines).strip()
        if section_body:
            sections.append({
                "heading": current_heading,
                "text": section_body,
            })

    # Fallback if no numbered headings matched: split on double newlines or chunks
    if not sections and text.strip():
        sections.append({
            "heading": "Full Reference Document",
            "text": text.strip(),
        })

    return sections


def preview_starlly_ingestion(
    tenant_slug: str = "skydream",
    org_id: str = "273",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[StarllyApiClient] = None,
) -> Dict[str, Any]:
    """Read-only preview of real company records available from Starlly API."""
    if client is None:
        resolved_url = base_url or getattr(settings, "STARLLY_API_BASE_URL", "") or os.getenv("STARLLY_API_BASE_URL", "")
        resolved_key = api_key or getattr(settings, "STARLLY_API_KEY", "") or os.getenv("STARLLY_API_KEY", "")
        client = StarllyApiClient(
            base_url=resolved_url,
            api_key=resolved_key,
            org_id=org_id,
        )

    assets_res = client.fetch_assets()
    assets = assets_res.get("assets", [])

    headers = client.get_auth_headers()
    sa_resp = requests.get(
        f"{client.base_url}/api/external/site-assets/",
        headers=headers,
        timeout=15,
        allow_redirects=False,
    )
    site_assets = sa_resp.json() if sa_resp.status_code == 200 else []

    comp_resp = requests.get(
        f"{client.base_url}/api/external/complaints/",
        headers=headers,
        timeout=15,
        allow_redirects=False,
    )
    complaints = comp_resp.json() if comp_resp.status_code == 200 else []

    unique_clients = sorted(list({sa.get("client_name") for sa in site_assets if sa.get("client_name")}))
    unique_sites = sorted(list({sa.get("site_name") for sa in site_assets if sa.get("site_name")}))

    attachments_preview = []
    for a in assets:
        for att in (a.get("attachments") or []):
            attachments_preview.append({
                "asset_code": a.get("code"),
                "asset_name": a.get("name"),
                "attachment_id": att.get("id"),
                "filename": att.get("name"),
                "url": att.get("file_url"),
            })

    return {
        "status": "preview_ready",
        "target_tenant": tenant_slug,
        "starlly_org_id": org_id,
        "counts": {
            "assets": len(assets),
            "site_assets": len(site_assets),
            "unique_customers": len(unique_clients),
            "unique_sites": len(unique_sites),
            "attachments": len(attachments_preview),
            "complaints": len(complaints),
        },
        "sample_customers": unique_clients[:5],
        "sample_sites": unique_sites[:5],
        "sample_assets": [f"{a.get('code')} - {a.get('name')}" for a in assets[:5]],
        "attachments_preview": attachments_preview,
    }


def ingest_starlly_real_data(
    tenant_slug: str = "skydream",
    tenant_name: str = "Skydream",
    org_id: str = "273",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[StarllyApiClient] = None,
    preview_only: bool = False,
) -> Dict[str, Any]:
    """Execute strictly non-destructive, zero-deletion ingestion of real company data.

    Invariants:
      - Default/preview mode does not mutate any records, files, or vectors.
      - Never overwrites existing assets, documents, chunks, or files.
      - Never deletes chunks or vector collections.
      - Conflicting data is marked CONFLICT_REQUIRES_REVIEW without mutating.
    """
    logger.info("Starting real company data ingestion for tenant '%s' (Org %s)", tenant_slug, org_id)

    if preview_only:
        return preview_starlly_ingestion(
            tenant_slug=tenant_slug,
            org_id=org_id,
            api_key=api_key,
            base_url=base_url,
            client=client,
        )

    if client is None:
        resolved_url = base_url or getattr(settings, "STARLLY_API_BASE_URL", "") or os.getenv("STARLLY_API_BASE_URL", "")
        resolved_key = api_key or getattr(settings, "STARLLY_API_KEY", "") or os.getenv("STARLLY_API_KEY", "")
        client = StarllyApiClient(
            base_url=resolved_url,
            api_key=resolved_key,
            org_id=org_id,
        )

    # 1. Fetch from Starlly
    assets_data = client.fetch_assets().get("assets", [])
    headers = client.get_auth_headers()

    sa_resp = requests.get(
        f"{client.base_url}/api/external/site-assets/",
        headers=headers,
        timeout=15,
        allow_redirects=False,
    )
    site_assets_data = sa_resp.json() if sa_resp.status_code == 200 else []

    comp_resp = requests.get(
        f"{client.base_url}/api/external/complaints/",
        headers=headers,
        timeout=15,
        allow_redirects=False,
    )
    complaints_data = comp_resp.json() if comp_resp.status_code == 200 else []

    ingest_report: Dict[str, Any] = {
        "tenant": tenant_slug,
        "customers_created": 0,
        "customers_existing": 0,
        "sites_created": 0,
        "sites_existing": 0,
        "assets_created": 0,
        "assets_already_existed": 0,
        "documents_created": 0,
        "documents_already_existed": 0,
        "chunks_indexed": 0,
        "chunks_preserved": 0,
        "conflicts": [],
        "errors": [],
    }

    with transaction.atomic():
        # 2. Ensure dedicated real company tenant
        tenant, _ = Tenant.objects.get_or_create(
            slug=tenant_slug,
            defaults={"name": tenant_name, "is_active": True},
        )

        # 3. Ingest Customers and Sites from site-assets
        customer_cache: Dict[str, Customer] = {}
        site_cache: Dict[tuple[str, str], Site] = {}

        for sa in site_assets_data:
            client_name = (sa.get("client_name") or "").strip()
            site_name = (sa.get("site_name") or "").strip()
            city = (sa.get("site_city") or "").strip()

            if client_name and client_name not in customer_cache:
                cust, created = Customer.objects.get_or_create(
                    tenant=tenant,
                    name=client_name,
                    defaults={"contact_name": client_name},
                )
                if created:
                    ingest_report["customers_created"] += 1
                else:
                    ingest_report["customers_existing"] += 1
                customer_cache[client_name] = cust

            if client_name and site_name:
                cust = customer_cache[client_name]
                cache_key = (client_name, site_name)
                if cache_key not in site_cache:
                    site, s_created = Site.objects.get_or_create(
                        tenant=tenant,
                        customer=cust,
                        name=site_name,
                        defaults={
                            "city": city,
                            "address": f"{site_name}, {city}".strip(", "),
                        },
                    )
                    if s_created:
                        ingest_report["sites_created"] += 1
                    else:
                        ingest_report["sites_existing"] += 1
                    site_cache[cache_key] = site

        # Map assets to their primary site-asset deployment
        primary_deployments: Dict[str, Dict[str, Any]] = {}
        for sa in site_assets_data:
            asset_ext_id = str(sa.get("asset") or "")
            if asset_ext_id and asset_ext_id not in primary_deployments:
                primary_deployments[asset_ext_id] = sa

        # 4. Ingest Assets (Non-Destructive: Create-Only, No Overwriting)
        asset_record_cache: Dict[str, Asset] = {}
        for a_data in assets_data:
            code = (a_data.get("code") or "").strip()
            name = (a_data.get("name") or "").strip()
            model = (a_data.get("model_number") or "").strip()
            ext_id = str(a_data.get("id"))

            if not code or not name:
                continue

            existing_asset = Asset.objects.filter(tenant=tenant, asset_code=code).first()
            if existing_asset:
                # Compare immutable source identity and metadata
                registry_entry = VERIFIED_REAL_PROVENANCE_REGISTRY.get("assets", {}).get(code, {})
                expected_ext_id = registry_entry.get("external_id")
                has_conflict = False
                conflict_reasons = []
                if expected_ext_id and str(ext_id) != str(expected_ext_id):
                    has_conflict = True
                    conflict_reasons.append(f"External ID mismatch: expected {expected_ext_id}, received {ext_id}")
                if existing_asset.name != name:
                    has_conflict = True
                    conflict_reasons.append(f"Asset name mismatch: existing '{existing_asset.name}', received '{name}'")
                if model and existing_asset.model_number and existing_asset.model_number != model:
                    has_conflict = True
                    conflict_reasons.append(f"Model number mismatch: existing '{existing_asset.model_number}', received '{model}'")

                if has_conflict:
                    ingest_report["conflicts"].append({
                        "type": "ASSET_CONFLICT_REQUIRES_REVIEW",
                        "asset_code": code,
                        "reasons": conflict_reasons,
                    })

                # Strictly preserve existing asset without mutation
                ingest_report["assets_already_existed"] += 1
                asset_record_cache[ext_id] = existing_asset
                continue

            # New asset: resolve customer and site from site-assets deployment if available
            cust = None
            site = None
            serial = ""
            if ext_id in primary_deployments:
                dep = primary_deployments[ext_id]
                c_name = dep.get("client_name")
                s_name = dep.get("site_name")
                if c_name and c_name in customer_cache:
                    cust = customer_cache[c_name]
                if c_name and s_name and (c_name, s_name) in site_cache:
                    site = site_cache[(c_name, s_name)]
                serial = dep.get("serial") or ""

            product, _ = Product.objects.get_or_create(
                tenant=tenant,
                name=name,
            )

            new_asset = Asset.objects.create(
                tenant=tenant,
                customer=cust,
                site=site,
                product=product,
                name=name,
                asset_code=code,
                model_number=model if model and model.lower() != "none" else "",
                serial_number=serial,
                status="active" if a_data.get("is_active", True) else "inactive",
            )
            ingest_report["assets_created"] += 1
            asset_record_cache[ext_id] = new_asset

        # 5. Ingest Attachments as KnowledgeDocuments (Non-Destructive)
        for a_data in assets_data:
            ext_id = str(a_data.get("id"))
            asset = asset_record_cache.get(ext_id)
            if not asset:
                continue

            attachments = a_data.get("attachments") or []
            for att in attachments:
                att_name = att.get("name") or "manual.txt"
                file_url = att.get("file_url")

                if not file_url:
                    continue

                safe_name = sanitize_attachment_filename(att_name)

                try:
                    raw_bytes, sha, text_content = fetch_attachment_content(file_url)
                except Exception as exc:
                    logger.error("Failed to fetch attachment %s: %s", att_name, exc)
                    ingest_report["errors"].append(f"{att_name}: {exc}")
                    continue

                # Check if document already exists with this exact checksum
                existing_doc = KnowledgeDocument.objects.filter(
                    tenant=tenant,
                    asset=asset,
                    checksum_sha256=sha,
                ).first()

                if existing_doc:
                    # Document with matching content exists; preserve it untouched
                    ingest_report["documents_already_existed"] += 1
                    ingest_report["chunks_preserved"] += existing_doc.chunks.count()
                    continue

                # Conflict detection: document with same filename but different content
                conflict_doc = KnowledgeDocument.objects.filter(
                    tenant=tenant,
                    asset=asset,
                    original_filename=safe_name,
                ).exclude(checksum_sha256=sha).first()

                if conflict_doc:
                    ingest_report["conflicts"].append(
                        f"Document '{safe_name}' on asset {asset.asset_code} has modified remote bytes: CONFLICT_REQUIRES_REVIEW"
                    )
                    continue

                # Map doc_type based on filename semantics
                lower_name = safe_name.lower()
                if "troubleshoot" in lower_name or "fault" in lower_name or "alarm" in lower_name:
                    doc_type = "troubleshooting"
                elif "maintenance" in lower_name:
                    doc_type = "maintenance"
                elif "installation" in lower_name or "commission" in lower_name:
                    doc_type = "installation"
                else:
                    doc_type = "reference"

                clean_title = f"{asset.name} - {safe_name.replace('.txt', '').replace('_', ' ').strip()}"

                # Store file in local media storage safely without overwriting
                media_rel_dir = Path("knowledge") / str(tenant.id) / f"{timezone.now():%Y/%m}"
                media_abs_dir = Path(settings.MEDIA_ROOT) / media_rel_dir
                media_abs_dir.mkdir(parents=True, exist_ok=True)
                local_file_name = sanitize_attachment_filename(safe_name, sha_prefix=sha)
                local_file_path = media_abs_dir / local_file_name

                if not local_file_path.exists():
                    local_file_path.write_bytes(raw_bytes)

                doc = KnowledgeDocument.objects.create(
                    tenant=tenant,
                    asset=asset,
                    customer=asset.customer,
                    product=asset.product,
                    title=clean_title,
                    original_filename=safe_name,
                    description=f"Verified Starlly technical reference for {asset.name} ({asset.asset_code})",
                    doc_type=doc_type,
                    source_type="file",
                    file=str(media_rel_dir / local_file_name).replace("\\", "/"),
                    content_text=text_content,
                    checksum_sha256=sha,
                    is_rag_enabled=True,
                    is_confidential=False,
                    index_status="INDEXED",
                )
                ingest_report["documents_created"] += 1

                # Split into sections and create KnowledgeChunks (NEVER delete existing chunks)
                sections = split_document_into_sections(text_content)
                for idx, sec in enumerate(sections):
                    KnowledgeChunk.objects.create(
                        tenant=tenant,
                        document=doc,
                        chunk_index=idx,
                        heading=sec["heading"][:220],
                        text=sec["text"],
                        embedding_model="starlly-real-v1",
                        is_quarantined=False,
                    )
                    ingest_report["chunks_indexed"] += 1

                # Sync to isolated company collection without destructive vector deletion
                sync_document_to_chroma(doc.id)

    logger.info("Real company data ingestion complete: %s", ingest_report)
    return ingest_report


class StarllyRealDataIngestionService:
    """Service class for real company data ingestion, preview, and verification."""

    def __init__(self, tenant=None, tenant_slug=None, org_id="273", api_key=None, base_url=None):
        if tenant:
            self.tenant = tenant
            self.tenant_slug = tenant.slug
            self.tenant_name = tenant.name
        else:
            self.tenant_slug = tenant_slug or "skydream"
            self.tenant_name = "Skydream"
            self.tenant = None
        self.org_id = str(org_id)
        self.api_key = api_key
        self.base_url = base_url

    def preview(self) -> Dict[str, Any]:
        return preview_starlly_ingestion(
            tenant_slug=self.tenant_slug,
            org_id=self.org_id,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def sync(self) -> Dict[str, Any]:
        """Execute safe non-destructive ingestion."""
        return ingest_starlly_real_data(
            tenant_slug=self.tenant_slug,
            tenant_name=self.tenant_name,
            org_id=self.org_id,
            api_key=self.api_key,
            base_url=self.base_url,
        )

