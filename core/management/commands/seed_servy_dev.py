"""Seed the isolated servy_dev database with synthetic records and the approved manuals.

Refuses unless connected to servy_dev as servy_dev_app (core.dev_database) and
CHROMA_PERSIST_DIR is a separate directory (never the production chroma_db).
Only the nine SHA-verified approved manual files are ingested; no company customer
or production database record is copied. Synthetic records use independent local
IDs; the approved source asset is kept only as mapping metadata. Embeddings are
real BGE vectors in the isolated ChromaDB directory; MiniLM is never loaded.
"""
import hashlib
import re
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from core.dev_database import DevDatabaseIsolationError, verify_isolated_database

DEV_TENANT_SLUG = "servy-local-dev"
# Synthetic asset code -> (approved source asset, manual model name)
ASSET_MAPPING = {
    "DEV-AST-059": ("SKY-AST-00001", 59, "Split AC Unit 2T"),
    "DEV-AST-060": ("SKY-AST-00002", 60, "VRF Outdoor Module"),
    "DEV-AST-061": ("SKY-AST-00003", 61, "Ductable AC 5T"),
}
MANUAL_DIR = Path(settings.BASE_DIR) / "media" / "knowledge" / "9" / "2026" / "10"


def verified_manuals():
    """[(sha256, source_asset_code, doc_type, path, text)] for the nine approved manuals."""
    from core.services.customer_isolation import VERIFIED_REAL_PROVENANCE_REGISTRY as REG
    by_hash = {}
    for path in sorted(MANUAL_DIR.glob("*.txt")):
        data = path.read_bytes()
        by_hash[hashlib.sha256(data).hexdigest()] = (path, data.decode("utf-8"))
    manuals = []
    for sha, entry in REG["documents"].items():
        if sha not in by_hash:
            raise CommandError(f"Approved manual for {entry['asset_code']} ({entry['doc_type']}) is missing or its SHA-256 differs.")
        path, text = by_hash[sha]
        manuals.append((sha, entry["asset_code"], entry["doc_type"], path, text))
    if len(manuals) != 9:
        raise CommandError(f"Expected 9 approved manuals, found {len(manuals)}.")
    return manuals


def chunk_approved_manual(document):
    """Create KnowledgeChunk rows exactly as company ingestion does (numbered sections).

    Uses core.services.starlly_ingest.split_document_into_sections, the chunker the
    approved manuals were originally ingested with, so section headings such as
    "3. SYMPTOM: POOR COOLING" are preserved for the answer stage.
    """
    from django.utils import timezone

    from core.models import KnowledgeChunk
    from core.services.starlly_ingest import split_document_into_sections

    text = document.content_text or ""
    sections = split_document_into_sections(text)
    KnowledgeChunk.objects.bulk_create([
        KnowledgeChunk(tenant=document.tenant, document=document, chunk_index=index,
                       heading=section["heading"][:220], text=section["text"],
                       embedding_model="approved-manual-sections-v1", is_quarantined=False)
        for index, section in enumerate(sections)])
    document.checksum_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document.index_status = "INDEXED"
    document.index_version = (document.index_version or 0) + 1
    document.indexed_at = timezone.now()
    document.save(update_fields=["checksum_sha256", "index_status", "index_version", "indexed_at"])
    return len(sections)


class Command(BaseCommand):
    help = "Seed servy_dev with synthetic records and the nine approved manuals (BGE, isolated ChromaDB)."

    def add_arguments(self, parser):
        parser.add_argument("--rechunk-manuals", action="store_true",
                            help="Re-chunk and re-embed only the nine development manual documents.")

    def _guard(self):
        try:
            verify_isolated_database("servy_dev", "servy_dev_app")
        except DevDatabaseIsolationError as exc:
            raise CommandError(f"Refusing to seed: {exc}") from exc
        chroma_dir = Path(settings.CHROMA_PERSIST_DIR).resolve()
        if chroma_dir == (Path(settings.BASE_DIR) / "chroma_db").resolve() or not chroma_dir.name:
            raise CommandError("Refusing to seed: CHROMA_PERSIST_DIR must be a separate servy_dev directory.")
        if getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "") != "bge-small":
            raise CommandError("Refusing to seed: the serving embedding profile must be bge-small.")
        return chroma_dir

    def _rechunk(self):
        chroma_dir = self._guard()
        from core.ml.indexer import index_single_document_bge, remove_document_bge
        from core.models import KnowledgeDocument, Tenant
        tenant = Tenant.objects.filter(slug=DEV_TENANT_SLUG).first()
        if tenant is None:
            raise CommandError("servy_dev is not seeded; run seed_servy_dev first.")
        expected = {sha for sha, *_ in verified_manuals()}
        documents = list(KnowledgeDocument.objects.filter(tenant=tenant, tags__startswith="approved-manual"))
        if len(documents) != 9 or {re.search(r"sha256 ([0-9a-f]{64})", d.description).group(1) for d in documents} != expected:
            raise CommandError("Development manual documents do not match the nine verified manuals; nothing changed.")
        chunks = vectors = 0
        for document in documents:
            remove_document_bge(document.id, profile_key="bge-small")
            with transaction.atomic():
                document.chunks.all().delete()
                chunks += chunk_approved_manual(document)
            vectors += index_single_document_bge(document, "bge-small")
        self.stdout.write(self.style.SUCCESS(
            f"Re-chunked {len(documents)} development manuals: {chunks} section chunks, {vectors} BGE vectors in {chroma_dir.name}."))

    def handle(self, *args, **options):
        if options["rechunk_manuals"]:
            return self._rechunk()
        chroma_dir = self._guard()

        from core.models import (Asset, Brand, Customer, KnowledgeDocument, Product, ProductCategory,
                                 ProductDomain, Site, StaffProfile, Tenant, TenantMembership)
        if Tenant.objects.filter(slug=DEV_TENANT_SLUG).exists():
            raise CommandError("servy_dev is already seeded; nothing was changed.")

        manuals = verified_manuals()
        self.stdout.write(f"Verified SHA-256 of {len(manuals)} approved manuals.")

        with transaction.atomic():
            tenant = Tenant.objects.create(name="Servy Local Development", slug=DEV_TENANT_SLUG, is_active=True)
            domain = ProductDomain.objects.create(tenant=tenant, name="HVAC (development)")
            category = ProductCategory.objects.create(tenant=tenant, domain=domain, name="Air Conditioning (development)")
            brand = Brand.objects.create(tenant=tenant, name="Synthetic Development Brand")
            customer = Customer.objects.create(
                tenant=tenant, name="Synthetic Development Customer", contact_name="Synthetic Contact",
                phone="+1 555 0100", email="dev-customer@example.invalid")
            site = Site.objects.create(tenant=tenant, customer=customer, name="Synthetic Development Site")
            assets = {}
            for code, (source_code, source_id, model) in ASSET_MAPPING.items():
                product = Product.objects.create(tenant=tenant, category=category, brand=brand, name=model)
                assets[source_code] = Asset.objects.create(
                    tenant=tenant, customer=customer, site=site, product=product, name=model,
                    asset_code=code, model_number=f"DEV-{source_id}")
            customer_user = User.objects.create(username="dev_customer", first_name="Synthetic", last_name="Customer")
            customer_user.set_unusable_password()
            customer_user.save()
            TenantMembership.objects.create(user=customer_user, tenant=tenant, role="customer", customer=customer)
            tech_user = User.objects.create(username="dev_technician", first_name="Synthetic", last_name="Technician")
            tech_user.set_unusable_password()
            tech_user.save()
            TenantMembership.objects.create(user=tech_user, tenant=tenant, role="technician")
            StaffProfile.objects.create(tenant=tenant, user=tech_user, full_name="Synthetic Technician", role="technician")

            documents = []
            for sha, source_code, doc_type, path, text in manuals:
                asset = assets[source_code]
                title = next((line.split(":", 1)[1].strip() for line in text.splitlines()
                              if line.startswith("Document:")), path.stem)
                documents.append(KnowledgeDocument.objects.create(
                    tenant=tenant, product=asset.product, asset=asset, domain=domain, category=category,
                    title=f"{asset.name} - {title}", description=(
                        f"Approved manual (source asset {source_code}, sha256 {sha}). "
                        "Development copy; mapping metadata only."),
                    doc_type=doc_type, source_type="text", content_text=text, original_filename=path.name,
                    tags=f"approved-manual,source:{source_code}", is_confidential=False, is_rag_enabled=True))

        # Section chunks exactly as company ingestion creates them, then real BGE
        # vectors only (no MiniLM sync is involved).
        from core.ml.indexer import index_single_document_bge
        chunks = vectors = 0
        for document in documents:
            chunks += chunk_approved_manual(document)
        for document in documents:
            vectors += index_single_document_bge(document, "bge-small")
        self.stdout.write(self.style.SUCCESS(
            f"Seeded servy_dev: tenant {tenant.id}, customer {customer.id}, assets "
            f"{ {code: a.id for code, a in assets.items()} }, {len(documents)} manuals, {chunks} chunks, "
            f"{vectors} BGE vectors in {chroma_dir.name}. No company customer or production records copied."))
