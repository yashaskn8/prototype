"""Create synthetic, asset-scoped RAG fixtures only in the isolated UI schema."""
import secrets
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from core.models import (Asset, Brand, Customer, KnowledgeDocument, Product,
                         ProductCategory, ProductDomain, Site, Tenant, TenantMembership)
from core.services.indexing import index_document


SYNTHETIC_GUIDES = (
    ("split", "Split AC 2T", "TEST-SPLIT-01", "Split AC troubleshooting", """
# Split AC 2T synthetic UI test guide
## Weak cooling after cleaning
Visible obstruction at the return grille may cause weak cooling.
Observe the selected mode, set temperature and any displayed code.
Observe whether the indoor air outlet and return grille are visibly blocked.
Observe whether the filter appears seated after the approved cleaning routine.
A weak-cooling symptom does not by itself prove a refrigerant fault.
## Safe next step
Stop using the unit if there is burning smell, exposed wiring or water near electrical parts.
Do not open electrical panels or connect gauges.
Observe the model, visible code and symptom timing to report if cooling remains weak.
"""),
    ("vrf", "VRF Outdoor Unit", "TEST-VRF-01", "VRF troubleshooting", """
# VRF outdoor unit synthetic UI test guide
## Low-pressure or capacity alert
Visible outdoor air-path obstruction may cause reduced capacity.
Observe the exact displayed fault code, time and operating mode.
Observe the outdoor fan and visible air paths from a safe public area.
The display alert alone does not establish refrigerant charge or a failed compressor.
## Technician boundary
Pressure measurement, refrigerant work and electrical cabinet inspection require authorized technicians.
Do not reset a repeating protective alarm.
Request technician inspection with the recorded code and symptom history.
"""),
    ("duct", "Ductable AC 5T", "TEST-DUCT-01", "Ductable AC airflow guide", """
# Ductable AC 5T synthetic UI test guide
## Low airflow
Visible obstruction at accessible grilles may cause low airflow.
Observe the thermostat setting, affected rooms and onset time.
Observe accessible supply and return grilles for visible obstruction.
Uneven airflow can have multiple causes; this observation does not confirm a motor or duct fault.
## Next step
Keep hands away from moving or electrical parts.
Do not open the fan cabinet or adjust wiring.
Observe the model, affected rooms and any displayed code to report to a trained technician.
"""),
)


class Command(BaseCommand):
    help = "Seed only a newly migrated, isolated MySQL UI test schema with synthetic fixtures."

    def add_arguments(self, parser):
        parser.add_argument("--refresh-guides", action="store_true",
                            help="Refresh only the three synthetic UI guide fixtures in the isolated schema.")

    def handle(self, *args, **options):
        from tools.servy_ui_test_isolation import PRIVATE, TEST_DB, TEST_USER, load_test_env
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE(), CURRENT_USER()")
            db_name, account = cursor.fetchone()
        if (db_name != TEST_DB or not account.startswith(f"{TEST_USER}@")
            or settings.SERVY_CUSTOMER_DATA_MODE != "test_isolated"
            or Path(settings.CHROMA_PERSIST_DIR).resolve() != (PRIVATE / "chroma").resolve()
            or Path(settings.MEDIA_ROOT).resolve() != (PRIVATE / "media").resolve()):
            raise CommandError("Synthetic seed refused: isolated database or storage identity failed.")
        if options["refresh_guides"]:
            if Tenant.objects.exclude(slug="servy-ui-isolated").exists():
                raise CommandError("Guide refresh refused: unexpected tenant exists in isolated schema.")
            docs = []
            for _, _, code, title, text in SYNTHETIC_GUIDES:
                try:
                    doc = KnowledgeDocument.objects.select_related("asset", "tenant").get(
                        tenant__slug="servy-ui-isolated", title=title, asset__asset_code=code,
                        source_type="text", is_confidential=False)
                except KnowledgeDocument.DoesNotExist as exc:
                    raise CommandError("Guide refresh refused: synthetic fixture identity changed.") from exc
                doc.content_text = text.strip()
                doc.save(update_fields=["content_text", "updated_at"])
                docs.append(doc)
            for doc in docs:
                index_document(doc)
            from core.ml.indexer import index_single_document_bge
            count = sum(index_single_document_bge(doc, profile_key="bge-small") for doc in docs)
            self.stdout.write(self.style.SUCCESS(f"Refreshed {len(docs)} isolated synthetic guides and {count} BGE vectors."))
            return
        if Tenant.objects.exists() or User.objects.exists():
            raise CommandError("Synthetic seed refused: business or login records already exist.")
        passwords = load_test_env()
        with transaction.atomic():
            tenant = Tenant.objects.create(name="Servy Isolated UI Test", slug="servy-ui-isolated")
            primary = Customer.objects.create(tenant=tenant, name="Synthetic Test Customer A")
            other = Customer.objects.create(tenant=tenant, name="Synthetic Test Customer B")
            primary_site = Site.objects.create(tenant=tenant, customer=primary,
                                               name="Synthetic HVAC Lab", city="Local Test")
            other_site = Site.objects.create(tenant=tenant, customer=other,
                                             name="Other Customer Site", city="Local Test")
            domain = ProductDomain.objects.create(tenant=tenant, name="Synthetic HVAC")
            category = ProductCategory.objects.create(tenant=tenant, domain=domain,
                                                      name="Synthetic Air Conditioning")
            brand = Brand.objects.create(tenant=tenant, name="Synthetic UI Fixture")
            docs = []
            for slug, product_name, code, title, text in SYNTHETIC_GUIDES:
                product = Product.objects.create(tenant=tenant, category=category, brand=brand,
                                                 name=product_name)
                asset = Asset.objects.create(tenant=tenant, customer=primary, site=primary_site,
                                             product=product, name=product_name,
                                             asset_code=code, model_number=f"UI-{slug.upper()}-1")
                docs.append(KnowledgeDocument.objects.create(
                    tenant=tenant, customer=primary, asset=asset, product=product,
                    category=category, domain=domain, title=title, doc_type="troubleshooting",
                    source_type="text", content_text=text.strip(), is_rag_enabled=True))
            undocumented = Product.objects.create(tenant=tenant, category=category, brand=brand,
                                                   name="Undocumented Test Equipment")
            Asset.objects.create(tenant=tenant, customer=primary, site=primary_site,
                                 product=undocumented, name="Undocumented Test Equipment",
                                 asset_code="TEST-NODOC-01", model_number="UI-NODOC-1")
            other_product = Product.objects.create(tenant=tenant, category=category, brand=brand,
                                                   name="Other Customer Test Equipment")
            other_asset = Asset.objects.create(tenant=tenant, customer=other, site=other_site,
                                               product=other_product, name="Other Customer Test Equipment",
                                               asset_code="TEST-OTHER-01", model_number="UI-OTHER-1")
            docs.append(KnowledgeDocument.objects.create(
                tenant=tenant, customer=other, asset=other_asset, product=other_product,
                category=category, domain=domain, title="Other customer private test guide",
                doc_type="troubleshooting", source_type="text",
                content_text="# Other customer guide\nThis text is visible only to Synthetic Test Customer B.",
                is_rag_enabled=True))
            docs.append(KnowledgeDocument.objects.create(
                tenant=tenant, customer=primary, asset=Asset.objects.get(asset_code="TEST-VRF-01", tenant=tenant),
                title="Confidential synthetic service note", doc_type="service_manual",
                source_type="text", content_text="# Internal test note\nTechnician-only confidential fixture.",
                is_confidential=True, is_rag_enabled=True))
            main_user = User.objects.create_user(
                username="servy_ui_tester", password=passwords["SERVY_UI_TEST_LOGIN_PASSWORD"],
                is_staff=False, is_superuser=False)
            TenantMembership.objects.create(user=main_user, tenant=tenant, role="customer",
                                            customer=primary, can_view_confidential=False)
            other_password = secrets.token_urlsafe(36)
            other_user = User.objects.create_user(username="servy_ui_other", password=other_password,
                                                  is_staff=False, is_superuser=False)
            TenantMembership.objects.create(user=other_user, tenant=tenant, role="customer",
                                            customer=other, can_view_confidential=False)
        # Keep the second account's password in the ACL-restricted local folder.
        (PRIVATE / "other_test_password.txt").write_text(other_password, encoding="utf-8")
        for doc in docs:
            if not index_document(doc):
                raise CommandError("Synthetic document indexing produced no chunks.")
        from core.ml.indexer import index_single_document_bge
        indexed = sum(index_single_document_bge(doc, profile_key="bge-small") for doc in docs)
        self.stdout.write(self.style.SUCCESS(
            f"Isolated synthetic fixtures: 2 customers, 5 assets, {len(docs)} documents, "
            f"{indexed} BGE vectors; tester username servy_ui_tester. No company records copied."))
