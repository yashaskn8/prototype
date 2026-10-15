"""
Comprehensive Unit Tests for Canonical Knowledge Scope Policy (Operation 12.0).

Verifies exact 5-level equipment hierarchy:
1. Exact Asset
2. Product
3. Category
4. Domain
5. Fleet General

Verifies negative gates:
- Cross-asset rejection
- Cross-product rejection
- Cross-category rejection
- Cross-domain rejection
- Unscoped query rejection of equipment-specific documents
- Predicate-to-queryset consistency across retrieval, grounding, and safety checks.
"""
from django.test import TestCase
from core.models import (
    Tenant, Customer, ProductDomain, ProductCategory,
    Product, Asset, KnowledgeDocument, KnowledgeChunk
)
from core.services.knowledge_scope import (
    get_applicable_knowledge_queryset,
    get_applicable_chunks_queryset,
    is_document_applicable_to_asset,
    is_chunk_applicable_to_asset,
)


class KnowledgeScopeHierarchyTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant = Tenant.objects.create(name="Scope Test Tenant", slug="scope-tenant")
        cls.customer = Customer.objects.create(tenant=cls.tenant, name="Scope Customer")

        # Two distinct domains
        cls.domain_hvac = ProductDomain.objects.create(tenant=cls.tenant, name="HVAC")
        cls.domain_electrical = ProductDomain.objects.create(tenant=cls.tenant, name="Electrical")

        # Categories
        cls.cat_cooling = ProductCategory.objects.create(tenant=cls.tenant, domain=cls.domain_hvac, name="Cooling")
        cls.cat_heating = ProductCategory.objects.create(tenant=cls.tenant, domain=cls.domain_hvac, name="Heating")
        cls.cat_ups = ProductCategory.objects.create(tenant=cls.tenant, domain=cls.domain_electrical, name="UPS")

        # Products
        cls.prod_chiller = Product.objects.create(tenant=cls.tenant, category=cls.cat_cooling, name="Industrial Chiller")
        cls.prod_split = Product.objects.create(tenant=cls.tenant, category=cls.cat_cooling, name="Split AC")
        cls.prod_boiler = Product.objects.create(tenant=cls.tenant, category=cls.cat_heating, name="Industrial Boiler")

        # Assets
        cls.asset_chiller_1 = Asset.objects.create(
            tenant=cls.tenant, customer=cls.customer, product=cls.prod_chiller,
            name="Chiller Unit #1", asset_code="CH-01", model_number="CH-500",
        )
        cls.asset_chiller_2 = Asset.objects.create(
            tenant=cls.tenant, customer=cls.customer, product=cls.prod_chiller,
            name="Chiller Unit #2", asset_code="CH-02", model_number="CH-500",
        )
        cls.asset_split_1 = Asset.objects.create(
            tenant=cls.tenant, customer=cls.customer, product=cls.prod_split,
            name="Split AC #1", asset_code="SP-01", model_number="SP-100",
        )

        # Level 1: Document bound to exact asset (Chiller #1)
        cls.doc_asset = KnowledgeDocument.objects.create(
            tenant=cls.tenant, asset=cls.asset_chiller_1, title="Chiller #1 Custom Wiring",
            is_rag_enabled=True, checksum_sha256="hash-asset",
        )
        cls.chunk_asset = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_asset, chunk_index=0, heading="Wiring", text="Custom wiring for Chiller 1."
        )

        # Level 2: Document bound to product (Industrial Chiller)
        cls.doc_prod = KnowledgeDocument.objects.create(
            tenant=cls.tenant, product=cls.prod_chiller, title="Industrial Chiller Service Manual",
            is_rag_enabled=True, checksum_sha256="hash-prod",
        )
        cls.chunk_prod = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_prod, chunk_index=0, heading="Service", text="General chiller service."
        )

        # Level 3: Document bound to category (Cooling)
        cls.doc_cat = KnowledgeDocument.objects.create(
            tenant=cls.tenant, category=cls.cat_cooling, title="Cooling Standards Guide",
            is_rag_enabled=True, checksum_sha256="hash-cat",
        )
        cls.chunk_cat = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_cat, chunk_index=0, heading="Standards", text="Cooling system standards."
        )

        # Level 4: Document bound to domain (HVAC)
        cls.doc_dom = KnowledgeDocument.objects.create(
            tenant=cls.tenant, domain=cls.domain_hvac, title="HVAC Safety Protocol",
            is_rag_enabled=True, checksum_sha256="hash-dom",
        )
        cls.chunk_dom = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_dom, chunk_index=0, heading="Safety", text="HVAC safety protocols."
        )

        # Level 5: Fleet General (all hierarchy foreign keys NULL)
        cls.doc_gen = KnowledgeDocument.objects.create(
            tenant=cls.tenant, title="General Customer Service Terms",
            is_rag_enabled=True, checksum_sha256="hash-gen",
        )
        cls.chunk_gen = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_gen, chunk_index=0, heading="Terms", text="Terms and warranty."
        )

        # Disconnected/Opposite documents for negative tests
        cls.doc_boiler = KnowledgeDocument.objects.create(
            tenant=cls.tenant, product=cls.prod_boiler, title="Boiler Maintenance",
            is_rag_enabled=True, checksum_sha256="hash-boiler",
        )
        cls.chunk_boiler = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_boiler, chunk_index=0, heading="Burner", text="Boiler burner maintenance."
        )

        cls.doc_elec = KnowledgeDocument.objects.create(
            tenant=cls.tenant, domain=cls.domain_electrical, title="Electrical Domain Guide",
            is_rag_enabled=True, checksum_sha256="hash-elec",
        )
        cls.chunk_elec = KnowledgeChunk.objects.create(
            tenant=cls.tenant, document=cls.doc_elec, chunk_index=0, heading="High Voltage", text="Electrical guide."
        )

    def test_level_1_exact_asset_inheritance(self):
        """For Chiller #1, all 5 levels (Asset, Product, Category, Domain, General) apply."""
        ctx = {"tenant": self.tenant, "role": "customer", "customer": self.customer}
        qs = get_applicable_knowledge_queryset(self.asset_chiller_1, ctx)
        doc_ids = set(qs.values_list("id", flat=True))

        self.assertIn(self.doc_asset.id, doc_ids)
        self.assertIn(self.doc_prod.id, doc_ids)
        self.assertIn(self.doc_cat.id, doc_ids)
        self.assertIn(self.doc_dom.id, doc_ids)
        self.assertIn(self.doc_gen.id, doc_ids)

        # Foreign product/category/domain documents must be strictly excluded
        self.assertNotIn(self.doc_boiler.id, doc_ids)
        self.assertNotIn(self.doc_elec.id, doc_ids)

    def test_cross_asset_exclusion(self):
        """Chiller #2 shares product/category/domain but NOT exact asset doc of Chiller #1."""
        ctx = {"tenant": self.tenant, "role": "customer", "customer": self.customer}
        qs = get_applicable_knowledge_queryset(self.asset_chiller_2, ctx)
        doc_ids = set(qs.values_list("id", flat=True))

        self.assertNotIn(self.doc_asset.id, doc_ids)
        self.assertIn(self.doc_prod.id, doc_ids)
        self.assertIn(self.doc_cat.id, doc_ids)
        self.assertIn(self.doc_dom.id, doc_ids)
        self.assertIn(self.doc_gen.id, doc_ids)

    def test_cross_product_same_category(self):
        """Split AC shares Category (Cooling) and Domain (HVAC), but NOT Product (Chiller)."""
        ctx = {"tenant": self.tenant, "role": "customer", "customer": self.customer}
        qs = get_applicable_knowledge_queryset(self.asset_split_1, ctx)
        doc_ids = set(qs.values_list("id", flat=True))

        self.assertNotIn(self.doc_asset.id, doc_ids)
        self.assertNotIn(self.doc_prod.id, doc_ids)
        self.assertIn(self.doc_cat.id, doc_ids)
        self.assertIn(self.doc_dom.id, doc_ids)
        self.assertIn(self.doc_gen.id, doc_ids)

    def test_unscoped_query_allows_only_fleet_general(self):
        """When asset is None, ONLY Fleet General documents are returned."""
        ctx = {"tenant": self.tenant, "role": "customer", "customer": self.customer}
        qs = get_applicable_knowledge_queryset(None, ctx)
        doc_ids = set(qs.values_list("id", flat=True))

        self.assertEqual(doc_ids, {self.doc_gen.id})
        self.assertNotIn(self.doc_asset.id, doc_ids)
        self.assertNotIn(self.doc_prod.id, doc_ids)
        self.assertNotIn(self.doc_cat.id, doc_ids)
        self.assertNotIn(self.doc_dom.id, doc_ids)

    def test_in_memory_predicate_consistency(self):
        """is_document_applicable_to_asset matches get_applicable_knowledge_queryset exactly."""
        all_docs = [
            self.doc_asset, self.doc_prod, self.doc_cat,
            self.doc_dom, self.doc_gen, self.doc_boiler, self.doc_elec,
        ]

        # Test Chiller #1
        for doc in all_docs:
            expected = doc.id in {self.doc_asset.id, self.doc_prod.id, self.doc_cat.id, self.doc_dom.id, self.doc_gen.id}
            actual = is_document_applicable_to_asset(doc, self.asset_chiller_1)
            self.assertEqual(actual, expected, f"Mismatch on {doc.title} for Chiller #1")

        # Test Split AC
        for doc in all_docs:
            expected = doc.id in {self.doc_cat.id, self.doc_dom.id, self.doc_gen.id}
            actual = is_document_applicable_to_asset(doc, self.asset_split_1)
            self.assertEqual(actual, expected, f"Mismatch on {doc.title} for Split AC")

        # Test Unscoped (None)
        for doc in all_docs:
            expected = (doc.id == self.doc_gen.id)
            actual = is_document_applicable_to_asset(doc, None)
            self.assertEqual(actual, expected, f"Mismatch on {doc.title} for unscoped query")

    def test_chunk_predicate_and_queryset_consistency(self):
        """Chunks queryset and chunk predicate produce matching outcomes."""
        ctx = {"tenant": self.tenant, "role": "customer", "customer": self.customer}
        chunk_qs = get_applicable_chunks_queryset(self.asset_chiller_1, ctx)
        chunk_ids = set(chunk_qs.values_list("id", flat=True))

        self.assertIn(self.chunk_asset.id, chunk_ids)
        self.assertIn(self.chunk_prod.id, chunk_ids)
        self.assertIn(self.chunk_cat.id, chunk_ids)
        self.assertIn(self.chunk_dom.id, chunk_ids)
        self.assertIn(self.chunk_gen.id, chunk_ids)
        self.assertNotIn(self.chunk_boiler.id, chunk_ids)
        self.assertNotIn(self.chunk_elec.id, chunk_ids)

        self.assertTrue(is_chunk_applicable_to_asset(self.chunk_asset, self.asset_chiller_1))
        self.assertFalse(is_chunk_applicable_to_asset(self.chunk_boiler, self.asset_chiller_1))
