from django.test import TestCase, override_settings
from core.models import (
    Asset, Brand, Customer, KnowledgeDocument, Product, ProductCategory,
    ProductDomain, Site, Tenant,
)
from core.services.indexing import index_document
from core.services.rag import ask_rag


@override_settings(SERVY_LLM_PROVIDER="extractive", SERVY_RAG_MIN_SCORE=0.0)
class RagIsolationTests(TestCase):
    def setUp(self):
        self.t1 = Tenant.objects.create(name="Tenant A", slug="tenant-a")
        self.t2 = Tenant.objects.create(name="Tenant B", slug="tenant-b")
        self.c1 = Customer.objects.create(tenant=self.t1, name="Customer A")
        self.c2 = Customer.objects.create(tenant=self.t2, name="Customer B")
        self.s1 = Site.objects.create(tenant=self.t1, customer=self.c1, name="Site A")
        self.s2 = Site.objects.create(tenant=self.t2, customer=self.c2, name="Site B")

        d1 = ProductDomain.objects.create(tenant=self.t1, name="Dairy")
        d2 = ProductDomain.objects.create(tenant=self.t2, name="Dairy")
        c1 = ProductCategory.objects.create(tenant=self.t1, domain=d1, name="Milk")
        c2 = ProductCategory.objects.create(tenant=self.t2, domain=d2, name="Milk")
        b1 = Brand.objects.create(tenant=self.t1, name="Brand A")
        b2 = Brand.objects.create(tenant=self.t2, name="Brand B")
        p1 = Product.objects.create(tenant=self.t1, category=c1, brand=b1, name="Analyzer")
        p2 = Product.objects.create(tenant=self.t2, category=c2, brand=b2, name="Analyzer")
        self.a1 = Asset.objects.create(tenant=self.t1, customer=self.c1, site=self.s1, product=p1, name="A1", asset_code="A1", model_number="M1")
        self.a2 = Asset.objects.create(tenant=self.t2, customer=self.c2, site=self.s2, product=p2, name="B1", asset_code="B1", model_number="M1")

        doc1 = KnowledgeDocument.objects.create(tenant=self.t1, product=p1, asset=self.a1, title="Tenant A Guide", content_text="# Fix\nCheck the thermostat cup and sampling tube.", tags="thermostat", is_rag_enabled=True)
        doc2 = KnowledgeDocument.objects.create(tenant=self.t2, product=p2, asset=self.a2, title="Tenant B Secret", content_text="# Secret\nUse the private Tenant B procedure.", tags="private", is_rag_enabled=True)
        index_document(doc1)
        index_document(doc2)

    def test_tenant_isolation(self):
        result = ask_rag(tenant=self.t1, customer=self.c1, asset=self.a1, question="thermostat issue")
        titles = [r["title"] for r in result["references"]]
        self.assertIn("Tenant A Guide", titles)
        self.assertNotIn("Tenant B Secret", titles)

    def test_grounded_fallback(self):
        result = ask_rag(tenant=self.t1, customer=self.c1, asset=self.a1, question="What should I check for the thermostat?")
        self.assertIn("thermostat", result["answer"].lower())
        self.assertTrue(result["references"])
