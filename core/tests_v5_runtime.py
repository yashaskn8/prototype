"""Opt-in real-weight V5 request-path check using disposable synthetic records.

Run with SERVY_TEST_REAL_V5=1 and an isolated test database. CI skips this
test because the private checkpoint is deliberately absent from Git.
"""
import math
import os
from unittest import mock, skipUnless

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from core.ml.reranker import get_reranker
from core.models import (
    Asset, Customer, KnowledgeDocument, Product, ServiceCall, Site,
    StaffProfile, Tenant, TenantMembership,
)
from core.services.indexing import index_document
from core.services.rag import ask_rag
from core.services.retriever import retrieve


@skipUnless(os.getenv("SERVY_TEST_REAL_V5") == "1", "Requires the local V5 weights")
@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_LLM_PROVIDER="extractive",
    SERVY_RAG_MIN_SCORE=0.0,
    SERVY_RERANKER_ENABLED=True,
)
class RealV5RequestPathTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="V5 test tenant", slug="v5-test")
        other_tenant = Tenant.objects.create(name="Other test tenant", slug="v5-other")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Test customer")
        other_customer = Customer.objects.create(tenant=self.tenant, name="Other customer")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Test site")
        product = Product.objects.create(tenant=self.tenant, name="Hydraulic pump")
        other_product = Product.objects.create(tenant=self.tenant, name="Laser cutter")
        self.asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site,
            product=product, name="Pump A", asset_code="PUMP-A",
        )

        def document(tenant, title, text, *, customer=None, product=None, confidential=False):
            item = KnowledgeDocument.objects.create(
                tenant=tenant, customer=customer, product=product, title=title,
                content_text=text, is_confidential=confidential, is_rag_enabled=True,
                source_type="text",
            )
            index_document(item)
            return item

        self.allowed = {
            document(
                self.tenant, "Pump intake manual",
                "If the hydraulic pump loses intake pressure, inspect the intake strainer for blockage.",
                product=product,
            ).id,
            document(
                self.tenant, "Pump pressure guide",
                "For low hydraulic pressure, confirm reservoir level and check the pressure gauge.",
                product=product,
            ).id,
        }
        self.forbidden = {
            document(
                self.tenant, "Other customer procedure",
                "OTHER_CUSTOMER_SECRET hydraulic pump pressure procedure.",
                customer=other_customer, product=product,
            ).id,
            document(
                self.tenant, "Confidential factory procedure",
                "CONFIDENTIAL_SECRET hydraulic pump pressure procedure.",
                product=product, confidential=True,
            ).id,
            document(
                self.tenant, "Wrong equipment procedure",
                "WRONG_EQUIPMENT_SECRET hydraulic pump pressure procedure.",
                product=other_product,
            ).id,
            document(
                other_tenant, "Other tenant procedure",
                "OTHER_TENANT_SECRET hydraulic pump pressure procedure.",
            ).id,
        }

        self.customer_user = User.objects.create_user("v5_customer", password="test-password")
        TenantMembership.objects.create(
            user=self.customer_user, tenant=self.tenant,
            role="customer", customer=self.customer,
        )
        self.engineer_user = User.objects.create_user("v5_engineer", password="test-password")
        TenantMembership.objects.create(
            user=self.engineer_user, tenant=self.tenant, role="technician",
        )
        StaffProfile.objects.create(
            tenant=self.tenant, user=self.engineer_user,
            full_name="Test engineer", role="technician",
        )
        self.call = ServiceCall.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site,
            asset=self.asset, servy_id=424242, complaint_type="Low pressure",
            complaint_text="Hydraulic pump loses intake pressure under load.",
            status="open",
        )

    def test_real_v5_scores_authorized_chunks_through_rag_and_http(self):
        model = get_reranker()
        self.assertIsNotNone(model, "Verified local V5 checkpoint did not load")
        question = "Hydraulic pump loses intake pressure under load; what should I check?"
        with mock.patch.object(model, "predict", wraps=model.predict) as predict:
            retrieved = retrieve(
                self.tenant, question, asset=self.asset, customer=self.customer,
                include_confidential=False, include_past_resolutions=False,
                top_k=5, min_score=0.0,
            )
            self.assertTrue(retrieved.reranker_applied)
            self.assertEqual(retrieved.reranker_model, "servy_v5_bge_verified")
            self.assertTrue(predict.called, "RAG did not execute real V5 inference")
            scored_pairs = [pair for call in predict.call_args_list for pair in call.args[0]]
            self.assertTrue(scored_pairs)
            for _, passage in scored_pairs:
                for marker in (
                    "OTHER_CUSTOMER_SECRET", "CONFIDENTIAL_SECRET",
                    "WRONG_EQUIPMENT_SECRET", "OTHER_TENANT_SECRET",
                ):
                    self.assertNotIn(marker, passage)

        self.assertTrue(retrieved)
        self.assertTrue({item.document_id for item in retrieved} <= self.allowed)
        self.assertFalse({item.document_id for item in retrieved} & self.forbidden)
        for item in retrieved:
            self.assertIsNotNone(item.chunk_id)
            self.assertTrue(math.isfinite(item.reranker_score))

        result = ask_rag(
            tenant=self.tenant, question=question, asset=self.asset,
            customer=self.customer, include_confidential=False, staff_mode=False,
        )
        self.assertTrue(result["reranker_applied"])
        self.assertEqual(result["reranker_model"], "servy_v5_bge_verified")
        self.assertTrue(result["retrieval_backend"])
        self.assertTrue({ref["document_id"] for ref in result["references"]} <= self.allowed)
        self.assertTrue(all(ref["chunk_id"] for ref in result["references"]))

        with mock.patch("core.ml.reranker.get_reranker", return_value=None):
            fallback = retrieve(
                self.tenant, question, asset=self.asset, customer=self.customer,
                include_confidential=False, include_past_resolutions=False,
                top_k=5, min_score=0.0,
            )
        self.assertFalse(fallback.reranker_applied)
        self.assertIsNone(fallback.reranker_model)
        self.assertTrue({item.document_id for item in fallback} <= self.allowed)

        self.client.force_login(self.customer_user)
        customer_response = self.client.post(reverse("api_customer_support_query"), {
            "site_id": self.site.id, "asset_id": self.asset.id, "question": question,
        }, content_type="application/json")
        self.assertEqual(customer_response.status_code, 200, customer_response.content)
        self.assertTrue(customer_response.json().get("reranker_applied"))
        self.assertEqual(customer_response.json().get("reranker_model"), "servy_v5_bge_verified")

        self.client.force_login(self.engineer_user)
        engineer_response = self.client.post(reverse("api_engineer_copilot_query"), {
            "call_id": self.call.id, "question": question,
        }, content_type="application/json")
        self.assertEqual(engineer_response.status_code, 200, engineer_response.content)
        self.assertTrue(engineer_response.json().get("reranker_applied"))
        self.assertEqual(engineer_response.json().get("reranker_model"), "servy_v5_bge_verified")
        self.assertTrue({ref["document_id"] for ref in engineer_response.json()["references"]} <= self.allowed)
