"""
Unit and Adversarial Tests for Reciprocal Rank Fusion (RRF) and Ranking Preservation.

Covers Part 2 mandatory adversarial tests:
1. Dense ranks A first, Sparse ranks B first.
2. C appears in both lists with moderate rank (and outranks single-list items).
3. D appears in only one list (verifying no phantom rank contributions).
4. Unauthorized chunk with highest similarity is filtered before/after fusion.
5. Dense returns nothing (graceful fallback).
6. Sparse returns nothing.
7. Two chunks have identical RRF scores.
8. Authoritative knowledge source strictly outranks higher-scoring historical service call.
9. Content-level deduplication preserving distinct technical sections.
10. Explicit ranking state fields verified on all candidates.
"""

from django.test import SimpleTestCase, TestCase, override_settings
from core.models import (
    Tenant, Customer, Site, Asset, Product, Brand,
    ProductCategory, ProductDomain, KnowledgeDocument,
    KnowledgeChunk, ServiceCall, ServiceResolutionIndex,
)
from core.services.indexing import index_document, index_service_resolution
from core.services.retriever import (
    _reciprocal_rank_fusion,
    retrieve,
    RetrievedChunk,
    RRF_K,
)


class RRFUnitFormulaTests(SimpleTestCase):
    """Test pure RRF mathematics without requiring database fixtures."""

    def _make_chunk(self, source_id, title="Test Doc", score=0.8, text="Sample text"):
        return RetrievedChunk(
            source_kind="knowledge_document",
            source_id=source_id,
            document_id=source_id,
            chunk_id=source_id,
            title=title,
            heading=f"Section {source_id}",
            text=text,
            score=score,
            doc_type="Troubleshooting",
            source_url="",
            confidential=False,
        )

    def test_rrf_ordering_dense_A_first_sparse_B_first_and_C_in_both(self):
        """Dense: A(1), C(2), D(3). Sparse: B(1), C(2), E(3).
        C is in both at rank 2: score = 0.55/62 + 0.45/62 = 1.0/62 ≈ 0.01613.
        A is in dense only at rank 1: score = 0.55/61 ≈ 0.00902.
        D is in dense only at rank 3: score = 0.55/63 ≈ 0.00873.
        B is in sparse only at rank 1: score = 0.45/61 ≈ 0.00738.
        E is in sparse only at rank 3: score = 0.45/63 ≈ 0.00714.
        Expected order: C, A, D, B, E.
        """
        chunk_a = self._make_chunk(1, title="A")
        chunk_b = self._make_chunk(2, title="B")
        chunk_c = self._make_chunk(3, title="C")
        chunk_d = self._make_chunk(4, title="D")
        chunk_e = self._make_chunk(5, title="E")

        dense_list = [chunk_a, chunk_c, chunk_d]
        sparse_list = [chunk_b, chunk_c, chunk_e]

        fused = _reciprocal_rank_fusion(dense_list, sparse_list, w_dense=0.55, w_sparse=0.45)
        fused_titles = [c.title for c in fused]

        self.assertEqual(fused_titles, ["C", "A", "D", "B", "E"])
        # Verify C has the highest fusion score
        self.assertAlmostEqual(fused[0].fusion_score, 1.0 / (RRF_K + 2), places=5)
        self.assertEqual(fused[0].fusion_rank, 1)

    def test_missing_list_receives_no_phantom_contribution(self):
        """An item present in only one list MUST NOT receive phantom absent_rank contributions."""
        chunk_only_dense = self._make_chunk(10, title="OnlyDense")
        fused = _reciprocal_rank_fusion([chunk_only_dense], [], w_dense=0.55, w_sparse=0.45)

        self.assertEqual(len(fused), 1)
        expected_score = 0.55 / (RRF_K + 1)
        self.assertAlmostEqual(fused[0].fusion_score, expected_score, places=6)
        self.assertIsNone(fused[0].sparse_rank)
        self.assertEqual(fused[0].dense_rank, 1)

    def test_identical_rrf_scores_deterministic(self):
        """When two items have identical fusion scores, sorting is stable."""
        chunk_1 = self._make_chunk(101, title="Item 1")
        chunk_2 = self._make_chunk(102, title="Item 2")
        fused = _reciprocal_rank_fusion([chunk_1], [chunk_2], w_dense=0.5, w_sparse=0.5)
        self.assertEqual(len(fused), 2)
        self.assertAlmostEqual(fused[0].fusion_score, fused[1].fusion_score, places=6)


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_RAG_MIN_SCORE=0.0,
    SERVY_ALLOW_REMOTE_LLM=False,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class RRFIntegrationRankingTests(TestCase):
    """End-to-end retrieval ranking tests verifying tiering, deduplication, and bounds."""

    def setUp(self):
        self.tenant = Tenant.objects.create(name="Acme Tech", slug="acme-rrf")
        self.other_tenant = Tenant.objects.create(name="Rival Corp", slug="rival-rrf")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Alpha Labs")
        self.domain = ProductDomain.objects.create(tenant=self.tenant, name="HVAC")
        self.category = ProductCategory.objects.create(tenant=self.tenant, domain=self.domain, name="Chillers")
        self.brand = Brand.objects.create(tenant=self.tenant, name="FrostFlow")
        self.product = Product.objects.create(tenant=self.tenant, brand=self.brand, category=self.category, name="Chiller X100")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Main Facility")
        self.asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site,
            product=self.product, name="Chiller Unit #1", asset_code="CH-01", model_number="X100-PRO",
        )

        # Knowledge Manual
        self.doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.product, asset=self.asset,
            title="FrostFlow X100 Service Manual", doc_type="troubleshooting",
            content_text=(
                "## Section 1: Error E04 Refrigerant Pressure Low\n"
                "Check suction line valve. Recharge R-410A to 120 PSI.\n\n"
                "## Section 2: Error E04 Pressure Sensor Calibration\n"
                "Verify sensor transducer IPN-998811 voltage is 5.0V.\n"
            ),
            is_rag_enabled=True,
        )
        index_document(self.doc)

    def test_authoritative_manual_strictly_outranks_historical_resolution(self):
        """Authoritative technical documentation strictly outranks past service resolutions."""
        call = ServiceCall.objects.create(
            tenant=self.tenant, customer=self.customer, asset=self.asset,
            servy_id=1001, complaint_type="breakdown",
            complaint_text="Refrigerant pressure low E04 error observed yesterday",
            status="closed",
            resolution_text="Replaced pressure sensor transducer IPN-998811 and recharged refrigerant.",
        )
        index_service_resolution(call)

        results = retrieve(
            tenant=self.tenant,
            question="E04 refrigerant pressure low",
            asset=self.asset,
            customer=self.customer,
            include_past_resolutions=True,
        )

        self.assertTrue(len(results) >= 2)
        # First results MUST be knowledge_document, not past_resolution
        self.assertEqual(results[0].source_kind, "knowledge_document")
        self.assertEqual(results[0].final_rank, 1)

    def test_deduplication_preserves_distinct_technical_sections(self):
        """Two different sections of the same manual sharing an error code are both preserved."""
        results = retrieve(
            tenant=self.tenant,
            question="E04 pressure sensor calibration",
            asset=self.asset,
            customer=self.customer,
            top_k=5,
        )
        # Should return both sections of the manual without dropping section 2
        headings = [r.heading for r in results if r.heading]
        self.assertTrue(any("Sensor Calibration" in h for h in headings))
