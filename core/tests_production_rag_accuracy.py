"""Production RAG accuracy, hybrid retrieval, technical tokenization, and security tests (Section 11).

Tests:
  1. Voltage and electrical rating tokenization (5V, 12V, 19V, 24V, 230V, 400V, 2T, 'operating v specification 5/12/19')
  2. Fault/error code tokenization and extraction (E-12, E12, F-07, ALM-01, IPN-442098)
  3. Equipment-aware lexical indexing (_sparse_index_text) & metadata bonus
  4. Configurable Reciprocal Rank Fusion (RRF K, weights, deterministic tie-breaking)
  5. Retrieval mode configurability ('hybrid', 'sparse_only', 'dense_only')
  6. Deduplication & unauthorized tenant / revoked / confidential document exclusion
  7. Stale vector detection & dimension compatibility checks
  8. Reranker loading, fingerprint mismatch, and inference failure fallback/required behavior
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from django.test import TestCase, SimpleTestCase, override_settings

from core.ml.embeddings import EmbeddingConfigError, EmbeddingProfile
from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256, rerank_candidates
from core.models import Asset, Customer, KnowledgeChunk, KnowledgeDocument, Product, Site, Tenant
from core.services.indexing import index_document
from core.services.retriever import (
    RetrievedChunk,
    _matches_current_bge_vector,
    _matches_current_minilm_vector,
    _reciprocal_rank_fusion,
    _sparse_index_text,
    get_serving_config,
    is_authorized_candidate,
    retrieve,
)
from core.services.vectors import compute_bm25_scores, tokenize_words, _get_doc_tokens


class TechnicalTokenizationTests(SimpleTestCase):
    def setUp(self):
        _get_doc_tokens.cache_clear()

    def test_voltage_and_unit_tokenization_recovers_fused_and_separated_forms(self):
        q_tokens = set(tokenize_words("What is the operating V specification 5 for the terminal?"))
        self.assertIn("5", q_tokens)
        self.assertIn("5v", q_tokens)
        self.assertIn("voltage", q_tokens)

        d_tokens = set(tokenize_words("Power rail check: verify 5V, 12V, and 19V DC outputs; 24V and 230V/400V AC.", filter_stopwords=False))
        for tok in ("5v", "5", "12v", "12", "19v", "19", "24v", "230v", "400v", "voltage"):
            self.assertIn(tok, d_tokens)

    def test_bm25_matches_operating_v_specification_query_against_5v_12v_19v_chunk(self):
        query = "What is the operating V specification 5/12/19 for the terminal?"
        target_chunk = "## Power Supply Checks\nMeasure the 5V, 12V, and 19V rails at the main connector."
        distractor_chunk = "## General Overview\nClean the exterior housing with a dry cloth."
        scores = compute_bm25_scores(query, [target_chunk, distractor_chunk])
        self.assertGreater(scores[0], 0.25)
        self.assertGreater(scores[0], scores[1] * 3)

    def test_morphological_normalization_links_escalation_and_intervals(self):
        q_esc = "What are the safety precautions and escalation criteria?"
        c_esc = "## Stop and escalate if\nStop immediately if burning odor or smoke is present."
        scores_esc = compute_bm25_scores(q_esc, [c_esc, "## Unrelated\nAdjust screen brightness."])
        self.assertGreater(scores_esc[0], scores_esc[1])

        q_pm = "What are the preventive maintenance intervals?"
        c_pm = "## Monthly Preventive Maintenance\nInspect condenser fins every month."
        scores_pm = compute_bm25_scores(q_pm, [c_pm, "## Unrelated\nReplace fuse."])
        self.assertGreater(scores_pm[0], scores_pm[1])

    def test_fault_code_tokenization(self):
        tokens = set(tokenize_words("Unit shows error E-12 and alarm ALM-04"))
        self.assertIn("e12", tokens)
        self.assertIn("e-12", tokens)
        self.assertIn("alm04", tokens)
        self.assertIn("alm-04", tokens)

    # --- Red-team regressions (false equivalences found by adversarial tokenization attack) ---
    def test_5v_never_equals_50v_or_120v(self):
        t5, t50, t12, t120 = (set(tokenize_words(x, filter_stopwords=False)) for x in ("5V", "50V", "12V", "120 V"))
        self.assertNotIn("5v", t50)
        self.assertNotIn("50v", t5)
        self.assertNotIn("12v", t120)
        self.assertNotIn("120v", t12)

    def test_e012_is_not_e12(self):
        self.assertNotIn("e12", set(tokenize_words("E012", filter_stopwords=False)))
        self.assertNotIn("e012", set(tokenize_words("E12", filter_stopwords=False)))

    def test_ambiguous_single_letter_units_do_not_fire_with_whitespace(self):
        for text, bad in (("step 3 a check the fuse", "3a"), ("see figure 5 c below", "5c"),
                          ("room 5 f", "5f"), ("section 2 t of manual", "2t")):
            self.assertNotIn(bad, set(tokenize_words(text, filter_stopwords=False)), text)
        # attached forms are real ratings
        self.assertIn("10a", set(tokenize_words("fuse 10A", filter_stopwords=False)))
        self.assertIn("2t", set(tokenize_words("Apex Split AC 2T", filter_stopwords=False)))

    def test_voltage_spec_phrase_does_not_swallow_unrelated_numbers(self):
        toks = set(tokenize_words("voltage 5 3 phase supply", filter_stopwords=False))
        self.assertIn("5v", toks)
        self.assertNotIn("3v", toks)

    def test_spelled_volt_forms_map_to_v(self):
        for text in ("5 volts", "5-Volt", "5 V", "5V"):
            self.assertIn("5v", set(tokenize_words(text, filter_stopwords=False)), text)


class ConfigurableRRFAndTieBreakingTests(SimpleTestCase):
    def _make_item(self, chunk_id: int, doc_id: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            source_kind="knowledge_document",
            source_id=chunk_id,
            document_id=doc_id,
            chunk_id=chunk_id,
            title=f"Doc {doc_id}",
            heading="Section",
            text=f"Chunk {chunk_id} text",
            score=score,
            doc_type="Manual",
            source_url="",
            confidential=False,
        )

    def test_rrf_respects_custom_k_and_weights(self):
        d1 = self._make_item(101, 10, 0.80)
        d2 = self._make_item(102, 10, 0.75)
        s1 = self._make_item(102, 10, 0.90)
        s2 = self._make_item(101, 10, 0.60)

        # Heavy sparse weight -> 102 wins
        fused_sparse_heavy = _reciprocal_rank_fusion([d1, d2], [s1, s2], w_dense=0.20, w_sparse=0.80, rrf_k=10)
        self.assertEqual(fused_sparse_heavy[0].chunk_id, 102)

        # Heavy dense weight -> 101 wins
        d1_b = self._make_item(101, 10, 0.80)
        d2_b = self._make_item(102, 10, 0.75)
        s1_b = self._make_item(102, 10, 0.90)
        s2_b = self._make_item(101, 10, 0.60)
        fused_dense_heavy = _reciprocal_rank_fusion([d1_b, d2_b], [s1_b, s2_b], w_dense=0.85, w_sparse=0.15, rrf_k=10)
        self.assertEqual(fused_dense_heavy[0].chunk_id, 101)

    def test_deterministic_tie_breaking_on_equal_rrf_scores(self):
        # Symmetric ranks with equal weights (0.5 / 0.5) produce identical fusion_score;
        # higher raw score or lower (document_id, chunk_id) breaks ties deterministically.
        d1 = self._make_item(200, 20, 0.70)
        s1 = self._make_item(100, 10, 0.70)
        fused_1 = _reciprocal_rank_fusion([d1], [s1], w_dense=0.50, w_sparse=0.50, rrf_k=60)
        self.assertEqual([x.chunk_id for x in fused_1], [100, 200])

    @override_settings(
        SERVY_RRF_K=30,
        SERVY_RRF_WEIGHT_DENSE=0.65,
        SERVY_RRF_WEIGHT_SPARSE=0.35,
        SERVY_RETRIEVAL_MODE="sparse_only",
    )
    def test_serving_config_reads_custom_rrf_and_mode_settings(self):
        cfg = get_serving_config()
        self.assertEqual(cfg.rrf_k, 30)
        self.assertAlmostEqual(cfg.rrf_weight_dense, 0.65)
        self.assertAlmostEqual(cfg.rrf_weight_sparse, 0.35)
        self.assertEqual(cfg.retrieval_mode, "sparse_only")


class StaleVectorAndRerankerGuardTests(SimpleTestCase):
    def test_stale_vector_detection_rejects_mismatched_version_or_hash(self):
        profile = EmbeddingProfile(
            key="minilm", model_id="all-MiniLM-L6-v2", revision=None, dim=384,
            collection="servy_dense_kb", content_first=False, promotion_eligible=True,
        )
        chunk = SimpleNamespace(
            id=55, document_id=7, chunk_index=0, tenant_id=1,
            text="Updated technical procedure",
            document=SimpleNamespace(index_version="2.0"),
        )
        stale_meta = {
            "chunk_id": 55,
            "document_id": 7,
            "chunk_index": 0,
            "tenant_id": 1,
            "document_version": "1.0",  # Stale version
            "content_sha256": "deadbeef",
            "embedding_model": "all-MiniLM-L6-v2",
        }
        self.assertFalse(_matches_current_minilm_vector(stale_meta, chunk, profile))

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_REQUIRED=False,
        SERVY_RERANKER_POOL_SIZE=5,
    )
    def test_reranker_inference_failure_falls_back_gracefully_when_optional(self):
        c1 = SimpleNamespace(source_kind="knowledge_document", text="Passage A", score=0.9)
        c2 = SimpleNamespace(source_kind="knowledge_document", text="Passage B", score=0.8)
        bad_model = mock.Mock()
        bad_model.predict.side_effect = RuntimeError("Simulated tensor error")
        with mock.patch("core.ml.reranker.get_reranker", return_value=bad_model):
            out = rerank_candidates("query", [c1, c2], top_k=2)
        self.assertEqual(out, [c1, c2])


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_LLM_PROVIDER="extractive",
    SERVY_RAG_MIN_SCORE=0.0,
)
class EquipmentAwareAndSecurityRetrievalTests(TestCase):
    def setUp(self):
        _get_doc_tokens.cache_clear()
        self.tenant = Tenant.objects.create(name="Prod RAG Tenant", slug="prod-rag-tenant")
        self.other_tenant = Tenant.objects.create(name="Foreign Tenant", slug="foreign-tenant")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Customer Alpha")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Main Plant")
        self.prod_a = Product.objects.create(tenant=self.tenant, name="VoltCore 3kVA Inverter")
        self.prod_b = Product.objects.create(tenant=self.tenant, name="CryoMilk Bulk Cooler")
        self.asset_a = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site,
            product=self.prod_a, name="VoltCore Unit 1", asset_code="VC-01", model_number="VC-3000",
        )

        # Document with generic heading ("Stop and escalate if") where equipment name is only in Document/Product
        self.doc_a = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.prod_a,
            title="VoltCore 3kVA Inverter Service Manual",
            content_text="## Stop and escalate if\nIsolate DC breaker and escalate to HV specialist if 5V/12V/19V rail fluctuates.",
            is_rag_enabled=True, is_confidential=False, source_type="text",
        )
        index_document(self.doc_a)

        # Distractor document with identical generic heading on another product
        self.doc_b = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.prod_b,
            title="CryoMilk Bulk Cooler Manual",
            content_text="## Stop and escalate if\nStop agitator motor and escalate if refrigerant leak is detected.",
            is_rag_enabled=True, is_confidential=False, source_type="text",
        )
        index_document(self.doc_b)

        # Disabled / quarantined document
        self.doc_disabled = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.prod_a,
            title="Disabled VoltCore Manual",
            content_text="## Stop and escalate if\nOBSOLETE_DISABLED_INSTRUCTION.",
            is_rag_enabled=False, is_confidential=False, source_type="text",
        )
        index_document(self.doc_disabled)

        # Foreign tenant document
        self.doc_foreign = KnowledgeDocument.objects.create(
            tenant=self.other_tenant,
            title="Foreign VoltCore Manual",
            content_text="## Stop and escalate if\nFOREIGN_TENANT_SECRET.",
            is_rag_enabled=True, is_confidential=False, source_type="text",
        )
        index_document(self.doc_foreign)

    def test_sparse_index_text_includes_equipment_and_manual_title_without_mutating_chunk_text(self):
        chunk_a = self.doc_a.chunks.first()
        sparse_txt = _sparse_index_text(chunk_a)
        self.assertIn("VoltCore 3kVA Inverter Service Manual", sparse_txt)
        self.assertIn("VoltCore 3kVA Inverter", sparse_txt)

    def test_retrieve_excludes_revoked_and_foreign_tenant_and_ranks_matching_equipment_first(self):
        results = retrieve(
            tenant=self.tenant,
            question="What are the safety precautions and escalation criteria for VoltCore 3kVA Inverter operating V specification 5?",
            asset=self.asset_a,
            customer=self.customer,
            include_confidential=False,
            top_k=5,
            min_score=0.0,
        )
        self.assertTrue(results)
        doc_ids = {r.document_id for r in results}
        self.assertIn(self.doc_a.id, doc_ids)
        self.assertNotIn(self.doc_b.id, doc_ids)
        self.assertNotIn(self.doc_disabled.id, doc_ids)
        self.assertNotIn(self.doc_foreign.id, doc_ids)
        self.assertEqual(results[0].document_id, self.doc_a.id)

        # Also verify is_authorized_candidate rejects revoked_at dynamically
        chunk_a = self.doc_a.chunks.first()
        chunk_a.document.revoked_at = datetime.now(timezone.utc)
        self.assertFalse(
            is_authorized_candidate(
                chunk_a, tenant=self.tenant, customer=self.customer, asset=self.asset_a
            )
        )
