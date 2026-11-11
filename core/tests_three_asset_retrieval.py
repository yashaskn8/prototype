"""Synthetic retrieval regressions for the three approved asset families.

These tests exercise the real authorization, sparse candidate retrieval and V5
cross-encoder ranking path using disposable synthetic manuals. They do not load
company documents, the private V5 checkpoint, or the production database.
"""
from __future__ import annotations

from unittest import mock

from django.test import TestCase, override_settings

from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256
from core.models import Asset, Customer, KnowledgeChunk, KnowledgeDocument, Product, Tenant
from core.services.knowledge_scope import is_document_applicable_to_asset
from core.services.retriever import _knowledge_candidates, retrieve
from core.services.vectors import embed_text


DOC_TYPES = ("troubleshooting", "maintenance", "installation")
ASSET_SPECS = (
    ("SKY-AST-00001", "Split AC Unit 2T", "same-customer"),
    ("SKY-AST-00002", "VRF Outdoor Module", "vrf-customer"),
    ("SKY-AST-00003", "Ductable AC 5T", "same-customer"),
)


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_RETRIEVAL_MODE="sparse_only",
    SERVY_RAG_MIN_SCORE=0.0,
    SERVY_SPARSE_RELEVANCE_GATE=0.0,
    SERVY_DENSE_RELEVANCE_GATE=0.0,
    SERVY_RERANKER_ENABLED=True,
    SERVY_RERANKER_REQUIRED=False,
    SERVY_RERANKER_MODEL_PATH="synthetic-v5/best_model",
    SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
    SERVY_RERANKER_POOL_SIZE=5,
)
class ThreeAssetNineManualRetrievalTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant = Tenant.objects.create(
            name="Synthetic three-asset retrieval tenant",
            slug="synthetic-three-asset-retrieval",
        )
        cls.customers = {
            "same-customer": Customer.objects.create(
                tenant=cls.tenant, name="Synthetic customer shared by assets 1 and 3"
            ),
            "vrf-customer": Customer.objects.create(
                tenant=cls.tenant, name="Synthetic VRF customer"
            ),
        }
        cls.assets = {}
        cls.documents_by_asset = {}
        cls.markers = {}
        cls.target_chunks = {}

        for asset_code, asset_name, customer_key in ASSET_SPECS:
            product = Product.objects.create(
                tenant=cls.tenant, name=f"{asset_name} synthetic product"
            )
            asset = Asset.objects.create(
                tenant=cls.tenant,
                customer=cls.customers[customer_key],
                product=product,
                name=asset_name,
                asset_code=asset_code,
                model_number=f"MODEL-{asset_code[-5:]}",
                status="active",
            )
            cls.assets[asset_code] = asset
            cls.documents_by_asset[asset_code] = set()

            for doc_type in DOC_TYPES:
                marker = f"servyqa{asset_code.replace('-', '').lower()}{doc_type}"
                title = f"{asset_name} {doc_type} regression manual"
                target_text = (
                    f"{marker} synthetic answer-bearing evidence for the "
                    f"{doc_type} manual belonging only to {asset_code}."
                )
                document = KnowledgeDocument.objects.create(
                    tenant=cls.tenant,
                    customer=asset.customer,
                    product=product,
                    asset=asset,
                    title=title,
                    doc_type=doc_type,
                    source_type="text",
                    content_text=target_text,
                    is_confidential=False,
                    disable_sharing=False,
                    is_rag_enabled=True,
                    index_status="INDEXED",
                )
                cls.documents_by_asset[asset_code].add(document.id)
                cls.markers[(asset_code, doc_type)] = marker

                target_chunk = KnowledgeChunk.objects.create(
                    tenant=cls.tenant,
                    document=document,
                    chunk_index=0,
                    heading=f"{doc_type.title()} evidence",
                    text=target_text,
                    embedding=embed_text(f"{title} {asset_name} {target_text}"),
                    content_embedding=embed_text(target_text),
                    embedding_model="hashing-word12-v1",
                )
                cls.target_chunks[(asset_code, doc_type)] = target_chunk

                # A second, generic chunk makes the candidate pool larger than
                # one passage per manual and tests V5's fixed top-five boundary.
                distractor_text = (
                    f"Generic background notes for {asset_name}; no unique "
                    f"{doc_type} symptom, fault, or requested evidence is stated."
                )
                KnowledgeChunk.objects.create(
                    tenant=cls.tenant,
                    document=document,
                    chunk_index=1,
                    heading="General background",
                    text=distractor_text,
                    embedding=embed_text(f"{title} {asset_name} {distractor_text}"),
                    content_embedding=embed_text(distractor_text),
                    embedding_model="hashing-word12-v1",
                )

    def test_each_asset_can_retrieve_only_its_three_bound_manuals(self):
        for asset_code, _asset_name, _customer_key in ASSET_SPECS:
            asset = self.assets[asset_code]
            candidates = _knowledge_candidates(
                self.tenant, asset, asset.customer, False, backend="all"
            )
            self.assertEqual(
                {chunk.document_id for chunk in candidates},
                self.documents_by_asset[asset_code],
                f"Candidate scope is incomplete or crossed the asset boundary for {asset_code}.",
            )
            self.assertEqual(len(self.documents_by_asset[asset_code]), 3)
            for candidate in candidates:
                self.assertTrue(is_document_applicable_to_asset(candidate.document, asset))
                self.assertEqual(candidate.document.tenant_id, self.tenant.id)

    def test_all_nine_manual_families_retrieve_their_evidence_with_v5(self):
        model = mock.Mock()
        scored_pairs = []

        def score_pairs(pairs, **kwargs):
            scored_pairs[:] = list(pairs)
            # Deterministic synthetic scorer: the passage containing the
            # query's unique evidence marker wins. The actual private V5
            # checkpoint is intentionally not needed for this plumbing test.
            query = pairs[0][0].lower() if pairs else ""
            marker = next((value for value in self.markers.values() if value in query), None)
            return [
                8.0 if marker and marker in passage.lower() else -8.0
                for _, passage in pairs
            ]

        model.predict.side_effect = score_pairs

        with mock.patch("core.ml.reranker.get_reranker", return_value=model):
            for asset_code, asset_name, _customer_key in ASSET_SPECS:
                asset = self.assets[asset_code]
                own_document_ids = self.documents_by_asset[asset_code]
                foreign_markers = {
                    marker
                    for (code, _doc_type), marker in self.markers.items()
                    if code != asset_code
                }

                for doc_type in DOC_TYPES:
                    marker = self.markers[(asset_code, doc_type)]
                    target_chunk = self.target_chunks[(asset_code, doc_type)]
                    scored_pairs.clear()
                    model.predict.reset_mock()

                    results = retrieve(
                        tenant=self.tenant,
                        question=f"Please explain {marker}",
                        asset=asset,
                        customer=asset.customer,
                        include_confidential=False,
                        top_k=5,
                        min_score=0.0,
                        include_past_resolutions=False,
                        staff_mode=False,
                    )

                    self.assertTrue(results, f"No results for {asset_name} / {doc_type}")
                    self.assertEqual(
                        results[0].chunk_id,
                        target_chunk.id,
                        f"V5 did not rank the expected {doc_type} passage first for {asset_code}.",
                    )
                    self.assertTrue(
                        {item.document_id for item in results} <= own_document_ids,
                        f"Retrieval crossed the exact asset boundary for {asset_code}.",
                    )
                    self.assertTrue(results.reranker_applied)
                    self.assertEqual(results.reranker_model, "servy_v5_bge_verified")
                    self.assertEqual(model.predict.call_count, 1)
                    self.assertEqual(
                        len(scored_pairs), 5,
                        "The serving path must keep the empirically selected V5 pool size of five.",
                    )
                    for _query, passage in scored_pairs:
                        for foreign_marker in foreign_markers:
                            self.assertNotIn(
                                foreign_marker,
                                passage,
                                f"V5 received another asset's source passage: {foreign_marker}",
                            )

    def test_each_asset_has_exactly_three_distinct_manual_types(self):
        for asset_code, _asset_name, _customer_key in ASSET_SPECS:
            rows = KnowledgeDocument.objects.filter(
                tenant=self.tenant, asset=self.assets[asset_code]
            )
            self.assertEqual(rows.count(), 3)
            self.assertEqual(
                set(rows.values_list("doc_type", flat=True)),
                set(DOC_TYPES),
            )
            self.assertEqual(
                sum(1 for code, _doc_type in self.target_chunks if code == asset_code),
                3,
            )
