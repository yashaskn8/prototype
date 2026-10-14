"""Tests for the experimental embedding pipeline.

All fixtures are synthetic and test MECHANICS (identity, isolation, gating,
validation). They make no claim about retrieval accuracy on company data.
"""
import uuid
from unittest import mock

from django.test import TestCase, override_settings

from core.ml import embeddings as E
from core.ml.labels import validate_labels, split_by_group
from core.models import KnowledgeChunk, KnowledgeDocument, Tenant


def _model_available():
    try:
        E.load_model(E.get_profile("bge-small"))
        return True
    except Exception:
        return False


class RegistryTests(TestCase):
    def test_profiles_use_distinct_collections(self):
        cols = [p.collection for p in E.PROFILES.values()]
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(E.get_profile("minilm").collection, "servy_dense_kb")
        self.assertNotEqual(E.get_profile("bge-small").collection, "servy_dense_kb")

    def test_unknown_profile_rejected(self):
        with self.assertRaises(E.EmbeddingConfigError):
            E.get_profile("nope")

    def test_bge_revision_pinned(self):
        self.assertEqual(E.get_profile("bge-small").revision, "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")

    def test_experimental_off_by_default(self):
        from django.conf import settings
        self.assertFalse(settings.SERVY_USE_EXPERIMENTAL_RAG)
        self.assertEqual(settings.SERVY_EMBEDDING_MODEL, "minilm")

    def test_collection_stamped_by_other_model_rejected(self):
        fake = mock.MagicMock(metadata={"servy_model": "all-MiniLM-L6-v2"})
        client = mock.MagicMock(get_or_create_collection=mock.MagicMock(return_value=fake))
        with mock.patch("core.services.chroma_store.get_chroma_client", return_value=client):
            with self.assertRaises(E.EmbeddingConfigError):
                E.get_collection(E.get_profile("bge-small"))

    def test_unstamped_collection_rejected_for_bge(self):
        fake = mock.MagicMock(metadata={"hnsw:space": "cosine"})
        client = mock.MagicMock(get_or_create_collection=mock.MagicMock(return_value=fake))
        with mock.patch("core.services.chroma_store.get_chroma_client", return_value=client):
            with self.assertRaises(E.EmbeddingConfigError):
                E.get_collection(E.get_profile("bge-small"))

    def test_indexer_refuses_legacy_collection(self):
        from core.ml.indexer import build_index
        with mock.patch.dict(E.PROFILES, {"evil": E.EmbeddingProfile("evil", "x", None, 384, "servy_dense_kb")}):
            with self.assertRaises(E.EmbeddingConfigError):
                build_index("evil")

    def test_missing_model_gives_clear_error(self):
        p = E.EmbeddingProfile("ghost", "nonexistent/model-xyz", None, 384, "servy_dense_ghost")
        with self.assertRaises(E.EmbeddingConfigError):
            E.load_model(p)


class ContentFirstTests(TestCase):
    def setUp(self):
        self.t = Tenant.objects.create(name="T", slug="t-" + uuid.uuid4().hex[:6])
        self.doc = KnowledgeDocument.objects.create(
            tenant=self.t, title="upload_final_v2.pdf", description="", tags="CATEGORY_TAG_WRONG",
            content_text="x", is_rag_enabled=True)
        self.chunk = KnowledgeChunk.objects.create(
            tenant=self.t, document=self.doc, chunk_index=0, heading="Sample intake",
            text="If the analyzer does not draw a sample, inspect the intake tube.", embedding={})

    def test_bge_passage_excludes_filename_and_tags(self):
        txt = E.passage_text(E.get_profile("bge-small"), self.chunk)
        self.assertNotIn("upload_final_v2", txt)
        self.assertNotIn("CATEGORY_TAG_WRONG", txt)
        self.assertIn("intake tube", txt)

    def test_minilm_passage_unchanged_legacy_behaviour(self):
        txt = E.passage_text(E.get_profile("minilm"), self.chunk)
        self.assertIn("upload_final_v2", txt)


class LabelValidationTests(TestCase):
    def setUp(self):
        self.t1 = Tenant.objects.create(name="A", slug="a-" + uuid.uuid4().hex[:6])
        self.t2 = Tenant.objects.create(name="B", slug="b-" + uuid.uuid4().hex[:6])
        d1 = KnowledgeDocument.objects.create(tenant=self.t1, title="d1", content_text="x", is_rag_enabled=True)
        d2 = KnowledgeDocument.objects.create(tenant=self.t2, title="d2", content_text="x", is_rag_enabled=True)
        self.c1 = KnowledgeChunk.objects.create(tenant=self.t1, document=d1, chunk_index=0, text="a", embedding={})
        self.c1b = KnowledgeChunk.objects.create(tenant=self.t1, document=d1, chunk_index=1, text="b", embedding={})
        self.c2 = KnowledgeChunk.objects.create(tenant=self.t2, document=d2, chunk_index=0, text="c", embedding={})
        self.cq = KnowledgeChunk.objects.create(tenant=self.t1, document=d1, chunk_index=2, text="q",
                                                embedding={}, is_quarantined=True)

    def _row(self, **kw):
        r = {"query_id": uuid.uuid4().hex, "query_text": "q", "tenant_id": self.t1.id,
             "positive_chunk_ids": [self.c1.id], "negative_chunk_ids": [], "review_status": "approved"}
        r.update(kw); return r

    def test_rejections(self):
        rows = [
            self._row(),
            self._row(review_status="candidate"),
            self._row(query_text="  "),
            self._row(positive_chunk_ids=[]),
            self._row(negative_chunk_ids=[self.c1.id]),
            self._row(positive_chunk_ids=[self.c2.id]),
            self._row(positive_chunk_ids=[999999]),
            self._row(positive_chunk_ids=[self.cq.id]),
        ]
        out = validate_labels(rows)
        self.assertEqual(len(out["valid"]), 1)
        reasons = " ".join(r for _, r in out["rejected"])
        for s in ["not human-approved", "empty query", "no positive", "contradictory",
                  "cross-tenant", "missing", "quarantined"]:
            self.assertIn(s, reasons)

    def test_group_split_no_leakage(self):
        rows = [self._row(source_group=f"g{i % 7}") for i in range(70)]
        s = split_by_group(rows)
        groups = {k: {r["source_group"] for r in v} for k, v in s.items()}
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_training_refuses_unapproved(self):
        from core.ml.training import finetune
        with self.assertRaises(ValueError):
            finetune([{"query_text": "q", "positive_text": "p", "review_status": "candidate"}],
                     [], "unused", "rev")


class ModelSmokeTests(TestCase):
    def test_bge_embedding_shape_and_norm(self):
        if not _model_available():
            self.skipTest("BGE-small not downloaded locally")
        import numpy as np
        p = E.get_profile("bge-small")
        v = E.encode_queries(p, E.load_model(p), ["pump blockage", "gate interlock"])
        self.assertEqual(v.shape, (2, 384))
        self.assertTrue(np.isfinite(v).all())
        self.assertTrue(np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4))
