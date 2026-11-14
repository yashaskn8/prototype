"""Document ingestion writes approved BGE vectors only (MiniLM retired).

Runs the real, locally cached BGE-small model offline; skipped when the pinned
model is not available on this host (no download is ever attempted).
"""
import os
import tempfile
from unittest import mock, skipUnless

from django.test import TestCase

from core.ml import embeddings, indexer
from core.models import KnowledgeChunk, KnowledgeDocument, Tenant


def _bge_available():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        embeddings.load_model(embeddings.get_profile("bge-small"))
        return True
    except Exception:
        return False


@skipUnless(_bge_available(), "pinned BGE-small model is not cached locally")
class BgeIngestionTests(TestCase):
    def setUp(self):
        import chromadb
        from core.services import chroma_store
        self.tmp = tempfile.TemporaryDirectory(prefix="servy_bge_ingest_", ignore_cleanup_errors=True)
        self.client = chromadb.PersistentClient(path=self.tmp.name)
        patcher = mock.patch.object(chroma_store, "_chroma_client", self.client)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tenant = Tenant.objects.create(name="Ingest", slug="ingest-test")
        self.doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, title="Chiller manual", content_text="x", is_rag_enabled=True,
            index_status="INDEXING", index_version=3)
        self.chunks = [
            KnowledgeChunk.objects.create(tenant=self.tenant, document=self.doc, chunk_index=i,
                                          heading=heading, text=text, embedding={})
            for i, (heading, text) in enumerate([
                ("Alarm E4", "Check the condenser fan and clean the coil."),
                ("Filter cleaning", "Remove the filter, wash it with water and let it dry."),
            ])
        ]

    def tearDown(self):
        self.client = None
        self.tmp.cleanup()

    def test_sync_writes_stamped_bge_vectors_with_section_identity(self):
        from core.services.chroma_store import get_document_retrieval_readiness, sync_document_to_chroma
        self.assertTrue(sync_document_to_chroma(self.doc.id))
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.index_status, "INDEXED")

        names = {c.name for c in self.client.list_collections()}
        self.assertEqual(names, {"servy_dense_bge_small_v1"})   # no MiniLM collection touched
        coll = self.client.get_collection("servy_dense_bge_small_v1")
        self.assertEqual(coll.metadata["servy_model"], "BAAI/bge-small-en-v1.5")
        got = coll.get(include=["metadatas", "embeddings", "documents"])
        self.assertEqual(len(got["ids"]), 2)
        rows = sorted(zip(got["metadatas"], got["embeddings"], got["documents"]), key=lambda r: r[0]["chunk_index"])
        for chunk, (meta, vector, document) in zip(self.chunks, rows):
            self.assertEqual(len(vector), 384)
            self.assertEqual(meta["chunk_id"], chunk.id)
            self.assertEqual(meta["chunk_index"], chunk.chunk_index)
            self.assertEqual(meta["document_id"], self.doc.id)
            self.assertEqual(meta["tenant_id"], self.tenant.id)
            self.assertEqual(meta["document_version"], 3)
            self.assertEqual(meta["embedding_model"], "BAAI/bge-small-en-v1.5")
            self.assertEqual(meta["embedding_revision"], embeddings.BGE_SMALL_REVISION)
            self.assertEqual(document, chunk.text)
        readiness = get_document_retrieval_readiness(self.doc)
        self.assertTrue(readiness["bge_vectors_ready"])
        self.assertNotIn("minilm_vectors_ready", readiness)

        # The same BGE configuration embeds queries: the relevant section ranks first.
        profile = embeddings.get_profile("bge-small")
        query = embeddings.encode_queries(profile, embeddings.load_model(profile), ["how do I clean the filter"])
        hit = coll.query(query_embeddings=query.tolist(), n_results=1, include=["metadatas"])
        self.assertEqual(hit["metadatas"][0][0]["chunk_id"], self.chunks[1].id)

    def test_missing_bge_model_fails_truthfully_without_any_fallback(self):
        from core.services.chroma_store import sync_document_to_chroma
        with mock.patch.object(indexer, "load_model",
                               side_effect=embeddings.EmbeddingConfigError("not available locally")):
            self.assertFalse(sync_document_to_chroma(self.doc.id))
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.index_status, "FAILED")
        self.assertIn("not available locally", self.doc.index_error)
        self.assertEqual([c.name for c in self.client.list_collections()], [])
