"""Offline Qwen3 retrieval/encoder safeguards. No model weights or database needed."""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase, override_settings

from core.ml import embeddings, indexer


class Qwen3AccuracySafetyTests(SimpleTestCase):
    @override_settings(SERVY_QWEN3_MODEL_PATH="")
    def test_missing_local_model_is_blocked(self):
        profile = embeddings.get_profile("qwen3")
        self.assertEqual(profile.dim, 1024)
        self.assertNotEqual(
            profile.collection, embeddings.get_profile("minilm").collection
        )
        with self.assertRaisesRegex(embeddings.EmbeddingConfigError, "requires"):
            embeddings.load_model(profile)

    @override_settings(SERVY_QWEN3_MODEL_PATH="/tmp/servy-qwen3-test")
    def test_company_index_and_demo_index_do_not_share_vectors(self):
        company = embeddings.get_profile("company-qwen3")
        demo = embeddings.get_profile("qwen3")
        self.assertNotEqual(company.collection, demo.collection)
        self.assertEqual(company.model_id, demo.model_id)
        self.assertTrue(company.query_document_encoder)
        self.assertTrue(demo.query_document_encoder)

    def test_qwen_query_and_document_encoding_are_distinct(self):
        with override_settings(SERVY_QWEN3_MODEL_PATH="models/qwen"):
            profile = embeddings.get_profile("qwen3")
        model = mock.Mock()
        model.encode_query.return_value = "query_vectors"
        model.encode_document.return_value = "document_vectors"
        self.assertEqual(embeddings.encode_queries(profile, model, ["question"]), "query_vectors")
        self.assertEqual(embeddings.encode_passages(profile, model, ["passage"]), "document_vectors")
        model.encode_query.assert_called_once()
        model.encode_document.assert_called_once()
        model.encode.assert_not_called()

    def test_legacy_encode_stays_unchanged(self):
        profile = embeddings.get_profile("minilm")
        model = mock.Mock()
        embeddings.encode_queries(profile, model, ["question"])
        model.encode.assert_called_once()
        self.assertEqual(model.encode.call_args.args[0], ["question"])

    def test_qwen3_manifest_refuses_unstamped_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text("{}", encoding="utf-8")
            with override_settings(SERVY_QWEN3_MODEL_PATH=tmp):
                profile = embeddings.get_profile("qwen3")
                coll = mock.Mock(metadata={"hnsw:space": "cosine"})
                client = mock.Mock()
                client.get_collection.return_value = coll
                with mock.patch("core.services.chroma_store.get_chroma_client", return_value=client):
                    with self.assertRaises(embeddings.EmbeddingConfigError):
                        embeddings.get_collection(profile, create=False)

    def test_qwen3_load_requires_query_and_document_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            with override_settings(SERVY_QWEN3_MODEL_PATH=tmp):
                profile = embeddings.get_profile("qwen3")
                instance = mock.Mock(spec=["get_sentence_embedding_dimension", "encode"])
                instance.get_sentence_embedding_dimension.return_value = 1024
                with mock.patch("sentence_transformers.SentenceTransformer", return_value=instance):
                    with self.assertRaisesRegex(embeddings.EmbeddingConfigError, "encode_query"):
                        embeddings.load_model(profile)

    def test_indexer_uses_qwen_document_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            with override_settings(SERVY_QWEN3_MODEL_PATH=tmp):
                profile = embeddings.get_profile("qwen3")
                doc = SimpleNamespace(
                    index_version=1, is_confidential=False,
                    customer_id=None, product_id=None, asset_id=None
                )
                chunk = SimpleNamespace(
                    id=11, tenant_id=7, document_id=9,
                    chunk_index=0, heading="Power", text="Check inlet pressure",
                    document=doc,
                )
                coll = mock.Mock()
                coll.get.return_value = {"ids": [], "metadatas": []}
                coll.count.return_value = 1
                embedding = mock.Mock()
                embedding.tolist.return_value = [[0.0] * 1024]
                model = mock.Mock()
                with mock.patch.object(indexer, "get_profile", return_value=profile), \
                     mock.patch.object(indexer, "load_model", return_value=model), \
                     mock.patch.object(indexer, "get_collection", return_value=coll), \
                     mock.patch.object(indexer, "_eligible_chunks", return_value=[chunk]), \
                     mock.patch.object(indexer, "encode_passages", return_value=embedding) as encode:
                    report = indexer.build_index("qwen3", limit=1)
                self.assertEqual(report["embedded_now"], 1)
                encode.assert_called_once_with(profile, model, ["Power\nCheck inlet pressure"])
                coll.upsert.assert_called_once()
                coll.delete.assert_not_called()

    def test_company_qwen_full_reconcile_preserves_stale_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            with override_settings(SERVY_QWEN3_MODEL_PATH=tmp):
                profile = embeddings.get_profile("company-qwen3")
                coll = mock.Mock()
                coll.get.return_value = {"ids": ["stale"], "metadatas": [{}]}
                coll.count.return_value = 1
                with mock.patch.object(indexer, "get_profile", return_value=profile), \
                     mock.patch.object(indexer, "load_model", return_value=mock.Mock()), \
                     mock.patch.object(indexer, "get_collection", return_value=coll), \
                     mock.patch.object(indexer, "_eligible_chunks", return_value=[]):
                    report = indexer.build_index("company-qwen3")
                self.assertEqual(report["protected_stale_count"], 1)
                self.assertEqual(report["removed_stale"], 0)
                coll.delete.assert_not_called()
