"""Regression checks for checkpoint and chunk identity guards."""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from core.ml.embeddings import (
    EmbeddingConfigError, EmbeddingProfile, compute_checkpoint_fingerprint,
    get_collection, load_model,
)
from core.ml.indexer import _expected_vector_metadata, _metadata_matches, reconcile_index
from core.services.retriever import _matches_current_bge_vector


class CheckpointIdentityTests(SimpleTestCase):
    def test_collection_rejects_changed_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "weights.bin").write_bytes(b"current")
            profile = EmbeddingProfile(
                key="op21-test", model_id=directory, revision=None, dim=384,
                collection="op21-test", local_path=directory, content_first=True,
            )
            collection = SimpleNamespace(metadata={
                "servy_model": directory, "servy_checkpoint_fp": "old-checkpoint",
            })
            client = Mock()
            client.get_collection.return_value = collection
            with patch("core.services.chroma_store.get_chroma_client", return_value=client):
                with self.assertRaisesRegex(EmbeddingConfigError, "checkpoint fingerprint mismatch"):
                    get_collection(profile, create=False)
                collection.metadata["servy_checkpoint_fp"] = compute_checkpoint_fingerprint(directory)
                self.assertIs(get_collection(profile, create=False), collection)

    def test_model_cache_changes_when_profile_or_checkpoint_changes(self):
        from core.ml import embeddings

        fake_model = Mock()
        fake_model.get_embedding_dimension.return_value = 384
        original = embeddings._models
        embeddings._models = {}
        try:
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                Path(a, "weights.bin").write_bytes(b"a1")
                Path(b, "weights.bin").write_bytes(b"b1")
                first = EmbeddingProfile("same-key", a, None, 384, "unused", local_path=a)
                second = EmbeddingProfile("same-key", b, None, 384, "unused", local_path=b)
                with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as loader:
                    load_model(first)
                    load_model(first)
                    load_model(second)
                    Path(a, "weights.bin").write_bytes(b"a2")
                    load_model(first)
                    self.assertEqual(loader.call_count, 3)
        finally:
            embeddings._models = original


class VectorIdentityTests(SimpleTestCase):
    def setUp(self):
        self.document = SimpleNamespace(
            index_version=2, is_confidential=False, customer_id=None,
            product_id=None, asset_id=7,
        )
        self.chunk = SimpleNamespace(
            id=41, tenant_id=9, document_id=12, chunk_index=1,
            heading="Alarm", text="Check the controller status.",
            document=self.document,
        )
        self.profile = EmbeddingProfile(
            "test-bge", "BAAI/bge-small-en-v1.5", "revision-1", 384,
            "test-collection", content_first=True,
        )
        text = "Alarm\nCheck the controller status."
        self.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_stale_chunk_at_same_document_position_is_rejected(self):
        metadata = _expected_vector_metadata(self.chunk, self.profile, self.content_hash, "revision-1")
        self.assertTrue(_matches_current_bge_vector(metadata, self.chunk, self.profile))
        for change in (
            {"chunk_id": 40}, {"document_version": 1},
            {"content_sha256": "old-content"}, {"tenant_id": 6},
            {"embedding_model": "another-model"},
        ):
            with self.subTest(change=change):
                self.assertFalse(_matches_current_bge_vector(metadata | change, self.chunk, self.profile))

    def test_reconciliation_reports_stale_metadata_even_when_id_exists(self):
        expected = _expected_vector_metadata(self.chunk, self.profile, self.content_hash, "revision-1")
        self.assertFalse(_metadata_matches(expected | {"content_sha256": "old"}, expected))
        collection = Mock()
        collection.get.return_value = {
            "ids": ["t9_doc_12_v2_c1"],
            "metadatas": [expected | {"content_sha256": "old"}],
        }
        with (patch("core.ml.indexer.get_profile", return_value=self.profile),
              patch("core.ml.indexer.get_collection", return_value=collection),
              patch("core.ml.indexer._eligible_chunks", return_value=[self.chunk])):
            result = reconcile_index("test-bge", dry_run=True)
        self.assertEqual(result["missing_count"], 0)
        self.assertEqual(result["metadata_mismatch_count"], 1)
        self.assertFalse(result["is_consistent"])
        collection.delete.assert_not_called()
        collection.upsert.assert_not_called()
