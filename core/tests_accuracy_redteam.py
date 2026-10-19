"""Accuracy red-team regressions: no real model, GPU, Docker or company data."""
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from core.ml import indexer
from core.ml.embeddings import EmbeddingConfigError
from core.services.retriever import _qwen_dense_first_merge


def candidate(cid, score, *, dense=False, sparse=False):
    return SimpleNamespace(
        source_kind="knowledge_document",
        source_id=cid, score=score,
        dense_score=score if dense else None,
        dense_similarity=(score - 0.15) if dense else None,
        dense_rank=cid if dense else None,
        sparse_score=score if sparse else None,
        sparse_relevance=score if sparse else None,
        sparse_rank=cid if sparse else None,
    )


class AccuracyRedTeamTests(SimpleTestCase):
    def test_dense_first_retains_raw_dense_rank_and_sparse_evidence(self):
        first = candidate(1, 0.89, dense=True)
        second = candidate(2, 0.76, dense=True)
        lex_same = candidate(2, 0.33, sparse=True)
        lex_only = candidate(3, 0.41, sparse=True)

        results = _qwen_dense_first_merge(
            [first, second], [lex_only, lex_same]
        )
        self.assertEqual([r.source_id for r in results], [1, 2, 3])
        self.assertAlmostEqual(second.dense_similarity, 0.61)
        self.assertEqual(second.sparse_relevance, 0.33)
        self.assertEqual(first.dense_rank, 1)
        self.assertIsNone(first.sparse_score)

    def test_dense_first_does_not_raise_lexical_match_above_dense(self):
        top = candidate(1, 0.81, dense=True)
        low_lex = candidate(2, 0.98, sparse=True)
        result = _qwen_dense_first_merge([top], [low_lex])
        self.assertEqual([r.source_id for r in result], [1, 2])

    def test_company_per_document_refresh_updates_valid_without_deleting_stale(self):
        profile = SimpleNamespace(
            key="company-qwen3",
            collection="servy_company_qwen3_06b_v1",
        )
        document = mock.Mock()
        document.id = 12
        document.is_rag_enabled = True
        document.index_status = "INDEXED"
        document.index_version = 1

        chunk = SimpleNamespace(
            id=17, tenant_id=4, document_id=12, chunk_index=0,
            text="Fresh authorized manual passage",
            heading="Maintenance",
            document=document,
        )
        document.chunks.filter.return_value.exclude.return_value.order_by.return_value = [chunk]
        coll = mock.Mock()
        coll.get.return_value = {
            "ids": ["stale-company-vector"], "metadatas": [{}],
        }
        model = mock.Mock()
        embeddings = mock.Mock()
        embeddings.tolist.return_value = [[0.0] * 1024]
        with mock.patch.object(indexer, "get_profile", return_value=profile), \
             mock.patch.object(indexer, "get_collection", return_value=coll), \
             mock.patch.object(indexer, "load_model", return_value=model), \
             mock.patch.object(indexer, "encode_passages", return_value=embeddings), \
             mock.patch.object(indexer, "_profile_fingerprint", return_value="fingerprint"), \
             mock.patch.object(indexer, "_expected_vector_metadata", return_value={}):
            indexed = indexer.index_single_document_bge(document, profile_key=profile.key)
        self.assertEqual(indexed, 1)
        coll.upsert.assert_called_once()
        coll.delete.assert_not_called()

    def test_company_bge_vector_deletion_also_prohibited(self):
        profile = SimpleNamespace(
            key="company-bge-small", collection="servy_company_bge_small_v1"
        )
        coll = mock.Mock()
        coll.get.return_value = {"ids": ["company-vector"]}
        with mock.patch.object(indexer, "get_profile", return_value=profile), \
             mock.patch.object(indexer, "get_collection", return_value=coll):
            with self.assertRaisesRegex(EmbeddingConfigError, "prohibited"):
                indexer.remove_document_bge(99, profile_key=profile.key)
        coll.delete.assert_not_called()
