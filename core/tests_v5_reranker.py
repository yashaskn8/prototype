"""V5 local-reranker opt-in, integrity, and authorization-boundary tests.

No model download, weights, GPU, or database required.
"""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase, override_settings

from core.ml import reranker
from core.services.retriever import _matches_current_minilm_vector


def chunk(name, kind="knowledge_document", score=0.3):
    return SimpleNamespace(source_kind=kind, text=name, score=score)


class V5RerankerTests(SimpleTestCase):
    def setUp(self):
        reranker._cross_encoder = None
        reranker._loaded_config = None

    def tearDown(self):
        reranker._cross_encoder = None
        reranker._loaded_config = None

    @override_settings(SERVY_RERANKER_ENABLED=False)
    def test_disabled_does_not_load_model(self):
        original = [chunk("A"), chunk("B")]
        with mock.patch.object(reranker, "get_reranker") as loader:
            self.assertEqual(reranker.rerank_candidates("test", original), original)
        loader.assert_not_called()

    def _checkpoint(self, parent):
        directory = Path(parent) / "best_model"
        directory.mkdir()
        (directory / "config.json").write_text('{"architectures":[]}', encoding="utf-8")
        weights = directory / "model.safetensors"
        weights.write_bytes(b"test-weights")
        return directory, hashlib.sha256(weights.read_bytes()).hexdigest()

    def test_local_weights_digest_and_offline_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, digest = self._checkpoint(tmp)
            fake = mock.Mock()
            cross_encoder = mock.Mock(return_value=fake)
            with override_settings(
                SERVY_RERANKER_MODEL_PATH=str(directory),
                SERVY_RERANKER_EXPECTED_SHA256=digest,
                SERVY_RERANKER_DEVICE="cpu",
            ), mock.patch.dict(
                "sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=cross_encoder)}
            ):
                self.assertIs(reranker.get_reranker(), fake)
                self.assertIs(reranker.get_reranker(), fake)
            cross_encoder.assert_called_once_with(
                str(directory), device="cpu", max_length=256, local_files_only=True
            )

    def test_reject_modified_local_weights_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = self._checkpoint(tmp)
            cross_encoder = mock.Mock()
            with override_settings(
                SERVY_RERANKER_MODEL_PATH=str(directory),
                SERVY_RERANKER_EXPECTED_SHA256="0" * 64,
            ), mock.patch.dict(
                "sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=cross_encoder)}
            ):
                self.assertIsNone(reranker.get_reranker())
            cross_encoder.assert_not_called()

    def test_missing_local_checkpoint_does_not_fall_back_to_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            cross_encoder = mock.Mock()
            with override_settings(SERVY_RERANKER_MODEL_PATH=str(Path(tmp) / "absent")), mock.patch.dict(
                "sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=cross_encoder)}
            ):
                self.assertIsNone(reranker.get_reranker())
            cross_encoder.assert_not_called()

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=5)
    def test_reranks_authorized_pool_without_scoring_history(self):
        docs = [chunk("A"), chunk("B"), chunk("C")]
        history = chunk("H", kind="service_resolution")
        fake = mock.Mock()
        fake.predict.return_value = [-2.0, 4.0, 0.0]
        with mock.patch.object(reranker, "get_reranker", return_value=fake):
            result = reranker.rerank_candidates("repair", docs + [history], top_k=4)
        self.assertEqual([x.text for x in result], ["B", "C", "A", "H"])
        fake.predict.assert_called_once_with([
            ["repair", "A"], ["repair", "B"], ["repair", "C"]
        ])
        self.assertFalse(hasattr(history, "rerank_score"))
        self.assertAlmostEqual(docs[1].reranker_score, reranker.calibrated_sigmoid_score(4.0))

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=5)
    def test_pool_size_is_bounded(self):
        docs = [chunk(str(i)) for i in range(7)]
        fake = mock.Mock()
        fake.predict.return_value = [0.0, 2.0, 1.0, 3.0, 4.0]
        with mock.patch.object(reranker, "get_reranker", return_value=fake):
            ranked = reranker.rerank_candidates("repair", docs, top_k=7)
        self.assertEqual(len(fake.predict.call_args.args[0]), 5)
        self.assertEqual(ranked[0].text, "4")
        self.assertEqual([x.text for x in ranked[-2:]], ["5", "6"])

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=5)
    def test_invalid_scores_restore_order_without_partial_mutation(self):
        docs = [chunk("A"), chunk("B")]
        fake = mock.Mock()
        fake.predict.return_value = [3.0, float("nan")]
        with mock.patch.object(reranker, "get_reranker", return_value=fake):
            result = reranker.rerank_candidates("repair", docs, top_k=2)
        self.assertEqual([x.text for x in result], ["A", "B"])
        self.assertFalse(hasattr(docs[0], "reranker_score"))

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=10)
    def test_explicit_ten_candidate_pool_can_promote_tenth_authorized_chunk(self):
        docs = [chunk(f"passage-{i}") for i in range(10)]
        model = mock.Mock()
        model.predict.return_value = [float(i) for i in range(10)]
        with mock.patch.object(reranker, "get_reranker", return_value=model):
            ranked = reranker.rerank_candidates("diagnose", docs, top_k=10)
        self.assertEqual(ranked[0].text, "passage-9")
        self.assertEqual(len(model.predict.call_args.args[0]), 10)
        self.assertEqual(len(ranked), 10)

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=5)
    def test_default_five_candidate_pool_does_not_promote_tenth_chunk(self):
        docs = [chunk(f"passage-{i}") for i in range(10)]
        model = mock.Mock()
        model.predict.return_value = [float(i) for i in range(5)]
        with mock.patch.object(reranker, "get_reranker", return_value=model):
            ranked = reranker.rerank_candidates("diagnose", docs, top_k=10)
        self.assertEqual(ranked[0].text, "passage-4")
        self.assertEqual(ranked[-1].text, "passage-9")
        self.assertEqual(len(model.predict.call_args.args[0]), 5)

    def test_pinned_v5_digest_requires_local_model_path(self):
        with override_settings(
            SERVY_RERANKER_ENABLED=True,
            SERVY_RERANKER_MODEL_PATH="",
            SERVY_RERANKER_EXPECTED_SHA256=reranker.SERVY_V5_WEIGHTS_SHA256,
        ), mock.patch.dict(
            "sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=mock.Mock())}
        ) as _:
            with mock.patch("sentence_transformers.CrossEncoder") as loader:
                self.assertIsNone(reranker.get_reranker())
                loader.assert_not_called()

    @override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_POOL_SIZE=99)
    def test_invalid_pool_size_falls_back(self):
        docs = [chunk("A")]
        with mock.patch.object(reranker, "get_reranker", return_value=mock.Mock()):
            self.assertEqual(reranker.rerank_candidates("test", docs), docs)

    def test_sigmoid_is_finite_for_extreme_logits(self):
        self.assertEqual(reranker.calibrated_sigmoid_score(-1000), 0)
        self.assertEqual(reranker.calibrated_sigmoid_score(1000), 1)

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_POOL_SIZE=5,
        SERVY_RERANKER_MODEL_PATH="best_model",
        SERVY_RERANKER_EXPECTED_SHA256=reranker.SERVY_V5_WEIGHTS_SHA256,
    )
    def test_v5_model_identity_attached_only_to_scored_chunks(self):
        docs = [chunk("valid passage"), chunk("other passage")]
        model = mock.Mock()
        model.predict.return_value = [2.0, 1.0]
        with mock.patch.object(reranker, "get_reranker", return_value=model):
            ranked = reranker.rerank_candidates("diagnostic", docs, top_k=2)
        self.assertTrue(all(item.reranker_model == "servy_v5_bge_verified" for item in ranked))
        self.assertEqual([item.text for item in ranked], ["valid passage", "other passage"])

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_POOL_SIZE=5,
        SERVY_RERANKER_MODEL_PATH="best_model",
        SERVY_RERANKER_EXPECTED_SHA256=reranker.SERVY_V5_WEIGHTS_SHA256,
    )
    def test_scoring_failure_does_not_create_v5_identity(self):
        docs = [chunk("first"), chunk("second")]
        model = mock.Mock()
        model.predict.return_value = [1.0, float("nan")]
        with mock.patch.object(reranker, "get_reranker", return_value=model):
            ranked = reranker.rerank_candidates("diagnostic", docs, top_k=2)
        self.assertEqual(ranked, docs)
        self.assertTrue(all(not hasattr(item, "reranker_model") for item in ranked))

    def test_retrieval_response_omits_unscored_model_identity(self):
        from core.services.retriever import RetrievedResultList
        self.assertIsNone(RetrievedResultList(reranker_model="servy_v5_bge_verified").reranker_model)
        self.assertEqual(
            RetrievedResultList(reranker_applied=True, reranker_model="servy_v5_bge_verified").reranker_model,
            "servy_v5_bge_verified",
        )

    def test_stale_minilm_vector_cannot_supply_a_v5_candidate(self):
        source = SimpleNamespace(
            id=7, document_id=5, chunk_index=0, tenant_id=2,
            text="Current authorized manual passage",
            document=SimpleNamespace(index_version=3),
        )
        profile = SimpleNamespace(model_id="all-MiniLM-L6-v2")
        metadata = {
            "chunk_id": 7, "document_id": 5, "chunk_index": 0,
            "tenant_id": 2, "document_version": "3",
            "content_sha256": hashlib.sha256(source.text.encode("utf-8")).hexdigest(),
            "embedding_model": profile.model_id,
        }
        self.assertTrue(_matches_current_minilm_vector(metadata, source, profile))
        self.assertFalse(_matches_current_minilm_vector(
            {**metadata, "content_sha256": hashlib.sha256(b"stale passage").hexdigest()},
            source, profile,
        ))
        self.assertFalse(_matches_current_minilm_vector(
            {**metadata, "chunk_id": 999}, source, profile,
        ))
