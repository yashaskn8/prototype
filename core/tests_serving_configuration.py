"""Serving configuration is declared, validated and free of legacy preloads.

Settings are evaluated in a fresh interpreter for each case, because they are
read once at import. No model, index or database is loaded.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

from django.apps import apps
from django.test import SimpleTestCase

ROOT = Path(__file__).resolve().parent.parent


def _settings_value(name, **env):
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("SERVY_RETRIEVAL")}
    child_env.update({"DJANGO_SETTINGS_MODULE": "servy_rag.settings", "SERVY_DB_ENGINE": "sqlite3",
                      "DJANGO_DEBUG": "1", **env})
    return subprocess.run(
        [sys.executable, "-c",
         "import django; from django.conf import settings; django.setup(); "
         f"print(settings.{name})"],
        cwd=ROOT, env=child_env, capture_output=True, text=True, timeout=120,
    )


class RetrievalModeSettingTests(SimpleTestCase):
    def test_retrieval_mode_is_a_declared_setting_with_the_evaluated_default(self):
        result = _settings_value("SERVY_RETRIEVAL_MODE")
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "hybrid")

    def test_environment_value_is_honoured_not_silently_ignored(self):
        result = _settings_value("SERVY_RETRIEVAL_MODE", SERVY_RETRIEVAL_MODE="Sparse_Only ")
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "sparse_only")

    def test_unknown_retrieval_mode_fails_at_startup(self):
        result = _settings_value("SERVY_RETRIEVAL_MODE", SERVY_RETRIEVAL_MODE="hybird")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ImproperlyConfigured", result.stderr)
        self.assertIn("hybird", result.stderr)

    def test_serving_config_reports_the_declared_mode(self):
        from core.services.retriever import get_serving_config
        with self.settings(SERVY_RETRIEVAL_MODE="sparse_only"):
            self.assertEqual(get_serving_config().retrieval_mode, "sparse_only")
        with self.settings(SERVY_RETRIEVAL_MODE="hybrid"):
            self.assertEqual(get_serving_config().retrieval_mode, "hybrid")


class StartupModelLoadingTests(SimpleTestCase):
    def test_app_startup_does_not_load_the_legacy_minilm_model(self):
        config = apps.get_app_config("core")
        with mock.patch("core.services.chroma_store.get_sentence_transformer",
                        side_effect=AssertionError("legacy MiniLM must not load at startup")) as legacy, \
                mock.patch.object(threading.Thread, "start",
                                  side_effect=AssertionError("no background model thread")):
            config.ready()
        legacy.assert_not_called()

    def test_strict_v5_startup_check_is_still_enforced_when_required(self):
        config = apps.get_app_config("core")
        with self.settings(SERVY_RERANKER_REQUIRED=True), \
                mock.patch("core.ml.reranker.require_verified_v5_reranker",
                           side_effect=RuntimeError("V5 unavailable")) as required:
            with self.assertRaises(RuntimeError):
                config.ready()
        required.assert_called_once_with()


class StrictCheckIndexIsolationTests(SimpleTestCase):
    """The strict serving check reads a snapshot, never the configured index."""

    def _tree(self, root):
        import hashlib
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(root).rglob("*")) if p.is_file()}

    def test_opening_collections_leaves_the_configured_index_unchanged(self):
        import tempfile

        import chromadb

        from core.management.commands.check_rag_serving import index_snapshot
        from core.services import chroma_store

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "index"
            writer = chromadb.PersistentClient(path=str(source))
            writer.create_collection("probe").add(ids=["a", "b"], embeddings=[[0.0, 1.0], [1.0, 0.0]])
            writer._system.stop()
            del writer
            before = self._tree(source)
            sentinel = object()
            with self.settings(CHROMA_PERSIST_DIR=source), \
                    mock.patch.object(chroma_store, "_chroma_client", sentinel):
                with index_snapshot() as snapshot:
                    self.assertNotEqual(Path(snapshot), source)
                    client = chroma_store.get_chroma_client()
                    collection = client.get_collection("probe")
                    self.assertEqual(collection.count(), 2)
                    collection.query(query_embeddings=[[0.0, 1.0]], n_results=1)
                self.assertIs(chroma_store._chroma_client, sentinel)
            self.assertEqual(self._tree(source), before)

    def test_missing_index_is_not_created(self):
        import tempfile

        from core.management.commands.check_rag_serving import index_snapshot
        from core.services import chroma_store

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent"
            with self.settings(CHROMA_PERSIST_DIR=missing), \
                    mock.patch.object(chroma_store, "_chroma_client", None):
                with index_snapshot():
                    client = chroma_store.get_chroma_client()
                    self.assertEqual(list(client.list_collections()), [])
            self.assertFalse(missing.exists())
