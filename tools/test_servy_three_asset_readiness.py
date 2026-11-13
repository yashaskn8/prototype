"""Offline tests for three-asset / nine-manual readiness and the serving gate.

Synthetic records only: no database, model, index or company document is
used. These tests show the gates behave correctly; they are not evidence that
the real documents or models are ready.
"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("LOCALAPPDATA", "/tmp")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from core.management.commands import check_rag_serving as gate
from tools import servy_real_demo as demo
from tools import servy_three_asset_rag_audit as audit
from tools import servy_three_asset_scope as scope


class RequiredBindingTests(unittest.TestCase):
    def test_exact_asset_customer_and_nine_manual_bindings(self):
        self.assertEqual(
            {asset: (row["customer_id"], tuple(sorted(row["documents"])))
             for asset, row in scope.REQUIRED_MANUALS.items()},
            {59: (14, (168, 169, 170)), 60: (13, (171, 172, 173)), 61: (14, (174, 175, 176))})
        self.assertEqual(scope.REQUIRED_DOCUMENT_COUNT, 9)
        for row in scope.REQUIRED_MANUALS.values():
            self.assertEqual(set(row["documents"].values()),
                             {"troubleshooting", "maintenance", "installation"})

    def test_demo_and_audit_use_the_same_bindings(self):
        self.assertEqual(demo.ASSET_SCOPE, {a: r["customer_id"]
                                            for a, r in scope.REQUIRED_MANUALS.items()})
        self.assertEqual(audit.DOCUMENTS_BY_ASSET, {a: r["documents"]
                                                    for a, r in scope.REQUIRED_MANUALS.items()})
        self.assertEqual({a: r["customer_id"] for a, r in audit.ASSET_SCOPE.items()},
                         demo.ASSET_SCOPE)


class DocumentReadinessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.media = Path(self.folder.name)
        self.path = self.media / "manual.txt"
        self.path.write_bytes(b"synthetic manual body")
        self.digest = hashlib.sha256(b"synthetic manual body").hexdigest()

    def _doc(self, **overrides):
        values = dict(asset_id=60, customer_id=None, is_rag_enabled=True, is_confidential=False,
                      disable_sharing=False, checksum_sha256=self.digest,
                      file=SimpleNamespace(path=str(self.path)), title="Synthetic manual")
        values.update(overrides)
        return SimpleNamespace(**values)

    def _issues(self, doc, **overrides):
        values = dict(asset_id=60, customer_id=13, doc_type="troubleshooting",
                      registry_entry={"local_asset_id": 60, "doc_type": "troubleshooting",
                                      "registered_checksum": self.digest},
                      media_root=self.media, searchable_chunks=6, is_verified_document=True)
        values.update(overrides)
        return scope.document_issues(doc, **values)

    def test_fully_verified_manual_is_ready(self):
        self.assertEqual(self._issues(self._doc()), [])

    def test_a_database_record_alone_is_not_readiness(self):
        # The record exists and is flagged eligible, but nothing else holds.
        doc = self._doc(file=None, checksum_sha256="")
        issues = self._issues(doc, registry_entry=None, searchable_chunks=0,
                              is_verified_document=False)
        self.assertEqual(set(issues), {
            "PROVENANCE_NOT_VERIFIED", "NOT_IN_PROVENANCE_REGISTRY",
            "ATTACHMENT_OR_CHECKSUM_MISSING", "NO_SEARCHABLE_CHUNKS"})

    def test_each_failed_check_is_reported_by_name(self):
        cases = (
            (None, {}, "DOCUMENT_RECORD_MISSING"),
            (self._doc(asset_id=61), {}, "NOT_BOUND_TO_REQUIRED_ASSET"),
            (self._doc(customer_id=14), {}, "BOUND_TO_ANOTHER_CUSTOMER"),
            (self._doc(is_confidential=True), {}, "NOT_ELIGIBLE_FOR_CUSTOMER_RAG"),
            (self._doc(disable_sharing=True), {}, "NOT_ELIGIBLE_FOR_CUSTOMER_RAG"),
            (self._doc(is_rag_enabled=False), {}, "NOT_ELIGIBLE_FOR_CUSTOMER_RAG"),
            (self._doc(), {"is_verified_document": False}, "PROVENANCE_NOT_VERIFIED"),
            (self._doc(), {"doc_type": "maintenance"}, "DOCUMENT_TYPE_MISMATCH"),
            (self._doc(checksum_sha256="0" * 64), {}, "ATTACHMENT_CHECKSUM_MISMATCH"),
            (self._doc(), {"registry_entry": {"local_asset_id": 60, "doc_type": "troubleshooting",
                                              "registered_checksum": "f" * 64}},
             "REGISTRY_CHECKSUM_MISMATCH"),
            (self._doc(), {"registry_entry": {"local_asset_id": 59, "doc_type": "troubleshooting",
                                              "registered_checksum": self.digest}},
             "REGISTRY_ASSET_MISMATCH"),
            (self._doc(file=SimpleNamespace(path=str(self.media / "missing.txt"))), {},
             "ATTACHMENT_FILE_UNAVAILABLE"),
            (self._doc(), {"searchable_chunks": 0}, "NO_SEARCHABLE_CHUNKS"),
        )
        for doc, overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self._issues(doc, **overrides))

    def test_attachment_outside_the_media_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            other = Path(elsewhere) / "manual.txt"
            other.write_bytes(b"synthetic manual body")
            doc = self._doc(file=SimpleNamespace(path=str(other)))
            self.assertIn("ATTACHMENT_FILE_UNAVAILABLE", self._issues(doc))

    def test_failure_report_names_assets_and_documents_without_paths(self):
        report = {
            59: {"asset_issues": [], "documents": [
                {"id": 168, "doc_type": "troubleshooting", "ready": True, "issues": []},
                {"id": 169, "doc_type": "maintenance", "ready": False,
                 "issues": ["NO_SEARCHABLE_CHUNKS"]},
                {"id": 170, "doc_type": "installation", "ready": True, "issues": []}]},
            60: {"asset_issues": ["ASSET_CUSTOMER_BINDING_MISMATCH"], "documents": []},
        }
        lines = scope.readiness_failures(report)
        self.assertIn("asset 59 document 169 (maintenance): NO_SEARCHABLE_CHUNKS", lines)
        self.assertIn("asset 60: ASSET_CUSTOMER_BINDING_MISMATCH", lines)
        self.assertIn("asset 61: readiness was not evaluated", lines)
        self.assertFalse(any(self.folder.name in line or "\\" in line for line in lines))


class QueryGateTests(unittest.TestCase):
    def test_asset_with_incomplete_manuals_is_refused_before_any_lookup(self):
        readiness = {59: {"ready": True}, 60: {"ready": False}, 61: {"ready": True}}
        with patch.object(demo, "MANUAL_READINESS", readiness), \
                patch.object(demo, "_asset", side_effect=AssertionError("no lookup")):
            result = demo._answer(60, "Why is there a communication fault?")
        self.assertEqual(result["reason_code"], "REQUIRED_MANUALS_NOT_READY")
        self.assertEqual(result["citations"], [])
        self.assertIsNone(result["presentation"])

    def test_unevaluated_asset_is_not_treated_as_ready(self):
        with patch.object(demo, "MANUAL_READINESS", {59: {"ready": True}}), \
                patch.object(demo, "_asset", side_effect=AssertionError("no lookup")):
            result = demo._answer(61, "What causes weak airflow?")
        self.assertEqual(result["reason_code"], "REQUIRED_MANUALS_NOT_READY")


class StrictServingGateTests(unittest.TestCase):
    def _run(self, *, dim=384, vector=384, counts=None, v5=None, required=None, optional=()):
        counts = counts if counts is not None else {"bge-small": 10, "company-bge-small": 64}
        profiles = {key: SimpleNamespace(key=key, dim=dim) for key in
                    ("bge-small", "company-bge-small")}

        def collection(profile, create=True):
            self.assertFalse(create)                      # the gate never creates an index
            count = counts.get(profile.key)
            if count is None:
                return None
            return SimpleNamespace(count=lambda: count)

        with patch("core.ml.embeddings.get_profile", side_effect=profiles.__getitem__), \
                patch("core.ml.embeddings.load_model", return_value=object()) as load, \
                patch("core.ml.embeddings.encode_passages", return_value=[[0.0] * vector]), \
                patch("core.ml.embeddings.get_collection", side_effect=collection), \
                patch("core.ml.reranker.require_verified_v5_reranker", side_effect=v5) as scorer:
            kwargs = {"optional_collections": optional}
            if required is not None:
                kwargs["required_collections"] = required
            result = gate.verify_local_serving_artifacts(**kwargs)
        for call in load.call_args_list:
            self.assertFalse(call.kwargs.get("allow_download", True))   # never downloads
        return result, scorer

    def test_verified_stack_reports_dimensions_collections_and_v5(self):
        result, scorer = self._run()
        self.assertEqual(result["embedding_dimensions"],
                         {"bge-small": 384, "company-bge-small": 384})
        self.assertEqual(result["collections"], {"bge-small": 10, "company-bge-small": 64})
        self.assertEqual(result["v5"]["inference"], "verified")
        self.assertEqual(result["v5"]["weights_sha256"], gate.SERVY_V5_WEIGHTS_SHA256)
        self.assertEqual(scorer.call_count, 1)

    def test_wrong_embedding_dimension_fails(self):
        with self.assertRaisesRegex(gate.ServingArtifactError, "expected 384"):
            self._run(vector=768)

    def test_missing_or_empty_required_collection_fails(self):
        for counts in ({"bge-small": 10}, {"bge-small": 10, "company-bge-small": 0}):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(gate.ServingArtifactError, "company-bge-small"):
                    self._run(counts=counts)

    def test_optional_collection_is_reported_not_assumed(self):
        result, _ = self._run(counts={"company-bge-small": 64},
                              required=("company-bge-small",), optional=("bge-small",))
        self.assertEqual(result["collections"],
                         {"company-bge-small": 64, "bge-small": "NOT_PRESENT"})

    def test_v5_fallback_or_failure_is_never_reported_as_verified(self):
        with self.assertRaisesRegex(gate.ServingArtifactError, "V5 reranker did not run"):
            self._run(v5=RuntimeError("C:\\private\\model path must not leak"))
        try:
            self._run(v5=RuntimeError("C:\\private\\model path must not leak"))
        except gate.ServingArtifactError as exc:
            self.assertNotIn("private", str(exc))


if __name__ == "__main__":
    unittest.main()
