"""Offline tests for exhaustive approved-document scanning."""
import os

# unittest.mock.patch("core.services....") imports the target module while
# resolving the patch string. Initialize Django first so model imports are
# valid when this standalone module runs under GitHub Actions.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("LOCALAPPDATA", "/tmp")

import django
django.setup()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import servy_real_demo_exhaustive as scan


class _DocumentQuery:
    def __init__(self, docs):
        self.docs = docs

    def filter(self, **_kwargs):
        return self.docs


class ExhaustiveApprovedDocumentScanTests(unittest.TestCase):
    def _fixture(self, asset_name):
        tenant = SimpleNamespace(id=1)
        customer = SimpleNamespace(id=10)
        asset = SimpleNamespace(
            id=60, name=asset_name, model_number="MODEL-60",
            product=SimpleNamespace(name="HVAC", category=SimpleNamespace(name="HVAC")),
        )
        chunks, docs = [], []
        for doc_id in (101, 102, 103):
            document = SimpleNamespace(
                id=doc_id, tenant_id=1, title="Approved manual " + str(doc_id),
                is_confidential=False, disable_sharing=False, is_rag_enabled=True,
            )
            docs.append(document)
            chunks.append(SimpleNamespace(
                id=doc_id * 10, document_id=doc_id, tenant_id=1,
                text="Operating condition described in manual " + str(doc_id) + ".",
                heading="Troubleshooting", document=document, is_quarantined=False,
            ))
        return tenant, customer, asset, docs, chunks

    def test_each_demo_asset_family_scans_each_approved_manual(self):
        for name in ("Split AC Unit 2T", "VRF Outdoor Module", "Ductable AC 5T"):
            with self.subTest(asset=name):
                tenant, customer, asset, docs, chunks = self._fixture(name)
                ranked = [
                    SimpleNamespace(document_id=c.document_id, chunk_id=c.id, score=0.5)
                    for c in chunks
                ]
                with patch("core.services.knowledge_scope.get_applicable_knowledge_queryset",
                           return_value=_DocumentQuery(docs)), \
                     patch("core.services.retriever._knowledge_candidates", return_value=chunks), \
                     patch("core.services.retriever._sparse_score_candidates", return_value=ranked) as scoring, \
                     patch("core.services.retriever.get_serving_config", return_value=object()), \
                     patch("core.services.vectors.embed_text", side_effect=lambda _text: {"v": 1}), \
                     patch("core.services.identifiers.extract_identifiers", return_value=set()), \
                     patch("core.services.customer_isolation.is_verified_real_document", return_value=True), \
                     patch("core.services.content_safety.detect_document_prompt_injection", return_value=False):
                    result = scan.search_all_approved_documents(
                        tenant=tenant, customer=customer, asset=asset,
                        question=name + " not working",
                        original_complaint="The asset is not working",
                    )
                self.assertEqual(scoring.call_args.args[0], chunks)
                self.assertEqual({row["document_id"] for row in result["references"]}, {101, 102, 103})
                self.assertTrue(result["coverage"]["complete"])
                self.assertEqual(result["coverage"]["documents_searched"], 3)
                self.assertEqual(result["coverage"]["passages_searched"], 3)

    def test_cited_cause_phrases_become_additional_search_variants(self):
        tenant, customer, asset, docs, chunks = self._fixture("VRF Outdoor Module")
        with patch("core.services.knowledge_scope.get_applicable_knowledge_queryset",
                   return_value=_DocumentQuery(docs)), \
             patch("core.services.retriever._knowledge_candidates", return_value=chunks), \
             patch("core.services.retriever._sparse_score_candidates", return_value=[]) as scoring, \
             patch("core.services.retriever.get_serving_config", return_value=object()), \
             patch("core.services.vectors.embed_text", side_effect=lambda _text: {"v": 1}), \
             patch("core.services.identifiers.extract_identifiers", return_value=set()), \
             patch("core.services.customer_isolation.is_verified_real_document", return_value=True), \
             patch("core.services.content_safety.detect_document_prompt_injection", return_value=False):
            scan.search_all_approved_documents(
                tenant=tenant, customer=customer, asset=asset,
                question="Explain each issue",
                original_complaint="Outdoor module not working",
                cause_claims=[{"source_text": "A sensor or control issue may affect operation."}],
            )
        queries = [call.kwargs["evidence_text"] for call in scoring.call_args_list]
        self.assertTrue(any("sensor" in query.casefold() for query in queries))
        self.assertTrue(any(
            "outdoor" in query.casefold() and "working" in query.casefold()
            for query in queries
        ))
        self.assertGreaterEqual(len(queries), 2)

    def test_generic_failure_overview_adds_each_fault_heading_to_scan_queries(self):
        tenant, customer, asset, docs, _ = self._fixture("VRF Outdoor Module")
        headings = [
            "2. COMMUNICATION OR ADDRESSING FAULT",
            "3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT",
            "4. LOW-PRESSURE OR CAPACITY FAULT",
            "5. COMPRESSOR OR INVERTER FAULT",
            "6. FAN OR TEMPERATURE-SENSOR FAULT",
        ]
        chunks = []
        for index, heading in enumerate(headings):
            document = docs[index % len(docs)]
            chunks.append(SimpleNamespace(
                id=2000 + index, document_id=document.id, tenant_id=tenant.id,
                text="Possible causes:\n- Communication cable damage or incorrect topology.",
                heading=heading, document=document, is_quarantined=False,
            ))
        with patch("core.services.knowledge_scope.get_applicable_knowledge_queryset",
                   return_value=_DocumentQuery(docs)), \
             patch("core.services.retriever._knowledge_candidates", return_value=chunks), \
             patch("core.services.retriever._sparse_score_candidates", return_value=[]) as scoring, \
             patch("core.services.retriever.get_serving_config", return_value=object()), \
             patch("core.services.vectors.embed_text", side_effect=lambda _text: {"v": 1}), \
             patch("core.services.identifiers.extract_identifiers", return_value=set()), \
             patch("core.services.customer_isolation.is_verified_real_document", return_value=True), \
             patch("core.services.content_safety.detect_document_prompt_injection", return_value=False):
            scan.search_all_approved_documents(
                tenant=tenant, customer=customer, asset=asset,
                question="My VRF outdoor module is not working. What could be the possible reasons?",
                original_complaint="My VRF outdoor module is not working. What could be the possible reasons?",
                cause_claims=[{"source_text": "Retrieved possible cause " + str(i)} for i in range(8)],
            )
        queries = [call.kwargs["evidence_text"].casefold() for call in scoring.call_args_list]
        for heading in headings:
            self.assertTrue(any(heading.casefold() in query for query in queries), heading)


    def test_approved_document_without_chunks_marks_coverage_incomplete(self):
        tenant, customer, asset, docs, chunks = self._fixture("VRF Outdoor Module")
        docs.append(SimpleNamespace(
            id=104, tenant_id=1, title="Approved but not chunked",
            is_confidential=False, disable_sharing=False, is_rag_enabled=True,
        ))
        ranked = [
            SimpleNamespace(document_id=c.document_id, chunk_id=c.id, score=0.4)
            for c in chunks
        ]
        with patch("core.services.knowledge_scope.get_applicable_knowledge_queryset",
                   return_value=_DocumentQuery(docs)), \
             patch("core.services.retriever._knowledge_candidates", return_value=chunks), \
             patch("core.services.retriever._sparse_score_candidates", return_value=ranked), \
             patch("core.services.retriever.get_serving_config", return_value=object()), \
             patch("core.services.vectors.embed_text", side_effect=lambda _text: {"v": 1}), \
             patch("core.services.identifiers.extract_identifiers", return_value=set()), \
             patch("core.services.customer_isolation.is_verified_real_document", return_value=True), \
             patch("core.services.content_safety.detect_document_prompt_injection", return_value=False):
            result = scan.search_all_approved_documents(
                tenant=tenant, customer=customer, asset=asset,
                question="VRF outdoor module not working",
            )
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["documents_without_searchable_chunks"], 1)

    def test_safe_scan_limit_fails_closed_without_partial_results(self):
        tenant, customer, asset, docs, chunks = self._fixture("VRF Outdoor Module")
        with patch.object(scan, "MAX_APPROVED_CHUNKS", 2), \
             patch("core.services.knowledge_scope.get_applicable_knowledge_queryset",
                   return_value=_DocumentQuery(docs)), \
             patch("core.services.retriever._knowledge_candidates", return_value=chunks), \
             patch("core.services.retriever._sparse_score_candidates") as scoring, \
             patch("core.services.customer_isolation.is_verified_real_document", return_value=True), \
             patch("core.services.content_safety.detect_document_prompt_injection", return_value=False):
            result = scan.search_all_approved_documents(
                tenant=tenant, customer=customer, asset=asset,
                question="VRF outdoor module not working",
            )
        scoring.assert_not_called()
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["references"], [])


if __name__ == "__main__":
    unittest.main()
