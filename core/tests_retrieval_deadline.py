"""Bounded retrieval within one request deadline (live demo defect: repeated
"Dense retrieval failed gracefully ... no budget left" warnings in one turn).
"""
import time
from unittest import mock

from django.test import SimpleTestCase, override_settings

from core.services import cause_explanations
from core.services.request_deadline import request_deadline
from core.services.retriever import retrieve
from core.services.turn_diagnostics import build_outcome
from core.tests_demo_gateway import GatewayTestBase


class ExhaustedDeadlineRetrievalTests(GatewayTestBase):
    def _retrieve(self):
        return retrieve(tenant=self.tenant, question="poor cooling dirty filter", asset=self.asset,
                        customer=self.customer, include_confidential=False, top_k=6, min_score=0.0,
                        include_past_resolutions=False, staff_mode=False)

    @override_settings(SERVY_RAG_DEADLINE_SECONDS=5.0)
    def test_no_dense_attempt_or_warning_once_the_budget_is_spent(self):
        from core.services.rag_execution_log import capture_execution_events, captured_events
        with mock.patch("core.services.retriever._dense_score_candidates") as dense, \
                self.assertNoLogs("servy.rag.retriever", level="WARNING"), \
                capture_execution_events(), request_deadline(total_seconds=0.0) as deadline:
            for _ in range(8):
                results = self._retrieve()
            events = [e for e in captured_events() if e.get("stage") == "dense_retrieval"]
        dense.assert_not_called()
        self.assertEqual(deadline.skips["dense_retrieval"], 8)
        self.assertTrue(all(e.get("error_type") == "DeadlineExhausted" for e in events))
        # Authorized sparse evidence is still returned.
        self.assertTrue(results)
        self.assertTrue(all(r.document_id for r in results))

    def test_repeated_dense_failures_warn_once_per_request(self):
        def boom(*args, **kwargs):
            raise RuntimeError("vector store unavailable")
        with mock.patch("core.services.retriever._dense_score_candidates", side_effect=boom), \
                self.assertLogs("servy.rag.retriever", level="WARNING") as logs, \
                request_deadline(total_seconds=60.0):
            for _ in range(5):
                self._retrieve()
        warnings = [r for r in logs.records if "Dense retrieval failed" in r.getMessage()]
        self.assertEqual(len(warnings), 1)


class GroupedExplanationRetrievalTests(SimpleTestCase):
    def test_one_retrieval_per_manual_section_not_per_cause(self):
        causes = ([{"heading": "2. SYMPTOM: UNIT DOES NOT START", "source_text": f"Start cause {i}."} for i in range(4)]
                  + [{"heading": "3. SYMPTOM: POOR COOLING", "source_text": f"Cooling cause {i}."} for i in range(4)])
        with mock.patch.object(cause_explanations, "_retrieve_for_cause", return_value=[]) as single:
            fn = cause_explanations._section_grouped_retrieval(causes)
            for cause in causes:
                fn(tenant=None, customer=None, asset=None, cause=cause)
        self.assertEqual(single.call_count, 2)
        queries = [c.kwargs["cause"]["source_text"] for c in single.call_args_list]
        self.assertIn("Start cause 3.", queries[0])
        self.assertIn("Cooling cause 0.", queries[1])


class TruthfulRetrievalDiagnosticsTests(SimpleTestCase):
    def turn_outcome(self, events):
        contract = {"response_type": "DIAGNOSTIC_GUIDANCE",
                    "presentation": {"response_type": "DIAGNOSTIC_GUIDANCE", "citations": [{"id": "s1"}]},
                    "diagnostics": {"retrieval": {"passes": len(events)}}}
        return build_outcome(events, contract, None)

    def test_partial_dense_is_reported_as_degraded_not_hybrid(self):
        events = [{"stage": "dense_retrieval", "status": "completed"},
                  {"stage": "dense_retrieval", "status": "skipped", "error_type": "DeadlineExhausted"}]
        outcome = self.turn_outcome(events)
        self.assertEqual(outcome["retrieval_status"], "hybrid_partial_dense_timeout")
        self.assertEqual(outcome["dense_passes"], {"completed": 1, "skipped_or_failed": 1})
        self.assertEqual(outcome["status"], "degraded")

    def test_all_skipped_is_sparse_only(self):
        events = [{"stage": "dense_retrieval", "status": "skipped", "error_type": "DeadlineExhausted"}]
        self.assertEqual(self.turn_outcome(events)["retrieval_status"], "sparse_only_dense_timeout")
