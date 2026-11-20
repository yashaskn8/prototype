"""Request deadline, bounded stages, timeout-safe fallbacks and the answer-size contract."""
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests
from django.conf import settings
from django.test import SimpleTestCase, override_settings

from core.models import KnowledgeDocument
from core.services import customer_answer_generation as generation
from core.services.customer_answer import MAX_SECTION_CLAIMS
from core.services.indexing import index_document
from core.services.rag_execution_log import bind_execution_log
from core.services.request_deadline import (
    StageBusy, StageTimeout, current_deadline, request_deadline, run_bounded, stage_budget, wait_until_idle)
from core.services.turn_diagnostics import build_outcome
from core.tests_demo_gateway import GatewayTestBase

SPLIT_AC_QUESTION = "what could be the possible causes the split ac unit not working?"


class ContractLimitTests(SimpleTestCase):
    def test_backend_and_frontend_share_the_section_claim_limit(self):
        source = (Path(settings.BASE_DIR) / "frontend" / "src" / "components"
                  / "customerAnswerContract.js").read_text(encoding="utf-8")
        frontend = int(re.search(r"export const MAX_SECTION_CLAIMS = (\d+);", source).group(1))
        self.assertEqual(frontend, MAX_SECTION_CLAIMS)
        self.assertIn("section.claims.length > MAX_SECTION_CLAIMS", source)


class _IdleWorkers:
    def setUp(self):
        super().setUp()
        # Earlier suites may leave abandoned bounded work (for example a cold BGE load).
        self.assertTrue(wait_until_idle(900))


class DeadlineTests(_IdleWorkers, SimpleTestCase):
    def test_no_deadline_runs_inline_unchanged(self):
        self.assertIsNone(stage_budget("v5_reranking"))
        self.assertEqual(run_bounded("v5_reranking", lambda x: x + 1, 1), 2)

    def test_stage_budget_is_capped_and_keeps_the_reserve(self):
        with override_settings(SERVY_RAG_DEADLINE_RESERVE_SECONDS=0.4):
            with request_deadline(5.0) as deadline:
                self.assertLessEqual(stage_budget("v5_reranking"), deadline.caps["v5_reranking"])
                self.assertLessEqual(stage_budget("unknown_stage"), 5.0 - 0.4)
            with request_deadline(0.3):
                self.assertEqual(stage_budget("customer_qwen_generation"), 0.0)
        self.assertIsNone(current_deadline())

    def test_slow_stage_times_out_without_waiting_for_it(self):
        release = threading.Event()
        with request_deadline(0.6) as deadline:
            deadline.reserve = 0.0
            started = time.monotonic()
            with self.assertRaises(StageTimeout):
                run_bounded("v5_reranking", release.wait, 5)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertIn("v5_reranking", deadline.timeouts)
        release.set()

    def test_exhausted_budget_never_starts_work(self):
        called = []
        with request_deadline(0.0):
            with self.assertRaises(StageTimeout):
                run_bounded("dense_retrieval", called.append, 1)
        self.assertEqual(called, [])

    def test_abandoned_work_is_bounded_and_releases_its_slot(self):
        release = threading.Event()
        for _ in range(2):
            with request_deadline(0.3) as deadline:
                deadline.reserve = 0.0
                with self.assertRaises(StageTimeout):
                    run_bounded("v5_reranking", release.wait, 5)
        # Both workers are still busy: new work is refused, not queued behind them.
        with request_deadline(2.0):
            with self.assertRaises(StageBusy):
                run_bounded("v5_reranking", lambda: 1)
        release.set()
        time.sleep(0.2)
        with request_deadline(2.0):
            self.assertEqual(run_bounded("v5_reranking", lambda: 7), 7)


def _candidate(text, score):
    return SimpleNamespace(source_kind="knowledge_document", text=text, score=score, reranker_score=None)


@override_settings(SERVY_RERANKER_ENABLED=True, SERVY_RERANKER_REQUIRED=True, SERVY_RERANKER_POOL_SIZE=5)
class RerankerDeadlineTests(_IdleWorkers, SimpleTestCase):
    def _rerank(self, predict):
        from core.ml import reranker
        model = SimpleNamespace(predict=predict)
        events = []
        cands = [_candidate("first", 0.9), _candidate("second", 0.8)]
        with mock.patch.object(reranker, "get_reranker", return_value=model), \
                bind_execution_log(events.append), request_deadline(1.0) as deadline:
            deadline.reserve = 0.0
            deadline.caps["v5_reranking"] = 0.3
            ranked = reranker.rerank_candidates("q", cands, top_k=2)
        return cands, ranked, events

    def test_v5_timeout_keeps_authorized_order_and_reports_not_applied(self):
        gate = threading.Event()
        cands, ranked, events = self._rerank(lambda *a, **k: gate.wait(3) or [1.0, 2.0])
        gate.set()
        self.assertEqual([c.text for c in ranked], ["first", "second"])
        self.assertTrue(all(c.reranker_score is None for c in ranked))
        self.assertIn(("v5_reranking_outcome", "StageTimeout"),
                      [(e["stage"], e.get("error_type")) for e in events])

    def test_v5_success_scores_and_reorders(self):
        cands, ranked, _ = self._rerank(lambda *a, **k: [-1.0, 3.0])
        self.assertEqual([c.text for c in ranked], ["second", "first"])
        self.assertTrue(all(c.reranker_score is not None for c in ranked))

    def test_v5_inference_failure_is_not_reported_as_success(self):
        def broken(*a, **k):
            raise RuntimeError("bad weights")
        with override_settings(SERVY_RERANKER_REQUIRED=False):
            cands, ranked, _ = self._rerank(broken)
        self.assertTrue(all(c.reranker_score is None for c in ranked))


def _presentation(n=3):
    claims = [{"id": f"c{i + 1}", "kind": "possible_cause", "text": f"Possible documented cause: Cause {i + 1} may stop the unit.",
               "source_text": f"Cause {i + 1} may stop the unit.", "evidence_ids": [f"s{i + 1}"]} for i in range(n)]
    return {"equipment": {"name": "Split AC"}, "supporting_claims": claims,
            "sections": [{"number": 1, "title": "What could be wrong?", "claims": claims}],
            "citations": [{"id": f"s{i + 1}", "url": f"/api/customer-support/source/?citation=signed-{i + 1:04d}",
                           "document_id": 1, "chunk_id": i + 1} for i in range(n)]}


@override_settings(SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
                   SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
                   SERVY_CUSTOMER_GENERATION_MODEL="qwen3.5:2b-q4_K_M")
class QwenDeadlineTests(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.object(generation, "_server_online", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_timeout_keeps_exact_source_wording(self):
        presentation = _presentation()
        with mock.patch.object(requests.Session, "post", side_effect=requests.ReadTimeout()), request_deadline(5.0):
            report = generation.apply_customer_answer_generation_with_status(presentation, "q")
        self.assertEqual(report["status"], "timeout")
        self.assertEqual(report["applied_claims"], 0)
        self.assertEqual(presentation["supporting_claims"][0]["text"],
                         "Possible documented cause: Cause 1 may stop the unit.")

    def test_no_budget_left_skips_generation_without_calling_ollama(self):
        with mock.patch.object(requests.Session, "post") as post, request_deadline(0.5):
            report = generation.apply_customer_answer_generation_with_status(_presentation(), "q")
        post.assert_not_called()
        self.assertEqual(report["status"], "deadline_skipped")

    def test_request_timeout_and_claim_count_follow_the_remaining_budget(self):
        captured = {}

        def fake_post(self, url, **kwargs):
            captured.update(kwargs)
            raise requests.ReadTimeout()
        with mock.patch.object(requests.Session, "post", fake_post), request_deadline(2.0) as deadline:
            deadline.reserve = 0.0
            generation.apply_customer_answer_generation_with_status(_presentation(6), "q")
        self.assertLessEqual(captured["timeout"][1], 2.0)
        sent = json.loads(captured["json"]["messages"][1]["content"])["approved_claims"]
        self.assertLess(len(sent), 6)
        self.assertEqual(captured["json"]["keep_alive"], "30m")
        self.assertIn("source_quote", json.dumps(captured["json"]["format"]))

    def test_claims_are_planned_by_estimated_generation_time(self):
        short = {"source_text": "Fan stops."}
        long = {"source_text": "x" * 100}  # ~1.14 s each
        self.assertEqual(len(generation._affordable_claims([short] * 6, 5.0)), 6)
        self.assertEqual(len(generation._affordable_claims([long] * 6, 3.2)), 2)
        # At least one claim is always attempted when generation starts at all.
        self.assertEqual(len(generation._affordable_claims([long] * 3, 0.9)), 1)

    def test_invalid_json_is_rejected(self):
        response = mock.Mock(status_code=200, content=b"{}")
        response.json.return_value = {"message": {"content": "{not json"}}
        with mock.patch.object(requests.Session, "post", return_value=response):
            report = generation.apply_customer_answer_generation_with_status(_presentation(), "q")
        self.assertEqual(report["status"], "invalid_model_json")
        self.assertEqual(report["applied_claims"], 0)


class OutcomeTests(SimpleTestCase):
    def _contract(self, response_type="DIAGNOSTIC_GUIDANCE", citations=True):
        return {"response_type": response_type, "retrieval_backend": "hybrid", "reranker_model": "servy_v5_bge_verified",
                "presentation": {"response_type": response_type,
                                 "citations": [{"id": "s1", "title": "M", "heading": "H"}] if citations else []}}

    def _events(self, qwen_status="applied", accepted=3, v5=True, dense="completed"):
        events = [{"stage": "dense_retrieval", "status": dense,
                   "error_type": None if dense == "completed" else "StageTimeout"},
                  {"stage": "rank_fusion_summary", "metrics": {"dense_count": 3, "sparse_count": 2, "retrieved_count": 4}},
                  {"stage": "retrieval_summary", "metrics": {"reranker_applied": v5}},
                  {"stage": "v5_reranking", "status": "completed"},
                  {"stage": "customer_qwen_generation", "status": "completed"},
                  {"stage": "customer_qwen_outcome", "metrics": {"generation_status": qwen_status,
                                                                "candidate_count": 3, "supported_count": accepted}}]
        if not v5:
            events.append({"stage": "v5_reranking_outcome", "status": "skipped", "error_type": "StageTimeout"})
        return events

    def test_full_success_is_grounded_generated(self):
        with request_deadline(5.0) as deadline:
            o = build_outcome(self._events(), self._contract(), deadline)
        self.assertEqual((o["status"], o["response_mode"]), ("success", "grounded_generated"))
        self.assertTrue(o["reranker_applied"])
        self.assertEqual(o["reranker_model"], "servy_v5_bge_verified")
        self.assertTrue(o["generation_applied"])
        self.assertFalse(o["deadline_exceeded"])

    def test_qwen_timeout_is_degraded_extractive(self):
        o = build_outcome(self._events("timeout", 0), self._contract())
        self.assertEqual((o["status"], o["response_mode"]), ("degraded", "grounded_extractive"))
        self.assertFalse(o["generation_applied"])
        self.assertEqual(o["fallback_reason"], "generation_timeout")

    def test_v5_and_dense_timeouts_are_reported_not_hidden(self):
        o = build_outcome(self._events(v5=False, dense="skipped"), self._contract())
        self.assertFalse(o["reranker_applied"])
        self.assertIsNone(o["reranker_model"])
        self.assertEqual(o["reranker_status"], "timeout")
        self.assertEqual(o["retrieval_status"], "sparse_only_dense_timeout")
        self.assertEqual(o["status"], "degraded")

    def test_insufficient_evidence_and_safety(self):
        o = build_outcome([], self._contract("INSUFFICIENT_EVIDENCE", citations=False))
        self.assertEqual((o["status"], o["response_mode"]), ("insufficient_evidence", "no_validated_evidence"))
        o = build_outcome([], self._contract("ESCALATION_REQUIRED", citations=False))
        self.assertEqual(o["status"], "safety_escalation")

    def test_expired_deadline_is_flagged(self):
        with request_deadline(0.0) as deadline:
            o = build_outcome(self._events(), self._contract(), deadline)
        self.assertTrue(o["deadline_exceeded"])


def _causes_manual(n):
    lines = [f"- Possible fault number {i + 1} in the indoor control circuit stops the unit." for i in range(n)]
    return "2. SYMPTOM: UNIT DOES NOT START\nPossible causes:\n" + "\n".join(lines)


class SplitACAnswerGatewayTests(GatewayTestBase):
    """The original failing question through the real Customer gateway (JSON endpoint)."""

    def ask(self, question, asset=None):
        r = self.as_customer("post", "/demo-api/customer-support/conversations/",
                             data={"asset_id": (asset or self.asset).id}, content_type="application/json")
        uuid = r.json()["conversation_uuid"]
        r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/messages/",
                             data={"message": question}, content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content[:300])
        body = r.json()
        return (body.get("assistant_message") or {}).get("structured_contract") or body.get("contract")

    def assert_valid_contract(self, contract):
        """Mirror of the client contract (customerAnswerContract.js)."""
        pres = contract["presentation"]
        self.assertEqual(pres["equipment"]["id"], self.asset.id)
        ids = {c["id"] for c in pres["citations"]}
        self.assertLessEqual(len(pres["citations"]), 30)
        self.assertEqual(len(pres["sections"]), 4)
        for section in pres["sections"]:
            self.assertTrue(1 <= len(section["claims"]) <= MAX_SECTION_CLAIMS, len(section["claims"]))
            for claim in section["claims"]:
                self.assertTrue(set(claim["evidence_ids"]) <= ids)
        used = {e for s in pres["sections"] for c in s["claims"] for e in c["evidence_ids"]}
        for segment in (pres.get("response") or {}).get("segments") or []:
            if segment["kind"] in ("claim", "list_item"):
                self.assertEqual(len(segment["citation_ids"]), 1)
                self.assertIn(segment["citation_ids"][0], used)

    def test_original_split_ac_question_with_many_causes_is_a_valid_answer(self):
        for count in (1, 12, 13, 20, 25):
            with self.subTest(causes=count):
                KnowledgeDocument.objects.filter(asset=self.asset).delete()
                doc = KnowledgeDocument.objects.create(
                    tenant=self.tenant, asset=self.asset, title="Split AC Unit 2T - Troubleshooting",
                    source_type="text", is_rag_enabled=True, content_text=_causes_manual(count))
                index_document(doc)
                contract = self.ask(SPLIT_AC_QUESTION)
                self.assertEqual(contract["response_type"], "DIAGNOSTIC_GUIDANCE", contract.get("answer"))
                causes = [c for c in contract["presentation"]["sections"][0]["claims"] if c["kind"] == "possible_cause"]
                self.assertEqual(len(causes), min(count, MAX_SECTION_CLAIMS))
                self.assert_valid_contract(contract)
                self.assertIn(contract["outcome"]["status"], ("success", "degraded"))
                self.assertEqual(contract["outcome"]["response_mode"], "grounded_extractive")

    def test_wrong_equipment_is_refused(self):
        r = self.as_customer("post", "/demo-api/customer-support/conversations/",
                             data={"asset_id": self.other_asset.id}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_expired_deadline_still_returns_only_validated_cited_claims(self):
        with override_settings(SERVY_RAG_DEADLINE_SECONDS=0.0):
            contract = self.ask(SPLIT_AC_QUESTION)
        outcome = contract["outcome"]
        self.assertTrue(outcome["deadline_exceeded"])
        self.assertFalse(outcome["generation_applied"])
        self.assertFalse(outcome["reranker_applied"])
        if contract["response_type"] == "DIAGNOSTIC_GUIDANCE":
            self.assert_valid_contract(contract)
        else:
            self.assertEqual(outcome["status"], "insufficient_evidence")
            self.assertFalse(contract["presentation"].get("citations"))

    def test_unanswerable_question_reports_insufficient_evidence(self):
        contract = self.ask("What paint colour options exist for the remote control holder?")
        self.assertEqual(contract["outcome"]["status"], "insufficient_evidence")
        self.assertFalse((contract.get("presentation") or {}).get("citations"))

    def test_safety_question_keeps_safety_response(self):
        contract = self.ask("There is smoke and a burning smell coming from the indoor unit")
        self.assertEqual(contract["outcome"]["status"], "safety_escalation")

    def test_back_to_back_conversations_each_store_exactly_one_answer(self):
        """Separate requests share no per-request state and never duplicate answers.
        (True concurrency is measured live against MySQL; see the latency report.)"""
        results = []
        for _ in range(2):
            results.append(self.ask(SPLIT_AC_QUESTION))
        for contract in results:
            if contract["response_type"] == "DIAGNOSTIC_GUIDANCE":
                self.assert_valid_contract(contract)
        from core.models import CustomerMessage
        self.assertEqual(CustomerMessage.objects.filter(sender_role="assistant").count(), 2)
