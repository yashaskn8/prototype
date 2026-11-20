"""Per-request metrics: written only when enabled, privacy-safe, and summarized correctly."""
import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from core.management.commands.rag_metrics_report import summarize
from core.services import rag_metrics

SECRET_QUESTION = "my unit at 12 Private Road is not cooling"


def _contract(status="success", total=3200.0):
    return {"response_type": "DIAGNOSTIC_GUIDANCE", "answer": SECRET_QUESTION,
            "presentation": {"citations": [{"id": "s1"}], "equipment": {"id": 7}},
            "outcome": {"status": status, "response_mode": "grounded_generated", "total_duration_ms": total,
                        "deadline_exceeded": False, "reranker_status": "applied", "generation_status": "applied",
                        "generation_applied": status == "success"},
            "diagnostics": {"retrieval": {"retrieved": 6}, "qwen": {"accepted_claims": 3, "eligible_claims": 3}}}


class RagMetricsTests(SimpleTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "rag_metrics.jsonl"
        self.addCleanup(self._close)

    def _close(self):
        for handler in list(rag_metrics._logger.handlers):
            rag_metrics._logger.removeHandler(handler)
            handler.close()
        rag_metrics._configured_path = None

    def test_disabled_by_default_writes_nothing(self):
        with override_settings(SERVY_RAG_METRICS_LOG=""):
            rag_metrics.record(_contract(), [], asset_id=7)
        self.assertFalse(self.path.exists())

    def test_line_contains_counts_and_statuses_but_no_customer_text(self):
        events = [{"stage": "v5_reranking", "status": "completed", "duration_ms": 1250.0},
                  {"stage": "customer_qwen_generation", "status": "completed", "duration_ms": 2100.0}]
        with override_settings(SERVY_RAG_METRICS_LOG=str(self.path)):
            rag_metrics.record(_contract(), events, asset_id=7)
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("Private Road", raw)
        line = json.loads(raw.splitlines()[0])
        self.assertEqual(line["asset_id"], 7)
        self.assertEqual(line["stage_ms"], {"v5_reranking": 1250.0, "customer_qwen_generation": 2100.0})
        self.assertEqual(line["citations"], 1)
        self.assertEqual(line["qwen_accepted"], 3)

    def test_report_percentiles_and_rates(self):
        with override_settings(SERVY_RAG_METRICS_LOG=str(self.path)):
            for i in range(10):
                rag_metrics.record(_contract("success" if i < 8 else "degraded", 3000.0 + i * 100), [], asset_id=1)
        rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines()]
        report = summarize(rows)
        self.assertEqual(report["requests"], 10)
        self.assertEqual(report["latency_ms"]["p50"], 3400.0)
        self.assertEqual(report["latency_ms"]["max"], 3900.0)
        self.assertEqual(report["status"], {"success": 8, "degraded": 2})
        self.assertEqual(report["generated_share_of_cited"], 0.8)
