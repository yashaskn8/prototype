from __future__ import annotations

from threading import Thread

from django.test import SimpleTestCase

from core.services.rag_event_stream import rag_execution_stream
from core.services.rag_execution_log import (
    bind_execution_log,
    emit_execution_event,
    execution_stage,
)


class RAGExecutionLogTests(SimpleTestCase):
    def test_stage_emits_start_and_completion_with_duration(self):
        events = []
        with bind_execution_log(events.append):
            with execution_stage("retrieval", "Searching approved sources"):
                emit_execution_event(
                    "retrieval_summary", "Search summary", "info",
                    metrics={
                        "candidate_count": 12,
                        "question": "must never be serialized",
                        "customer_id": 999,
                    },
                )

        self.assertEqual([event["status"] for event in events], [
            "started", "info", "completed",
        ])
        self.assertEqual(events[0]["event_id"], events[2]["event_id"])
        self.assertGreaterEqual(events[2]["duration_ms"], 0)
        self.assertEqual(events[1]["metrics"], {"candidate_count": 12})
        self.assertNotIn("must never be serialized", str(events))
        self.assertNotIn("customer_id", str(events))

    def test_contextvar_sink_does_not_leak_into_other_threads(self):
        events = []
        with bind_execution_log(events.append):
            worker = Thread(target=lambda: emit_execution_event(
                "private", "Unbound worker event", "info"
            ))
            worker.start()
            worker.join()
            emit_execution_event("bound", "Request event", "info")
        self.assertEqual([event["stage"] for event in events], ["bound"])

    def test_sink_failure_does_not_interrupt_rag_work(self):
        def broken_sink(_event):
            raise RuntimeError("sink unavailable")

        with bind_execution_log(broken_sink):
            with execution_stage("retrieval", "Search"):
                emit_execution_event("summary", "Summary", "info")
        self.assertTrue(True)


class RAGExecutionStreamTests(SimpleTestCase):
    @staticmethod
    def _body(response):
        return b"".join(response.streaming_content).decode("utf-8")

    def test_live_stages_arrive_before_final_result(self):
        def work():
            with execution_stage("retrieval", "Searching authorized sources"):
                emit_execution_event(
                    "retrieval_summary", "Retrieval summary", "info",
                    metrics={"retrieved_count": 3},
                )
            return {"answer": "synthetic answer", "retrieved_count": 3}

        response = rag_execution_stream(work)
        body = self._body(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertEqual(response["Cache-Control"], "no-cache, no-transform, private")
        self.assertIn("event: log", body)
        self.assertIn('"stage":"retrieval"', body)
        self.assertIn('"status":"started"', body)
        self.assertIn('"status":"completed"', body)
        self.assertIn('"retrieved_count":3', body)
        self.assertIn("event: result", body)
        self.assertLess(body.index('"stage":"retrieval"'), body.index('"answer":"synthetic answer"'))

    def test_failed_worker_emits_generic_error_without_exception_text(self):
        def work():
            raise RuntimeError("private prompt and customer text")

        body = self._body(rag_execution_stream(work))
        self.assertIn("event: error", body)
        self.assertIn("RAG request failed. Please retry.", body)
        self.assertNotIn("private prompt", body)
        self.assertNotIn("customer text", body)

    def test_stage_errors_are_reported_without_raw_exception_message(self):
        def work():
            with execution_stage("vector_search", "Searching approved vector index"):
                raise ValueError("customer question should not appear")

        body = self._body(rag_execution_stream(work))
        self.assertIn('"status":"failed"', body)
        self.assertIn('"error_type":"ValueError"', body)
        self.assertNotIn("customer question", body)
