from __future__ import annotations

from threading import Thread
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth.models import User
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from core.api_views import (
    CustomerConversationMessagesStreamView,
    CustomerSupportQueryStreamView,
    EngineerCopilotQueryStreamView,
    EngineerCopilotQueryView,
)

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


class TechnicianCopilotStreamAccessTests(SimpleTestCase):
    def test_technician_stream_rejects_anonymous_requests_before_work(self):
        request = APIRequestFactory().post(
            "/api/engineer-copilot/query-stream/",
            {"call_id": 100, "question": "synthetic question"},
            format="json",
        )
        with patch.object(
            EngineerCopilotQueryStreamView,
            "_prepare",
            side_effect=AssertionError("unauthenticated requests must not prepare RAG work"),
        ):
            response = EngineerCopilotQueryStreamView.as_view()(request)
        self.assertEqual(response.status_code, 403)

    def test_authenticated_stream_forwards_live_events_and_final_copilot_result(self):
        request = APIRequestFactory().post(
            "/api/engineer-copilot/query-stream/",
            {"call_id": 100, "question": "synthetic question"},
            format="json",
        )
        user = User(username="stream-test", is_active=True)
        force_authenticate(request, user=user)
        prepared = {"scope": "synthetic authorized ticket"}

        def execute(_prepared):
            with execution_stage(
                "technician_ticket_authorization",
                "Revalidating technician ticket and tenant access",
            ):
                emit_execution_event(
                    "technician_confidential_scope",
                    "Confidential source access",
                    "info",
                    metrics={"confidential_scope": "restricted"},
                )
            return {"answer": "synthetic technician recommendation"}

        with (
            patch.object(EngineerCopilotQueryStreamView, "throttle_classes", []),
            patch.object(
                EngineerCopilotQueryStreamView,
                "_prepare",
                return_value=(None, prepared),
            ),
            patch.object(EngineerCopilotQueryView, "_execute", staticmethod(execute)),
        ):
            response = EngineerCopilotQueryStreamView.as_view()(request)
            body = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: log", body)
        self.assertIn("technician_ticket_authorization", body)
        self.assertIn("confidential_scope", body)
        self.assertIn('"answer":"synthetic technician recommendation"', body)


class CustomerRAGStreamAccessTests(SimpleTestCase):
    def test_customer_first_turn_stream_rejects_anonymous_requests_before_work(self):
        request = APIRequestFactory().post(
            "/api/customer-support/query-stream/",
            {"site_id": 10, "asset_id": 59, "question": "synthetic question"},
            format="json",
        )
        with patch.object(
            CustomerSupportQueryStreamView,
            "_prepare",
            side_effect=AssertionError("anonymous customer requests must not prepare RAG work"),
        ):
            response = CustomerSupportQueryStreamView.as_view()(request)
        self.assertEqual(response.status_code, 403)

    def test_authenticated_customer_first_turn_stream_forwards_real_stage_events(self):
        request = APIRequestFactory().post(
            "/api/customer-support/query-stream/",
            {"site_id": 10, "asset_id": 59, "question": "synthetic question"},
            format="json",
        )
        user = User(username="customer-stream-test", is_active=True)
        force_authenticate(request, user=user)

        def execute(_prepared):
            with execution_stage(
                "customer_safety_screening",
                "Checking customer-reported conditions",
            ):
                emit_execution_event(
                    "customer_safety_outcome",
                    "No safety escalation condition detected",
                    "info",
                )
            return {"response_type": "DIAGNOSTIC_GUIDANCE", "answer": "synthetic customer answer"}

        with (
            patch.object(CustomerSupportQueryStreamView, "throttle_classes", []),
            patch.object(
                CustomerSupportQueryStreamView,
                "_prepare",
                return_value=(None, {"asset_id": 59, "customer_id": 14}),
            ),
            patch.object(CustomerSupportQueryStreamView, "_execute", staticmethod(execute)),
        ):
            response = CustomerSupportQueryStreamView.as_view()(request)
            body = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: log", body)
        self.assertIn("customer_safety_screening", body)
        self.assertIn('"status":"started"', body)
        self.assertIn('"status":"completed"', body)
        self.assertIn('"answer":"synthetic customer answer"', body)

    def test_customer_conversation_followup_stream_rejects_anonymous_requests(self):
        conversation_uuid = uuid4()
        request = APIRequestFactory().post(
            f"/api/customer-support/conversations/{conversation_uuid}/messages-stream/",
            {"message": "synthetic follow-up"},
            format="json",
        )
        with patch.object(
            CustomerConversationMessagesStreamView,
            "_prepare_post",
            side_effect=AssertionError("anonymous follow-ups must not prepare RAG work"),
        ):
            response = CustomerConversationMessagesStreamView.as_view()(
                request, conversation_uuid=conversation_uuid
            )
        self.assertEqual(response.status_code, 403)

    def test_authenticated_customer_conversation_followup_stream_forwards_result(self):
        conversation_uuid = uuid4()
        request = APIRequestFactory().post(
            f"/api/customer-support/conversations/{conversation_uuid}/messages-stream/",
            {"message": "synthetic follow-up"},
            format="json",
        )
        user = User(username="customer-followup-stream-test", is_active=True)
        force_authenticate(request, user=user)

        def execute(_prepared):
            with execution_stage(
                "customer_conversation_context",
                "Reconstructing authorized conversation context",
            ):
                emit_execution_event(
                    "customer_context_summary", "Conversation context summary",
                    "info", metrics={"candidate_count": 2},
                )
            return {
                "user_message": {"sender_role": "customer", "message_text": "synthetic"},
                "assistant_message": {"sender_role": "assistant", "message_text": "synthetic"},
                "contract": {"response_type": "DIAGNOSTIC_GUIDANCE"},
            }

        with (
            patch.object(CustomerConversationMessagesStreamView, "throttle_classes", []),
            patch.object(
                CustomerConversationMessagesStreamView,
                "_prepare_post",
                return_value=(None, {"conversation_uuid": str(conversation_uuid)}),
            ),
            patch.object(
                CustomerConversationMessagesStreamView,
                "_execute_post",
                staticmethod(execute),
            ),
        ):
            response = CustomerConversationMessagesStreamView.as_view()(
                request, conversation_uuid=conversation_uuid
            )
            body = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertIn("customer_conversation_context", body)
        self.assertIn('"candidate_count":2', body)
        self.assertIn("event: result", body)
        self.assertIn('"response_type":"DIAGNOSTIC_GUIDANCE"', body)
