"""Single active RAG request: a second request is refused (not queued) and nothing is stored."""
import time

from django.test import SimpleTestCase, override_settings

from core.models import CustomerMessage
from core.services import rag_admission
from core.services.rag_admission import RagBusy, rag_request
from core.services.rag_event_stream import rag_execution_stream
from core.tests_demo_gateway import GatewayTestBase


def _wait_released(seconds=5.0):
    end = time.monotonic() + seconds
    while rag_admission.busy() and time.monotonic() < end:
        time.sleep(0.01)
    return not rag_admission.busy()


class AdmissionUnitTests(SimpleTestCase):
    def tearDown(self):
        self.assertTrue(_wait_released())

    def test_second_request_is_refused_while_one_is_active(self):
        with rag_request():
            self.assertTrue(rag_admission.busy())
            with self.assertRaises(RagBusy):
                with _OtherRequest():
                    pass
        self.assertFalse(rag_admission.busy())

    def test_slot_is_released_after_a_failure(self):
        with self.assertRaises(ValueError):
            with rag_request():
                raise ValueError("pipeline failed")
        self.assertFalse(rag_admission.busy())
        with rag_request():
            pass

    def test_disabled_single_request_mode_never_blocks(self):
        with override_settings(SERVY_RAG_SINGLE_REQUEST=False):
            with rag_request():
                with _OtherRequest():
                    pass

    def test_stream_refuses_with_busy_status_and_releases_after_completion(self):
        with rag_request():
            refused = rag_execution_stream(lambda: {"ok": True})
            self.assertEqual(refused.status_code, 503)
            self.assertIn(b"rag_busy", refused.content)
        response = rag_execution_stream(lambda: {"ok": True})
        body = b"".join(response.streaming_content)
        self.assertIn(b"event: result", body)
        self.assertTrue(_wait_released())

    def test_stream_releases_the_slot_when_the_work_fails(self):
        def boom():
            raise RuntimeError("model crashed")
        body = b"".join(rag_execution_stream(boom).streaming_content)
        self.assertIn(b"event: error", body)
        self.assertTrue(_wait_released())


class _OtherRequest:
    """A request arriving from another context (not the one holding the slot)."""

    def __enter__(self):
        from contextvars import copy_context
        self._ctx = copy_context()
        self._ctx.run(rag_admission._ADMITTED.set, False)
        self._cm = rag_request()
        return self._ctx.run(self._cm.__enter__)

    def __exit__(self, *exc):
        return self._ctx.run(self._cm.__exit__, *exc)


class AdmissionGatewayTests(GatewayTestBase):
    def test_busy_gateway_returns_503_and_stores_no_duplicate_turn(self):
        uuid = self.start_conversation()
        before = CustomerMessage.objects.count()
        self.assertTrue(rag_admission.try_acquire())
        try:
            r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/messages/",
                                 data={"message": "What can cause poor cooling?"}, content_type="application/json")
        finally:
            rag_admission.release()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["code"], "rag_busy")
        self.assertIn("previous question", r.json()["detail"])
        self.assertEqual(CustomerMessage.objects.count(), before)
        # Once the active request finishes, the same question is answered normally.
        r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/messages/",
                             data={"message": "What can cause poor cooling?"}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(CustomerMessage.objects.count(), before + 2)
