import unittest
from unittest.mock import Mock

from tools.servy_latency_profile import (
    summarize,
    soft_delete_created_conversation,
)


class LatencyProfileHelperTests(unittest.TestCase):
    def test_summarize_returns_none_when_stage_was_not_observed(self):
        self.assertIsNone(summarize([]))
        self.assertIsNone(summarize([None, None]))

    def test_summarize_reports_median_nearest_rank_p90_and_max(self):
        self.assertEqual(
            summarize([10, 20, 30, 40, 50]),
            {"n": 5, "median_ms": 30.0, "p90_ms": 50.0, "max_ms": 50.0},
        )

    def test_cleanup_targets_only_the_uuid_passed_by_this_run(self):
        session = Mock()
        response = Mock(status_code=204)
        session.delete.return_value = response
        cleaned = set()
        headers = {"Authorization": "Demo <redacted>"}

        status = soft_delete_created_conversation(
            session, headers, "profiler-owned-uuid", cleaned,
        )

        self.assertEqual(status, 204)
        self.assertEqual(cleaned, {"profiler-owned-uuid"})
        session.delete.assert_called_once_with(
            "http://127.0.0.1:8002/demo-api/customer-support/conversations/profiler-owned-uuid/",
            headers=headers,
            timeout=(10, 30),
        )

    def test_cleanup_is_idempotent_and_does_not_send_duplicate_delete(self):
        session = Mock()
        cleaned = {"already-cleaned-uuid"}

        status = soft_delete_created_conversation(
            session, {}, "already-cleaned-uuid", cleaned,
        )

        self.assertEqual(status, 204)
        session.delete.assert_not_called()

    def test_cleanup_accepts_already_absent_conversation(self):
        session = Mock()
        session.delete.return_value = Mock(status_code=404)
        cleaned = set()

        status = soft_delete_created_conversation(
            session, {}, "missing-profiler-uuid", cleaned,
        )

        self.assertEqual(status, 404)
        self.assertIn("missing-profiler-uuid", cleaned)


if __name__ == "__main__":
    unittest.main()
