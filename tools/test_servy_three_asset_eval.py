"""Integrity regressions for the live evaluation harness; no network or private data."""
import json
import unittest
from unittest.mock import patch

from tools import servy_three_asset_eval as evaluation


class EvaluationIntegrityTests(unittest.TestCase):
    def test_empty_answer_and_missing_expectation_never_pass(self):
        answer = {"http_status": 200, "response_type": "GREETING", "answer": "Hi"}
        self.assertEqual(evaluation.check({"types": ["GREETING"]}, {**answer, "answer": " "})[0], "BLOCKED")
        self.assertEqual(evaluation.check({}, answer)[0], "BLOCKED")
        self.assertEqual(evaluation.check({"types": ["GREETING"]}, answer)[0], "PASS")
        self.assertEqual(evaluation.check({"types": ["CLARIFICATION"]}, answer)[0], "FAIL")

    def test_missing_asset_accounts_for_every_turn(self):
        specs = [{"q": "Hi", "expect": {"types": ["GREETING"]}}, {"q": "Help"}]
        with patch.object(evaluation, "request") as request:
            conv, turns = evaluation.run_sequence("", "", None, specs, graded=True)
        request.assert_not_called()
        self.assertIsNone(conv)
        self.assertEqual([t["verdict"] for t in turns], ["BLOCKED", "BLOCKED"])

    def test_transport_failure_blocks_dependent_turns(self):
        with patch.object(evaluation, "request", return_value=(201, '{"conversation_uuid":"incident"}')), \
                patch.object(evaluation, "turn", return_value={"http_status": 200, "answer": ""}) as send:
            _, turns = evaluation.run_sequence("", "", {"id": 1}, [{"q": "Hi"}, {"q": "Next"}], graded=False)
        self.assertEqual(send.call_count, 1)
        self.assertEqual([t["verdict"] for t in turns], ["BLOCKED", "BLOCKED"])

    def test_visible_segments_are_graded_not_secondary_answer(self):
        result = {"contract": {"response_type": "DIAGNOSTIC_GUIDANCE", "answer": "hidden text",
                               "presentation": {"response": {"segments": [{"text": "visible answer"}]}}}}
        frame = "event: result\ndata: " + json.dumps(result) + "\n\n"
        with patch.object(evaluation, "request", return_value=(200, frame)):
            turn = evaluation.turn("", "", "incident", "question")
        self.assertEqual(turn["answer"], "visible answer")
        self.assertEqual(turn["contract_answer"], "hidden text")
        self.assertEqual(evaluation.check({"answer_any": ["hidden text"]}, turn)[0], "FAIL")

    def test_multiple_terminal_events_are_blocked(self):
        frame = 'event: result\ndata: {"contract":{"response_type":"GREETING","answer":"Hi"}}\n\n'
        with patch.object(evaluation, "request", return_value=(200, frame + frame)):
            turn = evaluation.turn("", "", "incident", "question")
        self.assertFalse(evaluation.complete_answer(turn))


if __name__ == "__main__":
    unittest.main()
