"""Offline security/behavior tests for optional open-weight dialogue planning.

No Ollama daemon, model download, real customer data, or DB needed.
"""
import copy
import json
import os
from unittest import TestCase
from unittest.mock import Mock, patch

from tools import servy_real_demo_local_planner as planner
from tools.servy_real_demo_dialogue import focus_answer, conversation_recap
from tools.test_servy_real_demo_dialogue import contract


class LocalOllamaPlannerTests(TestCase):
    def setUp(self):
        self.old_cooldown = planner._UNAVAILABLE_UNTIL
        planner._UNAVAILABLE_UNTIL = 0.0

    def tearDown(self):
        planner._UNAVAILABLE_UNTIL = self.old_cooldown

    def _call(self, answer, *, question="What causes weak cooling?", context=None):
        session = Mock()
        session.trust_env = True
        response = Mock()
        response.status_code = 200
        response.content = b'{"message":{"content":"selected"}}'
        response.json.return_value = {"message": {"content": json.dumps(answer)}}
        session.post.return_value = response
        with patch.dict(os.environ, {
            "SERVY_REAL_DEMO_DIALOGUE_MODE": "local",
            "SERVY_REAL_DEMO_CHAT_MODEL": "qwen2.5:3b",
            "SERVY_OLLAMA_URL": "https://evil.example.invalid/api/generate",
        }), patch.object(planner, "_online", return_value=True), \
             patch.object(planner.requests, "Session", return_value=session):
            result = planner.plan_turn(
                question, history=["The unit doesn't cool", "No fault code"],
                last_topic="causes", contract=context or contract(),
            )
        return result, session

    def test_local_planner_structured_selection_is_fixed_to_allowed_ids(self):
        choice = {
            "intent": "causes", "claim_ids": ["c2", "c1"],
            "follow_up": planner.FOLLOW_UPS["causes"][1], "style": "friendly",
        }
        result, session = self._call(choice)
        self.assertEqual(result, choice)
        self.assertFalse(session.trust_env)
        (args, kwargs) = session.post.call_args
        self.assertEqual(args, (planner.ENDPOINT,))
        self.assertEqual(planner.ENDPOINT, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(kwargs["json"]["stream"])
        self.assertFalse(kwargs["json"]["think"])
        self.assertEqual(kwargs["json"]["format"]["additionalProperties"], False)
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:3b")
        packed = kwargs["json"]["messages"][1]["content"]
        self.assertNotIn("synthetic-evidence-1", packed)  # Never send citation tokens.
        self.assertIn('"allowed_claim_ids"', packed)
        session.close.assert_called_once_with()

    def test_model_may_choose_relevant_verified_claims_but_cannot_invent_actions(self):
        result, _ = self._call({
            "intent": "observations", "claim_ids": ["c3"],
            "follow_up": planner.FOLLOW_UPS["observations"][0],
            "style": "friendly",
        }, question="and then?")
        reply, _ = focus_answer(
            contract(), "and then?", last_topic="causes",
            previous_facts=[], plan=result,
        )
        p = reply["presentation"]
        self.assertEqual([c["id"] for c in p["sections"][0]["claims"]], ["c3"])
        self.assertEqual(reply["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual(p["follow_up_question"], planner.FOLLOW_UPS["observations"][0])
        self.assertNotIn("Reduced airflow", reply["answer"])
        self.assertTrue(all(c["evidence_ids"] for c in p["supporting_claims"]))

    def test_explicit_cause_question_overrides_model_intent_and_keeps_citations(self):
        reply, _ = focus_answer(contract(), "What causes cooling trouble?", plan={
            "intent": "next", "claim_ids": ["c3"],
            "follow_up": planner.FOLLOW_UPS["next"][0], "style": "friendly",
        })
        self.assertEqual(reply["presentation"]["sections"][0]["title"], "Documented possible causes")
        self.assertEqual([x["id"] for x in reply["presentation"]["citations"]],
                         ["s1", "s2"])

    def test_unknown_claim_or_unauthorized_followup_forces_safe_fallback(self):
        invalid = [
            {"intent": "causes", "claim_ids": ["c999"], "follow_up": "",
             "style": "direct"},
            {"intent": "causes", "claim_ids": ["c1"], "follow_up": "Open the panel.",
             "style": "direct"},
            {"intent": "causes", "claim_ids": ["c1"], "follow_up": "",
             "style": "direct", "answer": "Invented diagnosis"},
            {"intent": "causes", "claim_ids": ["c1", "c1"], "follow_up": "",
             "style": "direct"},
        ]
        for answer in invalid:
            result, _ = self._call(answer)
            self.assertIsNone(result)

    def test_offline_and_opt_out_do_not_make_http_requests(self):
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "rules"}):
            with patch.object(planner, "_online", side_effect=AssertionError("network")):
                self.assertIsNone(planner.plan_turn(
                    "Why?", history=[], last_topic=None, contract=contract()
                ))
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "local"}):
            with patch.object(planner, "_online", return_value=False), \
                 patch.object(planner.requests, "Session", side_effect=AssertionError("network")):
                self.assertIsNone(planner.plan_turn(
                    "Why?", history=[], last_topic=None, contract=contract()
                ))

    def test_model_output_is_not_permitted_to_add_new_sources(self):
        changed = copy.deepcopy(contract())
        changed["presentation"]["citations"][0]["url"] = "https://untrusted.invalid"
        ids = [item["id"] for item in planner._candidate_evidence(changed)]
        self.assertNotIn("c1", ids)
        self.assertIn("c2", ids)
        changed["presentation"]["supporting_claims"] = []
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "local"}):
            with patch.object(planner, "_online", side_effect=AssertionError("network")):
                self.assertIsNone(planner.plan_turn(
                    "Why?", history=[], last_topic=None, contract=changed,
                ))

    def test_invalid_model_name_does_not_send_a_request(self):
        with patch.dict(os.environ, {
            "SERVY_REAL_DEMO_DIALOGUE_MODE": "local",
            "SERVY_REAL_DEMO_CHAT_MODEL": "../../../model\nexfil",
        }), patch.object(planner, "_online", return_value=True), \
             patch.object(planner.requests, "Session", side_effect=AssertionError("network")):
            self.assertIsNone(planner.plan_turn(
                "Why?", history=[], last_topic=None, contract=contract(),
            ))

    def test_local_model_failure_does_not_break_source_answer(self):
        session = Mock()
        session.post.side_effect = planner.requests.ConnectionError("offline")
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "local"}), \
             patch.object(planner, "_online", return_value=True), \
             patch.object(planner.requests, "Session", return_value=session):
            result = planner.plan_turn(
                "Why?", history=[], last_topic=None, contract=contract(),
            )
        self.assertIsNone(result)
        self.assertGreater(planner._UNAVAILABLE_UNTIL, 0)
        fallback, _ = focus_answer(contract(), "What causes weak cooling?", plan=result)
        self.assertEqual(fallback["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual(len(fallback["presentation"]["sections"]), 1)
        session.close.assert_called_once_with()

    def test_customer_recap_uses_only_signed_customer_turns_not_model_facts(self):
        history = ["AC is not cooling", "No code is displayed", "It still runs"]
        recap = conversation_recap("Summarize my issue", history)
        self.assertIn("AC is not cooling", recap)
        self.assertIn("No code is displayed", recap)
        self.assertIn("not a verified diagnosis", recap)
        self.assertNotIn("refrigerant leak", recap)
        self.assertIsNone(conversation_recap("What causes it?", history))
        self.assertIsNone(conversation_recap("Summarize my issue", []))
