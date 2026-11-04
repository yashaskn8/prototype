"""Offline regression tests for the focused, document-bound demo dialogue."""
import copy
import unittest

from tools.servy_real_demo_dialogue import (
    conversational_turn, fact_fingerprint, focus_answer, question_topic,
)


def contract():
    cause_a = {
        "id": "c1", "text": "Possible documented cause: Reduced airflow.",
        "source_text": "Reduced airflow.", "kind": "possible_cause", "evidence_ids": ["s1"],
    }
    cause_b = {
        "id": "c2", "text": "Possible documented cause: Blocked external clearance.",
        "source_text": "Blocked external clearance.", "kind": "possible_cause", "evidence_ids": ["s2"],
    }
    obs = {
        "id": "c3", "text": "Observe the external displayed error code.",
        "source_text": "Observe the external displayed error code.",
        "kind": "observation", "evidence_ids": ["s3"],
    }
    meaning = {
        "id": "c4", "text": "E-04 indicates reduced airflow.",
        "source_text": "E-04 indicates reduced airflow.",
        "kind": "information", "evidence_ids": ["s4"],
    }
    doc = {
        "id": "c5", "text": "The source lists a cooling fault record.",
        "source_text": "A cooling fault record.", "kind": "document_fact", "evidence_ids": ["s5"],
    }
    citations = [
        {
            "id": f"s{i}", "document_id": 3, "chunk_id": 8+i,
            "title": "Synthetic authorized cooling manual",
            "url": "/demo-api/source/?citation=synthetic-evidence-" + str(i),
        } for i in range(1, 6)
    ]
    return {
        "response_type": "DIAGNOSTIC_GUIDANCE",
        "response_mode": "ANSWER_WITH_VERIFIED_EVIDENCE",
        "can_resolve": True,
        "customer_safe_actions": [{"action_id": "c3", "action": obs["text"]}],
        "presentation": {
            "schema_version": "customer-answer-v1",
            "response_type": "DIAGNOSTIC_GUIDANCE",
            "equipment": {"id": 60, "name": "Synthetic Split AC"},
            "sections": [
                {"number": 1, "title": "What could be wrong?", "claims": [cause_a, cause_b, meaning]},
                {"number": 2, "title": "What can I safely check or report?", "claims": [obs]},
                {"number": 3, "title": "What is still unknown?", "claims": [
                    {"text": "Not confirmed.", "kind": "uncertainty", "evidence_ids": []}]},
                {"number": 4, "title": "What should I do next?", "claims": [
                    {"text": "Ask a technician.", "kind": "next_step", "evidence_ids": []}]},
            ],
            "citations": citations,
            "supporting_claims": [cause_a, cause_b, obs, meaning, doc],
            "customer_permitted_observations": [obs],
            "customer_authorized_actions": [],
            "message": "",
            "authorized_user_actions": {"can_resolve": True, "can_escalate": True},
        },
    }


class IntentFocusTests(unittest.TestCase):
    def test_user_asked_only_causes_not_four_predefined_questions(self):
        result, state = focus_answer(
            contract(), "My split AC is not cooling. What possible causes are documented?"
        )
        p = result["presentation"]
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual([s["title"] for s in p["sections"]], ["Documented possible causes"])
        self.assertTrue(p["view_mode"] == "focused")
        self.assertTrue(all(c["kind"] in {"possible_cause", "information"}
                            for c in p["sections"][0]["claims"]))
        self.assertEqual([c["id"] for c in p["citations"]], ["s1", "s2", "s4"])
        self.assertEqual(result["customer_safe_actions"], [])
        self.assertFalse(result["can_resolve"])
        self.assertEqual(state["topic"], "causes")
        self.assertEqual(len(state["facts"]), 3)
        self.assertIn("What can I safely observe?", state["suggestions"])

    def test_observations_never_include_cause_or_uncited_hardware_procedure(self):
        result, state = focus_answer(contract(), "What can I safely observe?")
        p = result["presentation"]
        self.assertEqual(state["topic"], "observations")
        self.assertEqual([c["kind"] for c in p["sections"][0]["claims"]], ["observation"])
        self.assertEqual([c["id"] for c in p["citations"]], ["s3"])
        self.assertNotIn("Reduced airflow", result["answer"])

    def test_next_without_approved_safe_evidence_is_not_a_cause_dump(self):
        base = contract()
        base["presentation"]["supporting_claims"] = base["presentation"]["supporting_claims"][:2]
        result, state = focus_answer(base, "What should I do next?", last_topic="causes")
        self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["presentation"]["sections"], [])
        self.assertEqual(result["presentation"]["citations"], [])
        self.assertIn("couldn't verify an approved", result["answer"])
        self.assertFalse(result["can_resolve"])
        self.assertEqual(state["suggestions"], [])

    def test_more_causes_does_not_repeat_already_shown_facts(self):
        initial, state = focus_answer(contract(), "What are the possible causes?")
        followup, state2 = focus_answer(
            contract(), "What other causes are documented?",
            last_topic=state["topic"], previous_facts=state["facts"],
        )
        self.assertEqual(followup["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(followup["presentation"]["citations"], [])
        self.assertIn("another distinct", followup["answer"])
        self.assertEqual(state2["facts"], state["facts"])
        repeated, _ = focus_answer(contract(), "What are the possible causes?",
                                   last_topic=state["topic"], previous_facts=state["facts"])
        self.assertEqual(repeated["response_type"], "DIAGNOSTIC_GUIDANCE")

    def test_more_causes_displays_only_new_source(self):
        base = contract()
        citation = base["presentation"]["citations"][0]
        claim = base["presentation"]["supporting_claims"][0]
        seen = [fact_fingerprint(claim, citation)]
        result, state = focus_answer(base, "Anything else?",
                                     last_topic="causes", previous_facts=seen)
        self.assertNotIn("s1", [s["id"] for s in result["presentation"]["citations"]])
        self.assertEqual(state["topic"], "causes")

    def test_code_report_routes_to_definition_not_generic_causes(self):
        self.assertEqual(question_topic("The display shows E-04"), "meaning")
        result, _ = focus_answer(contract(), "What does E-04 indicate?")
        self.assertEqual(result["presentation"]["sections"][0]["title"], "What the manual says")
        self.assertEqual([c["id"] for c in result["presentation"]["citations"]], ["s4"])

    def test_unresolved_check_report_routes_to_next(self):
        self.assertEqual(question_topic(
            "I already checked the visible filter but it is still not cooling", "causes"
        ), "next")

    def test_greetings_are_not_fake_technical_evidence(self):
        self.assertIn("what problem", conversational_turn("Hi").lower())
        self.assertIn("welcome", conversational_turn("Thanks").lower())
        self.assertIsNone(conversational_turn("Why is cooling weak?"))

    def test_policy_refusal_cannot_be_rewritten_into_diagnostic_guidance(self):
        base = contract()
        base["response_type"] = "POLICY_VIOLATION"
        untouched, state = focus_answer(base, "Can I bypass a safety interlock?")
        self.assertIs(untouched, base)
        self.assertEqual(state["suggestions"], [])

    def test_strict_citation_binding_drops_forged_and_uncited_claims(self):
        base = contract()
        mutated = copy.deepcopy(base)
        mutated["presentation"]["citations"][0]["url"] = "https://example.invalid/fake"
        mutated["presentation"]["supporting_claims"][1]["evidence_ids"] = ["nonexistent"]
        mutated["presentation"]["supporting_claims"][3]["source_text"] = None
        result, _ = focus_answer(mutated, "What possible causes are documented?")
        self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["presentation"]["citations"], [])

    def test_fingerprint_is_independent_of_response_local_citation_ids(self):
        base = contract()
        claim = base["presentation"]["supporting_claims"][0]
        citation = base["presentation"]["citations"][0]
        renamed = {**citation, "id": "different-response-local-id"}
        self.assertEqual(fact_fingerprint(claim, citation), fact_fingerprint(claim, renamed))


if __name__ == "__main__":
    unittest.main()
