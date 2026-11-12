"""Regression tests for the reproduced VRF customer conversation.

Runs consecutive turns through the real demo answer path over the real VRF
source documents (oracle retrieval, see ``test_servy_answer_stage``). Covers:
no "possible causes" lead-in without cited causes, a manual-based
clarification asked once, observations tied to their documented condition,
context kept across turns, honest document-coverage status, and asset
isolation. No database, network, model or ticket store is used.
"""
import json
import re
import unittest

from tools.servy_real_demo_dialogue import CATEGORY_QUESTION, _claim_topic, _match_kinds
from core.services.customer_answer import source_claims
from tools.test_servy_answer_stage import ASSETS, HAVE_DOCS, _chunks, _shown, converse

VRF_TURNS = [
    "What could be the issues if the VRF outdoor module is not working?",
    "What can I safely observe?",
    "What documented causes could explain this?",
]
VRF_CONDITIONS = (
    "Communication or addressing fault", "High-pressure or heat-exchange fault",
    "Low-pressure or capacity fault", "Compressor or inverter fault",
    "Fan or temperature-sensor fault",
)
RESTRICTED = re.compile(
    r"pressure-test|evacuate|refrigerant charge|brazing|isolate power|interlock|"
    r"bypass|lockout|electrical measurements|inverter diagnostics", re.I)


class ClaimCategoryIsolationTests(unittest.TestCase):
    def test_cause_topic_accepts_only_explicitly_classified_causes(self):
        self.assertEqual(_match_kinds("causes"), frozenset({"possible_cause"}))

    def test_information_and_document_facts_never_inherit_the_cause_topic(self):
        self.assertEqual(_claim_topic({"kind": "information"}, "causes"), "documents")
        self.assertEqual(_claim_topic({"kind": "document_fact"}, "causes"), "documents")
        self.assertEqual(_claim_topic({"kind": "observation"}, "causes"), "observations")
        self.assertEqual(_claim_topic({"kind": "customer_procedure"}, "causes"), "next")

    def test_only_explicit_causal_language_classifies_plain_prose_as_a_cause(self):
        claims = list(source_claims(
            "The compressor is caused by an inverter fault.\n"
            "The alarm indicates a condition requiring review.\n"
            "This code means the equipment has entered a protected state.\n"
            "The alarm occurs when communication is interrupted.\n"
        ))
        self.assertEqual(
            [kind for _text, kind, _start, _end in claims],
            ["possible_cause", "information", "information", "information"],
        )


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class VrfConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first, cls.second, cls.third = converse("vrf", VRF_TURNS)

    def test_turn_one_does_not_claim_causes_the_citations_do_not_establish(self):
        r = self.first
        presentation = r["presentation"]
        self.assertEqual(r["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("lists these possibilities", presentation["message"])
        self.assertEqual(presentation.get("supporting_claims"), [])
        self.assertEqual(r["citations"], [])
        dump = json.dumps(presentation)
        self.assertNotIn("qualified VRF/HVAC", dump)          # a warning is not a cause
        self.assertNotIn("Exact alarm definitions", dump)

    def test_turn_one_states_the_limit_and_asks_one_question_from_the_manual(self):
        r = self.first
        message = r["presentation"]["message"]
        self.assertIn("does not list causes for the problem as you described it", message)
        for condition in VRF_CONDITIONS:
            self.assertIn(condition, message)
        self.assertEqual(message.count("?"), 1)
        self.assertTrue(message.endswith(CATEGORY_QUESTION))
        self.assertEqual(r["reason_code"], "NEEDS_SYMPTOM_CLARIFICATION")
        # A clarification is not yet an escalation; the ticket button remains.
        self.assertIsNone(r["issue_assignment"])
        self.assertTrue(r["ticket_context"])

    def test_listed_conditions_are_the_manual_cause_section_headings(self):
        chunks = _chunks(ASSETS["vrf"][2], 600)
        headings = {re.sub(r"^\d+\.\s*", "", c.heading).capitalize()
                    for c in chunks.values() if "Possible causes:" in c.text}
        self.assertEqual(headings, set(VRF_CONDITIONS))

    def test_turn_two_ties_each_observation_to_its_documented_condition(self):
        rows = _shown(self.second)
        self.assertTrue(rows)
        message = self.second["presentation"]["message"]
        self.assertIn("under specific conditions", message)
        self.assertIn("does not establish which condition applies", message)
        chunks = _chunks(ASSETS["vrf"][2], 600)
        for _title, kind, text, source, heading in rows:
            self.assertEqual(kind, "observation")
            section = re.sub(r"^\d+\.\s*", "", heading).capitalize()
            self.assertTrue(text.startswith("Listed for “" + section + "”: "), text)
            self.assertTrue(text.endswith(source))          # the sentence is verbatim
            chunk = next(c for c in chunks.values() if c.heading == heading)
            self.assertIn(source, chunk.text)
            self.assertFalse(RESTRICTED.search(text), text)

    def test_turn_three_keeps_context_and_does_not_repeat_the_question(self):
        r = self.third
        message = r["presentation"]["message"]
        self.assertNotIn(CATEGORY_QUESTION, message)
        self.assertNotIn("What are you actually observing", message)
        self.assertIn("I won't guess a cause", message)
        self.assertIn("Communication or addressing fault", message)
        self.assertEqual(r["presentation"].get("supporting_claims"), [])
        block = r["issue_assignment"]
        self.assertEqual(block["status"], "AWAITING_CUSTOMER_CONFIRMATION")
        self.assertIn("VRF Outdoor Module", block["description"])
        self.assertIn("not working", block["description"])

    def test_answering_the_clarification_returns_the_documented_causes(self):
        results = converse("vrf", [VRF_TURNS[0],
                                   "It is a communication fault. What are the causes?",
                                   "Explain the second cause."])
        rows = _shown(results[1])
        self.assertEqual([r[3] for r in rows], [
            "Communication cable damage or incorrect topology.",
            "Duplicate/missing addresses or configuration mismatch.",
            "Power loss to a connected unit or module."])
        self.assertEqual({r[4] for r in rows}, {"2. COMMUNICATION OR ADDRESSING FAULT"})
        self.assertIn("lists these possibilities", results[1]["presentation"]["message"])
        followup = _shown(results[2])
        self.assertEqual(followup[0][3], "Duplicate/missing addresses or configuration mismatch.")
        self.assertEqual(results[2]["evidence_state"], "PARTIAL")

    def test_unsupported_fault_code_is_not_explained(self):
        (r,) = converse("vrf", ["The outdoor module shows fault code E-99. What does E-99 mean?"])
        self.assertEqual(r["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(r["citations"], [])
        self.assertNotIn("Possible documented cause", json.dumps(r["presentation"]))

    def test_incomplete_scan_is_reported_as_unavailable_not_as_zero_documents(self):
        (r,) = converse("vrf", ["Why is there a communication fault?"])
        coverage = r["presentation"]["document_coverage"]
        self.assertTrue(r["citations"])                       # ordinary retrieval still cited
        self.assertIs(coverage["complete"], False)
        self.assertIs(coverage["scan_completed"], False)
        self.assertIn("no review count is available", coverage["note"])

    def test_completed_scan_keeps_its_real_counts(self):
        scan = {"references": [], "coverage": {
            "complete": True, "scope": "selected-asset-approved-documents",
            "documents_searched": 3, "approved_documents": 3,
            "documents_without_searchable_chunks": 0, "passages_searched": 19,
            "documents_with_matches": 1, "matching_passages": 3,
            "references_in_answer_pool": 3, "document_titles": [], "note": "Searched all 3."}}
        (r,) = converse("vrf", ["Why is there a communication fault?"], scan=scan)
        coverage = r["presentation"]["document_coverage"]
        self.assertEqual(coverage["documents_searched"], 3)
        self.assertNotIn("scan_completed", coverage)

    def test_another_asset_gets_its_own_conditions_and_cannot_reuse_the_chat(self):
        (split,) = converse("split", ["What could be the issues if the unit is not working?"],
                            thread_id="thread-answer-stage-0002")
        message = split["presentation"]["message"]
        self.assertIn("Poor cooling", message)
        self.assertNotIn("Communication or addressing fault", message)
        with self.assertRaises(ValueError):
            converse("split", ["What can I safely observe?"],
                     continuation=self.first["continuation"])


if __name__ == "__main__":
    unittest.main()
