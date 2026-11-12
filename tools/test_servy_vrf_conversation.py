"""Regression tests for the reproduced VRF customer conversation.

Runs consecutive turns through the real demo answer path over the real VRF
source documents (oracle retrieval, see ``test_servy_answer_stage``). Covers:
no "possible causes" lead-in without cited causes, a manual-based
clarification asked once, observations tied to their documented condition,
context kept across turns, honest document-coverage status, and asset
isolation. No database, network, model or ticket store is used.
"""
import json
import os
import re
import unittest

# These tests import Django-backed answer services directly. Configure the
# project before importing those modules when the test is run standalone.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("LOCALAPPDATA", "/tmp")

import django
django.setup()

from tools.servy_real_demo_dialogue import CATEGORY_QUESTION, _claim_topic, _match_kinds
from core.services.customer_answer import source_claims, _descriptive_cause
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


    def test_checklist_and_observation_bullets_inside_cause_lists_are_not_causes(self):
        claims = list(source_claims(
            "Possible causes:\n"
            "- Communication cable damage or incorrect topology.\n"
            "- [ ] Look for oil residue or other evidence of a refrigerant leak.\n"
            "- Visible oil residue may indicate a refrigerant leak.\n"
            "- Inspect the internal wiring for damage.\n"
            "- Low refrigerant charge.\n"
        ))
        self.assertEqual([kind for _text, kind, _start, _end in claims], [
            "possible_cause", "information", "information", "information", "possible_cause",
        ])
        self.assertIn("Look for oil residue", claims[1][0])
        self.assertFalse(_descriptive_cause("[ ] Look for oil residue or evidence of a leak."))
        self.assertFalse(_descriptive_cause("Inspect the internal wiring for damage."))
        self.assertTrue(_descriptive_cause("Low refrigerant charge."))

@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class VrfConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first, cls.second, cls.third = converse("vrf", VRF_TURNS)

    def test_turn_one_returns_cited_conditional_causes_not_only_fault_headings(self):
        r = self.first
        presentation = r["presentation"]
        self.assertEqual(r["response_type"], "DIAGNOSTIC_GUIDANCE")
        claims = presentation.get("supporting_claims") or []
        self.assertTrue(claims)
        self.assertTrue(all(claim["kind"] == "possible_cause" for claim in claims))
        self.assertTrue(r["citations"])
        self.assertIn("none is a confirmed diagnosis", presentation["message"])
        self.assertTrue(all(claim["text"].startswith("Under ") for claim in claims))
        self.assertTrue(all(claim["evidence_ids"] for claim in claims))
        dump = json.dumps(presentation)
        self.assertNotIn("qualified VRF/HVAC", dump)
        self.assertNotIn("Exact alarm definitions", dump)

    def test_turn_one_asks_for_a_customer_visible_observation_after_the_cited_overview(self):
        r = self.first
        presentation = r["presentation"]
        self.assertIn("none is a confirmed diagnosis", presentation["message"])
        self.assertEqual(
            presentation.get("follow_up_question"),
            "Is a fault code or warning visible on the display?",
        )
        self.assertNotIn("Which of these documented conditions", presentation["message"])
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

    def test_turn_three_keeps_context_and_does_not_fall_back_to_a_category_picker(self):
        r = self.third
        self.assertNotIn("Which of these documented conditions", r["presentation"]["message"])
        if r["response_type"] == "DIAGNOSTIC_GUIDANCE":
            self.assertTrue(r["citations"])
            self.assertTrue(any(
                claim["kind"] == "possible_cause"
                for claim in r["presentation"].get("supporting_claims", [])
            ))
        else:
            self.assertEqual(r["presentation"].get("supporting_claims"), [])

    def test_exact_customer_wording_clarifies_then_answers_communication_causes(self):
        results = converse("vrf", [
            "What could be the possible reasons why my VRF outdoor module is not working?",
            "I think it may be a communication or addressing fault. What causes are documented in the approved manual?",
            "Explain the second documented cause of the communication or addressing fault. What does the approved manual say, and what does it not establish?",
        ])
        first = results[0]
        self.assertEqual(first["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertTrue(first["citations"])
        first_message = first["presentation"]["message"]
        self.assertIn("none is a confirmed diagnosis", first_message)
        self.assertTrue(all(
            claim["text"].startswith("Under ")
            for claim in first["presentation"]["supporting_claims"]
        ))
        self.assertEqual(
            first["presentation"]["follow_up_question"],
            "Is a fault code or warning visible on the display?",
        )

        rows = _shown(results[1])
        self.assertEqual([r[3] for r in rows], [
            "Communication cable damage or incorrect topology.",
            "Duplicate/missing addresses or configuration mismatch.",
            "Power loss to a connected unit or module."])
        self.assertEqual({r[4] for r in rows}, {"2. COMMUNICATION OR ADDRESSING FAULT"})
        self.assertIn("lists these possibilities", results[1]["presentation"]["message"])
        followup = _shown(results[2])
        self.assertEqual(followup[0][3], "Duplicate/missing addresses or configuration mismatch.")
        self.assertNotIn(
            "Evidence limitation",
            [section["title"] for section in results[2]["presentation"]["sections"]],
        )
        self.assertTrue(results[2]["presentation"]["cause_explanation_unavailable"])
        self.assertNotIn(
            "contain no further explanation",
            json.dumps(results[2]["presentation"]).lower(),
        )
        self.assertEqual(results[2]["evidence_state"], "PARTIAL")

    def test_selected_communication_fault_stays_scoped_across_followups(self):
        results = converse("vrf", [
            "What could be the possible reasons why my VRF outdoor module is not working?",
            "I think it may be a communication or addressing fault. What causes are documented in the approved manual?",
            "Explain the second documented cause of the communication or addressing fault. What does the manual say, and what does it not establish?",
            "What can I safely observe?",
            "What should I do next?",
            "What possible causes are documented?",
        ])

        expected_causes = [
            "Communication cable damage or incorrect topology.",
            "Duplicate/missing addresses or configuration mismatch.",
            "Power loss to a connected unit or module.",
        ]
        for index in (1, 5):
            rows = _shown(results[index])
            self.assertEqual([row[3] for row in rows], expected_causes)
            self.assertEqual(
                {row[4] for row in rows},
                {"2. COMMUNICATION OR ADDRESSING FAULT"},
            )
            self.assertEqual({row[1] for row in rows}, {"possible_cause"})

        for index in (3, 4):
            rows = _shown(results[index])
            self.assertTrue(rows)
            self.assertTrue(all(
                row[4] == "2. COMMUNICATION OR ADDRESSING FAULT" for row in rows
            ))
            self.assertTrue(all(
                row[1] in {"observation", "customer_procedure", "next_step"}
                for row in rows
            ))

        self.assertEqual(
            _shown(results[2])[0][3],
            "Duplicate/missing addresses or configuration mismatch.",
        )
        self.assertNotIn(
            "Evidence limitation",
            [section["title"] for section in results[2]["presentation"]["sections"]],
        )
        self.assertTrue(results[2]["presentation"]["cause_explanation_unavailable"])


    def test_short_followup_answers_are_acknowledged_without_rag_retrieval(self):
        results = converse("vrf", [
            "What could be the possible reasons why my VRF outdoor module is not working?",
            "I think it may be a communication or addressing fault. What causes are documented in the approved manual?",
            "no code",
            "suddenly",
            "intermittently",
        ])
        self.assertEqual(results[2]["response_type"], "CLARIFICATION")
        self.assertEqual(results[2]["retrieval_attempts"], 0)
        self.assertIn("suddenly or gradually", results[2]["clarification_question"])
        self.assertEqual(results[3]["response_type"], "CLARIFICATION")
        self.assertEqual(results[3]["retrieval_attempts"], 0)
        self.assertIn("problem started suddenly", results[3]["message"])
        self.assertIn("continuously or intermittently",
                      results[3]["clarification_question"])
        self.assertEqual(results[4]["response_type"], "CLARIFICATION")
        self.assertEqual(results[4]["retrieval_attempts"], 0)
        self.assertIn("problem happens intermittently", results[4]["message"])

    def test_generic_vrf_conversation_captures_no_code_then_sudden_and_intermittent(self):
        # Reproduces the customer-visible failure: no manual-category selection
        # is required before "no code" and "suddenly" can be captured.
        results = converse("vrf", [
            "What could be the possible reasons why my VRF outdoor module is not working?",
            "No code.",
            "Suddenly.",
            "Intermittently.",
        ])

        first = results[0]
        first_message = first["presentation"]["message"]
        self.assertEqual(first["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertTrue(first["citations"])
        self.assertIn("none is a confirmed diagnosis", first_message)
        self.assertEqual(
            first["presentation"]["follow_up_question"],
            "Is a fault code or warning visible on the display?",
        )
        self.assertNotIn("Which of these documented conditions", first_message)

        no_code = results[1]
        self.assertEqual(no_code["response_type"], "CLARIFICATION")
        self.assertEqual(no_code["retrieval_attempts"], 0)
        self.assertIn("no fault code or warning is visible", no_code["message"])
        self.assertIn("suddenly or gradually", no_code["clarification_question"])

        sudden = results[2]
        self.assertEqual(sudden["response_type"], "CLARIFICATION")
        self.assertEqual(sudden["retrieval_attempts"], 0)
        self.assertIn("problem started suddenly", sudden["message"])
        self.assertIn("continuously or intermittently",
                      sudden["clarification_question"])

        intermittent = results[3]
        self.assertEqual(intermittent["response_type"], "CLARIFICATION")
        self.assertEqual(intermittent["retrieval_attempts"], 0)
        self.assertIn("problem happens intermittently", intermittent["message"])

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
