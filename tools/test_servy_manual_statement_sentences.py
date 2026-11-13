"""A manual line that joins a statement with an instruction: answer-stage tests.

The nine source documents often write "<what this covers>. <instruction>." on
one line. The line as a whole must stay undisclosed, but the statement sentence
may answer a customer's "what does this guide cover" question when it is cited
by its own exact span. Runs the real answer stage over the real source files
with oracle retrieval (see ``test_servy_answer_stage``); it says nothing about
retrieval ranking. Skips when the git-ignored source files are absent.
"""
import os
import re
import unittest
from urllib.parse import parse_qs, urlsplit

os.environ.setdefault("LOCALAPPDATA", "/tmp")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from django.core import signing

from core.services import customer_answer as ca
from core.services.term_matching import stem
from tools.test_servy_answer_stage import HAVE_DOCS, ask

INSTRUCTION = re.compile(r"^(?:Confirm|Provide|Adjust|Use|Do not|Never)\b")


class SentenceSpanTests(unittest.TestCase):
    def test_only_multi_sentence_lines_are_split_and_spans_are_exact(self):
        chunk = "- Alpha statement here. Confirm the setting first.\n- Second line only."
        line = "Alpha statement here. Confirm the setting first."
        start = chunk.index("- Alpha")
        end = start + len("- " + line)
        spans = ca._sentence_spans(chunk, line, start, end)
        self.assertEqual([s for s, _a, _b in spans],
                         ["Alpha statement here.", "Confirm the setting first."])
        for sentence, a, b in spans:
            self.assertEqual(chunk[a:b], sentence)
        self.assertEqual(ca._sentence_spans(chunk, "Second line only.", 0, len(chunk)), [])

    def test_sentence_not_found_in_its_own_span_is_dropped(self):
        self.assertEqual(ca._sentence_spans("Unrelated text.", "One thing. Another thing.", 0, 15), [])

    def test_verb_and_plural_forms_fold_to_the_same_key(self):
        self.assertEqual(stem("provides"), stem("provide"))
        self.assertEqual(stem("passes"), "pass")
        self.assertEqual(stem("boxes"), "box")
        self.assertEqual(stem("causes"), "cause")
        self.assertEqual(stem("rooms"), "room")
        self.assertEqual(stem("E-04"), "e-04")          # identifiers never change


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class StatementSentenceAnswerTests(unittest.TestCase):
    CASES = (
        ("split", "What does the installation and commissioning guide provide?",
         "This document provides a practical checklist for installing and commissioning "
         "a nominal 2-ton split air-conditioning unit.", "1. PURPOSE",
         "Installation and Commissioning Guide"),
        ("duct", "What does the installation and commissioning guide cover?",
         "Generic guide for installation and commissioning of a nominal 5-ton ductable "
         "air-conditioning system.", "1. PURPOSE", "Installation and Commissioning Guide"),
        ("vrf", "What should the maintenance frequency be based on?",
         "Frequency should be based on manufacturer guidance, runtime, environment, and "
         "service history.", "1. PURPOSE", "Preventive Maintenance Checklist"),
    )

    def test_statement_sentence_answers_and_instruction_sentence_does_not(self):
        for asset, question, expected, heading, title in self.CASES:
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE")
                self.assertIn(expected, r.texts)
                self.assertIn((title, heading), r.cited)
                self.assertEqual(set(r.kinds), {"information"})
                for text in r.texts:
                    self.assertFalse(INSTRUCTION.match(text), text)
                self.assertNotIn("Confirm all requirements", r.dump)
                self.assertNotIn("Confirm unit type", r.dump)
                self.assertNotIn("Provide a documented maintenance routine", r.dump)

    def test_each_citation_opens_to_exactly_the_cited_sentence(self):
        for asset, question, expected, _heading, _title in self.CASES:
            with self.subTest(question=question):
                r = ask(asset, question)
                citations = {c["id"]: c for c in r.raw["presentation"]["citations"]}
                claim = next(c for c in r.claims if c["source_text"] == expected)
                citation = citations[claim["evidence_ids"][0]]
                token = parse_qs(urlsplit(citation["url"]).query)["citation"][0]
                payload = signing.loads(token, salt=ca.SALT)
                chunk = r.chunks[payload["chunk"]]
                self.assertEqual(chunk.text[payload["start"]:payload["end"]], expected)
                self.assertEqual(claim["text"], expected)          # shown verbatim

    def test_safety_classification_still_blocks_a_restricted_statement(self):
        # The VRF installation guide's purpose sentence contains a term the
        # existing safety classifier restricts. It must remain undisclosed.
        r = ask("vrf", "What does the VRF installation and commissioning guide cover?")
        self.assertNotIn("Variable Refrigerant Flow", r.dump)

    def test_purpose_written_as_an_instruction_is_not_disclosed(self):
        r = ask("vrf", "What does the VRF preventive maintenance checklist provide?")
        self.assertNotIn("Provide a documented maintenance routine", r.dump)

    def test_splitting_does_not_surface_instructions_for_procedure_questions(self):
        for asset, question in (
            ("split", "How should the refrigerant circuit be charged?"),
            ("vrf", "How is the pressure test performed?"),
            ("duct", "How do I change the fan speed setting?"),
            ("vrf", "What vacuum level should be reached during evacuation?"),
        ):
            with self.subTest(question=question):
                r = ask(asset, question)
                # A cited manual warning ("Do not ...") may accompany a refusal;
                # no instruction may appear as information or as a procedure.
                for claim in r.claims:
                    self.assertNotEqual(claim["kind"], "customer_procedure")
                    if claim["kind"] != "safety_warning":
                        self.assertFalse(INSTRUCTION.match(claim["source_text"]),
                                         claim["source_text"])
                self.assertFalse(re.search(
                    r"pressure-test|evacuate|charge the refrigerant|brazing|interlock",
                    " ".join(r.texts), re.I), r.texts)


if __name__ == "__main__":
    unittest.main()
