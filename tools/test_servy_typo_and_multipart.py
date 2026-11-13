"""Regressions for misspelled and multi-part customer questions.

Both failures were found in the live three-asset evaluation: retrieval ranked
the right passage first, but (1) the evidence checks could not match
misspelled words, and (2) the answer focus dropped every cause of a question
that named two conditions. Answer-stage tests run over the real source files
with oracle retrieval (see ``test_servy_answer_stage``) and skip without them.
"""
import os
import unittest

os.environ.setdefault("LOCALAPPDATA", "/tmp")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from core.services.retriever import _correct_spelling, normalize_domain_terms
from tools.test_servy_answer_stage import HAVE_DOCS, ask


class SpellingCorrectionTests(unittest.TestCase):
    def test_misspellings_from_the_evaluation_are_corrected(self):
        cases = {
            "how ofen shuld the filtres be cleened?": "how often should the filters be cleaned?",
            "wat causes a comunication or adressing fault?":
                "what causes a communication or addressing fault?",
            "wat shud be done evry 6 months?": "what should be done every 6 months?",
            "my ac is runing but not coolng properly, what cuases this?":
                "my ac is running but not cooling properly, what causes this?",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_domain_terms(raw), expected)

    def test_identifiers_numbers_and_known_words_never_change(self):
        for text in ("What does fault code E-17 mean on model APX-2T-INV?",
                     "Is the loud noise from the fan?",
                     "The seal is fine but the coil is cold.",
                     "Every 6 months: document airflow, static pressure.",
                     "Error ERR42 on unit VRF-OUT-TS-001"):
            with self.subTest(text=text):
                self.assertEqual(normalize_domain_terms(text), text)

    def test_correction_is_bounded_and_unambiguous(self):
        self.assertIsNone(_correct_spelling("abc"))          # too short
        self.assertIsNone(_correct_spelling("e17x"))         # not alphabetic
        self.assertIsNone(_correct_spelling("zzzzzz"))       # nothing close
        self.assertIsNone(_correct_spelling("filter"))       # already known
        self.assertIsNone(_correct_spelling("kooling"))      # first letter must match
        self.assertEqual(_correct_spelling("compresor"), "compressor")
        # Real words two edits from a known word are never "corrected".
        self.assertIsNone(_correct_spelling("indicate"))
        self.assertIsNone(_correct_spelling("detection"))   # not a near-miss of a known word
        self.assertEqual(normalize_domain_terms("What does E-04 indicate?"),
                         "What does E-04 indicate?")


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class MisspelledQuestionAnswerTests(unittest.TestCase):
    def test_misspelled_questions_reach_the_documented_answer(self):
        cases = (
            ("split", "how ofen shuld the filtres be cleened?", "Quarterly: clean filters",
             "2. SUGGESTED MAINTENANCE INTERVALS"),
            ("vrf", "wat causes a comunication or adressing fault?",
             "Communication cable damage or incorrect topology.",
             "2. COMMUNICATION OR ADDRESSING FAULT"),
            ("duct", "wat shud be done evry 6 months?", "Every 6 months: document airflow",
             "2. SUGGESTED INTERVALS"),
            ("split", "my ac is runing but not coolng properly, what cuases this?",
             "Dirty filter or coil.", "3. SYMPTOM: POOR COOLING"),
        )
        for asset, question, expected, heading in cases:
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE")
                self.assertTrue(any(text.startswith(expected) for text in r.texts), r.texts)
                self.assertIn(heading, {h for _t, h in r.cited})


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class MultiPartCauseQuestionTests(unittest.TestCase):
    def test_each_named_condition_keeps_exactly_its_own_causes(self):
        cases = (
            ("split", "What can cause poor cooling, and what can cause water leakage?",
             {"3. SYMPTOM: POOR COOLING", "4. SYMPTOM: WATER LEAKAGE"}),
            ("vrf", "What causes a high-pressure fault and what causes a low-pressure fault?",
             {"3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT", "4. LOW-PRESSURE OR CAPACITY FAULT"}),
            ("duct", "What causes weak airflow, and what causes uneven room temperatures?",
             {"2. SYMPTOM: WEAK AIRFLOW", "3. SYMPTOM: UNEVEN ROOM TEMPERATURES"}),
        )
        for asset, question, headings in cases:
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE")
                self.assertEqual({h for _t, h in r.cited}, headings)
                self.assertEqual(set(r.kinds), {"possible_cause"})

    def test_single_condition_questions_are_unchanged(self):
        r = ask("duct", "What causes weak airflow?")
        self.assertEqual({h for _t, h in r.cited}, {"2. SYMPTOM: WEAK AIRFLOW"})
        r = ask("split", "What can cause water leakage from the unit?")
        self.assertEqual({h for _t, h in r.cited}, {"4. SYMPTOM: WATER LEAKAGE"})

    def test_a_named_component_question_is_not_widened(self):
        # Two subjects joined by "and" that are not two fault sections keep
        # the existing subject focus.
        r = ask("split", "Could the filter and the coil cause poor cooling?")
        self.assertEqual(r.texts, ["Dirty filter or coil."])


if __name__ == "__main__":
    unittest.main()
