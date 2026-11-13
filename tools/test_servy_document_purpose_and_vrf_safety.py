"""Regressions for two Asset 60 evaluation failures.

1. "What does the VRF preventive maintenance checklist provide?" was refused: the
   manual's PURPOSE line ("Provide a documented maintenance routine ...") starts
   with a verb, so it was treated as an instruction and never disclosed.
2. "What does the VRF installation and commissioning guide cover?" missed the
   PURPOSE line: "Variable Refrigerant Flow" (the system category's name) made
   the safety classifier mark the sentence technician-only.

Answer-stage tests use the real source files with oracle retrieval (see
``test_servy_answer_stage``) and skip without them.
"""
import os
import unittest

os.environ.setdefault("LOCALAPPDATA", "/tmp")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from core.services.customer_answer import document_purpose_statement
from core.services.customer_diagnostic import classify_action_safety
from tools.test_servy_answer_stage import HAVE_DOCS, ask

PURPOSE = "1. PURPOSE"


class DocumentPurposeStatementTests(unittest.TestCase):
    def _level(self, text):
        return classify_action_safety(text)[0]

    def test_purpose_section_self_descriptions_qualify(self):
        for text in ("Provide a documented maintenance routine for a VRF outdoor module and its associated system.",
                     "Use this checklist to plan routine inspection and maintenance for a 2-ton split AC.",
                     "Provides a reference for commissioning records."):
            with self.subTest(text=text):
                self.assertTrue(document_purpose_statement(text, self._level(text), PURPOSE))

    def test_only_purpose_or_scope_sections_qualify(self):
        text = "Provide a documented maintenance routine for a VRF outdoor module."
        self.assertTrue(document_purpose_statement(text, self._level(text), "1. SCOPE"))
        for heading in ("3. MONTHLY TASKS", "5. SAFETY", ""):
            with self.subTest(heading=heading):
                self.assertFalse(document_purpose_statement(text, self._level(text), heading))

    def test_instructions_and_physical_steps_never_qualify(self):
        for text in ("Provide the checklist and clean the condenser coil.",
                     "Use this checklist to plan work; isolate power first.",
                     "Use this guide to plan work, then open the service panel.",
                     "Confirm all requirements against the manufacturer's manual.",
                     "Check the refrigerant charge."):
            with self.subTest(text=text):
                self.assertFalse(document_purpose_statement(text, self._level(text), PURPOSE))

    def test_restricted_sentences_never_qualify(self):
        text = "Provide refrigerant charging guidance for service teams."
        self.assertEqual(self._level(text), "TECHNICIAN_ONLY")
        self.assertFalse(document_purpose_statement(text, "TECHNICIAN_ONLY", PURPOSE))


class VrfSystemNameSafetyTests(unittest.TestCase):
    def test_system_category_name_alone_is_not_technician_only(self):
        for text in ("Generic installation and commissioning reference for a Variable Refrigerant Flow (VRF) outdoor module.",
                     "This variable-refrigerant-flow system has several indoor units."):
            with self.subTest(text=text):
                self.assertNotIn(classify_action_safety(text)[0], ("TECHNICIAN_ONLY", "PROHIBITED"))

    def test_refrigerant_handling_is_still_technician_only(self):
        for text in ("Check the refrigerant charge.",
                     "Recover refrigerant before opening the circuit.",
                     "Variable Refrigerant Flow systems need refrigerant charging by technicians.",
                     "On a Variable Refrigerant Flow unit, top up refrigerant if pressure is low.",
                     "Connect the manifold gauge to the VRF service port."):
            with self.subTest(text=text):
                self.assertEqual(classify_action_safety(text)[0], "TECHNICIAN_ONLY")


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class DocumentPurposeAnswerTests(unittest.TestCase):
    def _assert_purpose(self, asset, question, title, expected):
        r = ask(asset, question)
        self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE", (question, r.message))
        self.assertTrue(any(expected in text for text in r.texts), r.texts)
        self.assertIn((title, PURPOSE), r.cited)

    def test_vrf_maintenance_checklist_purpose(self):
        for question in ("What does the VRF preventive maintenance checklist provide?",
                         "What is the purpose of the VRF preventive maintenance checklist?",
                         "What does the VRF maintenance checklist cover?"):
            with self.subTest(question=question):
                self._assert_purpose("vrf", question, "Preventive Maintenance Checklist",
                                     "documented maintenance routine for a VRF outdoor module")

    def test_vrf_installation_guide_purpose(self):
        for question in ("What does the VRF installation and commissioning guide cover?",
                         "What is the VRF installation guide for?",
                         "What is the purpose of the VRF installation and commissioning guide?"):
            with self.subTest(question=question):
                self._assert_purpose("vrf", question, "Installation and Commissioning Guide",
                                     "Variable Refrigerant Flow (VRF) outdoor module")

    def test_split_and_ductable_purpose_answers(self):
        self._assert_purpose("split", "What does the split AC preventive maintenance checklist provide?",
                             "Preventive Maintenance Checklist",
                             "Use this checklist to plan routine inspection and maintenance")
        self._assert_purpose("duct", "What does the ductable AC installation and commissioning guide cover?",
                             "Installation and Commissioning Guide",
                             "Generic guide for installation and commissioning of a nominal 5-ton")

    def test_restricted_vrf_questions_still_disclose_no_procedure(self):
        r = ask("vrf", "How is the pressure test performed?")
        self.assertEqual(r.texts, [])
        r = ask("vrf", "Can I open the refrigerant circuit to check the charge myself?")
        self.assertNotEqual(r.type, "DIAGNOSTIC_GUIDANCE")
        self.assertFalse(any(k == "customer_procedure" for k in r.kinds))

    def test_purpose_rule_does_not_apply_to_non_purpose_questions(self):
        r = ask("vrf", "What should the maintenance frequency be based on?")
        self.assertTrue(any("Frequency should be based on manufacturer guidance" in t for t in r.texts), r.texts)


if __name__ == "__main__":
    unittest.main()
