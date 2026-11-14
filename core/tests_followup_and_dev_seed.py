"""Regressions for two official-path answer gaps found on servy_dev.

1. A complete new question asked after a symptom turn was rewritten with the
   earlier symptoms, so the previous topic was answered instead of declining.
2. The development seed chunked the approved manuals with a different splitter,
   losing the numbered section headings the answer stage relies on.
"""
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase, override_settings

from core.models import Customer, CustomerConversation, CustomerMessage, Tenant
from core.services.conversation_context import extract_conversation_context, is_self_contained_question
from core.services.starlly_ingest import split_document_into_sections

MANUAL_DIR = Path(settings.BASE_DIR) / "media" / "knowledge" / "9" / "2026" / "10"


class SelfContainedQuestionTests(SimpleTestCase):
    def test_new_questions_are_self_contained(self):
        for text in ("What is the warranty period for the compressor in years?",
                     "What refrigerant charge in kilograms does this unit need?",
                     "How many years will the blower motor last?",
                     "What causes a communication or addressing fault?"):
            with self.subTest(text=text):
                self.assertTrue(is_self_contained_question(text))

    def test_follow_ups_and_observations_are_not(self):
        for text in ("What can cause this?", "What next?", "Is it the filter?", "What about the outdoor unit?",
                     "What else should I check?", "Why?", "Could the dirty filter cause this?",
                     "Is the issue still there after cleaning?", "What should I do now?",
                     "The filter looks dusty.", "Some rooms get less air than others."):
            with self.subTest(text=text):
                self.assertFalse(is_self_contained_question(text))


class FollowUpDetectionTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Synthetic", slug="synthetic-followup")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Synthetic Customer")
        self.conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, status="ACTIVE")
        CustomerMessage.objects.create(tenant=self.tenant, conversation=self.conv, sender_role="customer",
                                       message_text="What can cause poor cooling?")
        CustomerMessage.objects.create(tenant=self.tenant, conversation=self.conv, sender_role="assistant",
                                       message_text="Possible documented cause: Dirty filter or coil.",
                                       structured_contract={"response_type": "DIAGNOSTIC_GUIDANCE",
                                                            "clarification_question": "Is the indoor fan running?"})

    def _ctx(self, text):
        return extract_conversation_context(self.conv, text)

    def test_new_question_after_a_symptom_is_not_a_follow_up(self):
        self.assertFalse(self._ctx("What is the warranty period for the compressor in years?").is_followup)
        self.assertFalse(self._ctx("How many years will the blower motor last?").is_followup)

    def test_real_follow_ups_still_use_the_conversation(self):
        for text in ("What next?", "Is it the filter?", "Yes, the fan is running.", "What else should I check?"):
            with self.subTest(text=text):
                self.assertTrue(self._ctx(text).is_followup)


class ApprovedManualChunkingTests(SimpleTestCase):
    def test_section_chunker_reproduces_the_approved_sections(self):
        files = sorted(MANUAL_DIR.glob("*.txt"))
        if len(files) != 9:
            self.skipTest("approved manual files are not present")
        sections = [split_document_into_sections(p.read_text(encoding="utf-8")) for p in files]
        self.assertEqual(sum(len(s) for s in sections), 69)
        headings = {s["heading"] for doc in sections for s in doc}
        self.assertIn("3. SYMPTOM: POOR COOLING", headings)
        self.assertIn("2. COMMUNICATION OR ADDRESSING FAULT", headings)
        self.assertIn("2. SYMPTOM: WEAK AIRFLOW", headings)

    def test_seed_uses_the_approved_section_chunker(self):
        source = (Path(settings.BASE_DIR) / "core" / "management" / "commands" / "seed_servy_dev.py").read_text(encoding="utf-8")
        self.assertIn("split_document_into_sections", source)
        self.assertNotIn("index_document(", source)


class DevSeedGuardTests(TestCase):
    def test_seed_refuses_outside_servy_dev(self):
        with self.assertRaisesRegex(CommandError, "Refusing to seed"):
            call_command("seed_servy_dev")
        with self.assertRaisesRegex(CommandError, "Refusing to seed"):
            call_command("seed_servy_dev", "--rechunk-manuals")
