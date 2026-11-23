"""Section identity in both ingestion paths (R04, R05). No index is rebuilt here."""
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from core.services.chunking import chunk_markdownish
from core.services.starlly_ingest import split_document_into_sections

MANUAL_DIR = Path(settings.BASE_DIR) / "media" / "knowledge" / "9" / "2026" / "10"


class ChunkerSectionIdentityTests(SimpleTestCase):
    def test_cause_labels_stay_under_their_fault_heading(self):
        chunks = chunk_markdownish("2. SYMPTOM: UNIT DOES NOT START\nThe unit is silent when switched on.\n"
                                   "Possible causes:\n- Fuse blown\n- Loose plug\nInitial checks:\n- Verify supply")
        self.assertEqual([h for h, _ in chunks], ["2. SYMPTOM: UNIT DOES NOT START"])
        self.assertIn("- Fuse blown", chunks[0][1])
        self.assertIn("- Verify supply", chunks[0][1])

    def test_mixed_case_symptom_heading_is_a_heading(self):
        chunks = chunk_markdownish("2. Symptom: Unit does not start\nPossible causes:\n- Fuse blown")
        self.assertEqual(chunks, [("2. Symptom: Unit does not start", "Possible causes:\n- Fuse blown")])

    def test_next_numbered_section_still_splits(self):
        chunks = chunk_markdownish("2. SYMPTOM: A\nPossible causes:\n- a\n\n3. SYMPTOM: B\nPossible causes:\n- b")
        self.assertEqual([h for h, _ in chunks], ["2. SYMPTOM: A", "3. SYMPTOM: B"])


class StarllySectionIdentityTests(SimpleTestCase):
    def test_first_section_keeps_its_own_heading(self):
        sections = split_document_into_sections("1. SYMPTOM: X\nPossible causes:\n- a\n- b\n2. SYMPTOM: Y\n- c\n- d")
        self.assertEqual([s["heading"] for s in sections], ["1. SYMPTOM: X", "2. SYMPTOM: Y"])

    def test_short_section_does_not_swallow_the_next_fault_heading(self):
        sections = split_document_into_sections("INTRO\nline one\nline two\n1. SYMPTOM: X\n- a\n2. SYMPTOM: Y\n- b\n")
        headings = [s["heading"] for s in sections]
        self.assertIn("1. SYMPTOM: X", headings)
        self.assertIn("2. SYMPTOM: Y", headings)
        self.assertNotIn("2. SYMPTOM: Y", next(s for s in sections if s["heading"] == "1. SYMPTOM: X")["text"])

    def test_procedure_steps_are_not_strong_section_headings(self):
        from core.services.starlly_ingest import _STRONG_SECTION
        for step in ("1. Isolate power.", "2. Open the panel.", "3. Inspect the filter."):
            self.assertIsNone(_STRONG_SECTION.match(step), step)
        for heading in ("3. SYMPTOM: POOR COOLING", "2. Symptom: Unit does not start", "SYMPTOM: Ice on coil"):
            self.assertIsNotNone(_STRONG_SECTION.match(heading), heading)

    def test_approved_manuals_keep_their_sections(self):
        files = sorted(MANUAL_DIR.glob("*.txt"))
        if len(files) != 9:
            self.skipTest("approved manual files are not present")
        sections = [split_document_into_sections(p.read_text(encoding="utf-8")) for p in files]
        self.assertEqual(sum(len(s) for s in sections), 69)
        headings = {s["heading"] for doc in sections for s in doc}
        for expected in ("3. SYMPTOM: POOR COOLING", "6. SYMPTOM: NOISE OR VIBRATION",
                         "2. COMMUNICATION OR ADDRESSING FAULT"):
            self.assertIn(expected, headings)
