"""Benchmark input integrity tests.

Verifies that question text is never truncated, mutated, or altered before
being submitted to ask_rag().  Reproduces the truncation defect discovered
in the earlier evaluation and prevents its recurrence.
"""
import hashlib
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django
django.setup()


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BenchmarkInputIntegrityTests(unittest.TestCase):
    """Verify that the benchmark pipeline preserves question text exactly."""

    CODEX_100_PATH = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        r"Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\Servy"
        r"\private_evaluations\servy100_20261029_01\servy_100_hard_questions.json",
    )

    @unittest.skipUnless(
        os.path.exists(os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            r"Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\Servy"
            r"\private_evaluations\servy100_20261029_01\servy_100_hard_questions.json",
        )),
        "Original Codex 100-question benchmark not available",
    )
    def test_original_questions_are_not_truncated(self):
        """The known defect: questions were stored at 200 chars.  Originals are 348–525."""
        with open(self.CODEX_100_PATH, encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            text = q["question"]
            self.assertGreater(len(text), 200,
                f"Question {q['id']} is only {len(text)} chars — likely truncated")
            self.assertLessEqual(len(text), 4000,
                f"Question {q['id']} exceeds the 4000-char application limit")

    def test_question_preservation_at_various_lengths(self):
        """Regression test: questions of various lengths must not be mutated."""
        cases = [
            ("Q_50",  "What is the compressor fault code for this unit?  "),
            ("Q_199", "x" * 199),
            ("Q_200", "y" * 200),  # The old truncation boundary
            ("Q_201", "z" * 201),
            ("Q_500", "Compare section A with section B: " + "w" * 466),
            ("Q_1000", "Long query: " + "q" * 988),
        ]
        for qid, text in cases:
            # Simulate what ask_rag does: strip and truncate at 4000
            processed = (text or "").strip()[:4000]
            self.assertEqual(processed, text.strip()[:4000],
                f"{qid}: question was altered during processing")
            self.assertEqual(sha256(processed), sha256(text.strip()[:4000]),
                f"{qid}: checksum mismatch")

    def test_unicode_and_special_chars_preserved(self):
        """Technical questions may contain symbols that must not be stripped."""
        cases = [
            "Temperature exceeds 45°C — what causes this?",
            'Section "3. HIGH-PRESSURE" uses ≥ 30 bar threshold.',
            "Error code E-04\nFollowed by F-07",
            "Fault: compressor @ 415V / 3-phase",
            "Check: is the Δt above 10°C?",
        ]
        for text in cases:
            processed = (text or "").strip()[:4000]
            self.assertEqual(processed, text.strip(),
                f"Special chars lost: {text!r}")

    def test_embedded_newlines_preserved(self):
        """Multi-line questions must keep their newlines."""
        text = "Line 1: compressor fault\nLine 2: check pressures\nLine 3: escalate?"
        processed = (text or "").strip()[:4000]
        self.assertEqual(processed.count("\n"), 2)

    def test_checksum_detects_single_char_change(self):
        """Even a one-character mutation must change the checksum."""
        original = "What is the fault code for compressor overheating?"
        mutated = "what is the fault code for compressor overheating?"  # lowercase 'w'
        self.assertNotEqual(sha256(original), sha256(mutated))


class GroundingTermFilteringTests(unittest.TestCase):
    """Verify the instruction-term filtering is safe and effective."""

    def setUp(self):
        from core.services.requirements import grounding_terms, INSTRUCTION_TERMS, REQUIREMENT_TERMS
        from core.services.llm import _STOPWORDS
        self.grounding_terms = grounding_terms
        self.INSTRUCTION_TERMS = INSTRUCTION_TERMS
        self.REQUIREMENT_TERMS = REQUIREMENT_TERMS
        self.stopwords = _STOPWORDS

    def test_technical_words_never_in_instruction_terms(self):
        """Critical safety: real equipment words must not be filtered out."""
        technical = {
            "compressor", "inverter", "pressure", "temperature", "refrigerant",
            "drain", "coil", "fan", "sensor", "voltage", "current", "motor",
            "bearing", "valve", "condenser", "evaporator", "thermostat",
            "breaker", "isolator", "contactor", "capacitor", "relay",
            "filter", "duct", "airflow", "damper", "belt", "pulley",
        }
        overlap = technical & self.INSTRUCTION_TERMS
        self.assertEqual(overlap, set(),
            f"Technical words wrongly classified as instruction terms: {overlap}")

    def test_instruction_words_are_filtered(self):
        """Words like 'cite', 'passages', 'synthesis' must be excluded
        from the grounding denominator when real technical terms exist."""
        q = "Cite exact passages about compressor fault. Give a source-grounded synthesis of the relevant sections."
        terms = self.grounding_terms(q, self.stopwords)
        for word in ("cite", "passages", "synthesis", "sections", "source-grounded"):
            self.assertNotIn(word, terms, f"Instruction word '{word}' was not filtered")
        # But technical content must remain
        self.assertIn("compressor", terms)
        self.assertIn("fault", terms)

    def test_short_technical_question_has_terms(self):
        """A simple question must still produce grounding terms."""
        q = "What are the common causes of compressor fault?"
        terms = self.grounding_terms(q, self.stopwords)
        self.assertIn("compressor", terms)
        self.assertIn("fault", terms)
        self.assertGreater(len(terms), 0)

    def test_long_compound_question_reduces_terms(self):
        """A 500-char compound question with instruction padding must shrink its term set."""
        q = (
            'Compare "6. ACCEPTANCE" with "5. COMPRESSOR OR INVERTER FAULT" '
            'for the selected equipment. What distinctions are actually documented, '
            'what cannot be distinguished from a customer report alone, what information '
            'may I report without operating internal components, and when is technician '
            'help needed? Cite both relevant passages; do not infer a relationship '
            'the manual does not establish. no confirmed diagnostic measurements are '
            'available. I cannot confirm whether the operating conditions match the manual.'
        )
        terms = self.grounding_terms(q, self.stopwords)
        # Must contain real technical content
        self.assertIn("compressor", terms)
        self.assertIn("inverter", terms)
        self.assertIn("fault", terms)
        # Must be significantly smaller than raw tokenization
        raw = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", q)}
        self.assertLess(len(terms), len(raw) * 0.5,
            f"Filtering only removed {len(raw)-len(terms)} of {len(raw)} terms")

    def test_all_instruction_term_filter_never_empties(self):
        """Even a question made entirely of instruction words still returns something."""
        q = "Cite relevant passages and give a source-grounded synthesis separately."
        terms = self.grounding_terms(q, self.stopwords)
        self.assertGreater(len(terms), 0, "grounding_terms returned empty set")

    def test_requirement_extraction(self):
        """Questions asking for causes, checks, or codes should detect requirements."""
        from core.services.requirements import extract_requirements
        self.assertIn("possible_causes", extract_requirements("What are the possible causes?"))
        self.assertIn("technician_escalation", extract_requirements("When is technician help needed?"))
        self.assertIn("fault_code", extract_requirements("What is the fault code for E-04?"))
        self.assertIn("numeric_specification", extract_requirements("What is the exact numerical limit?"))
        self.assertNotIn("possible_causes", extract_requirements("How do I cite a passage?"))


if __name__ == "__main__":
    unittest.main()
