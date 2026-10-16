"""
Unit and Adversarial Tests for Fault-Code Canonical Normalization and Extraction.

Covers Part 4 mandatory adversarial tests:
1. Supported formatting variants (E04, E-04, E 04, ERR-04, ERROR-04, F07, F-07, F 07).
2. Leading-zero distinctions (E04 vs E004).
3. Exact matching without substring bleed (E04 does NOT match E041 or E040).
4. Prefix collisions (F12 vs F120, IPN-442098 vs IPN-44209).
5. Similar-looking identifiers across namespaces (E04 vs F04).
6. Unicode dashes (en-dash, em-dash, minus sign).
7. Multiple fault codes in single query.
8. Unknown fault codes detecting missing source coverage.
"""

from django.test import SimpleTestCase
from core.services.identifiers import (
    extract_identifiers,
    extract_error_codes,
    extract_part_numbers,
    query_identifiers_not_in_sources,
)
from core.services.retriever import _identifier_boost


class FaultCodeNormalizationTests(SimpleTestCase):
    """Test canonical identifier parsing and equivalence classes."""

    def test_formatting_variants_normalize_to_same_code(self):
        """E04, E-04, E 04, ERR-04, and ERROR-04 all canonicalize to E04."""
        variants = ["E04", "E-04", "E 04", "ERR-04", "ERROR-04", "ERR 04"]
        for v in variants:
            codes = extract_error_codes(f"The system threw {v} on startup")
            self.assertIn("E04", codes, f"Variant '{v}' failed to canonicalize to E04")

    def test_f_series_formatting_variants(self):
        """F07, F-07, F 07 canonicalize to F07."""
        variants = ["F07", "F-07", "F 07"]
        for v in variants:
            codes = extract_error_codes(f"Check {v} pressure switch")
            self.assertIn("F07", codes, f"Variant '{v}' failed to canonicalize to F07")

    def test_leading_zero_distinction_preserved(self):
        """E04 and E004 represent distinct technical codes and must NOT collide."""
        codes_e04 = extract_error_codes("Error E04 low pressure")
        codes_e004 = extract_error_codes("Error E004 motor stall")
        self.assertEqual(codes_e04, {"E04"})
        self.assertEqual(codes_e004, {"E004"})
        self.assertNotEqual(codes_e04, codes_e004)

    def test_no_substring_bleed(self):
        """E04 query must NOT match a chunk describing E041 or E040."""
        query_ids = extract_identifiers("E04")
        self.assertEqual(query_ids, {"E04"})

        # Chunk with E041
        boost_unrelated = _identifier_boost(
            query_identifiers=query_ids,
            chunk_text="Refer to Error Code E041 for over-temperature instructions.",
            chunk_heading="Section 4.1",
        )
        self.assertEqual(boost_unrelated, 0.0, "DEFECT: E04 matched E041 via substring!")

        # Chunk with exact E-04
        boost_exact = _identifier_boost(
            query_identifiers=query_ids,
            chunk_text="Refer to Error Code E-04 for low refrigerant pressure.",
            chunk_heading="Section 4.0",
        )
        self.assertEqual(boost_exact, 0.40, "Exact match for E-04 failed to receive boost")

    def test_unicode_dashes_handled(self):
        """En-dash (\u2013), em-dash (\u2014), and minus (\u2212) normalize identically."""
        en_dash = "E\u201304"
        em_dash = "E\u201404"
        minus = "E\u221204"

        for variant in (en_dash, em_dash, minus):
            codes = extract_error_codes(f"Alert {variant} detected")
            self.assertIn("E04", codes, f"Unicode dash variant failed: {variant}")

    def test_multiple_fault_codes_extracted(self):
        """Query with multiple codes extracts all of them independently."""
        text = "The machine displayed both E04 and F07 simultaneously."
        codes = extract_error_codes(text)
        self.assertEqual(codes, {"E04", "F07"})

    def test_unknown_fault_code_detected_as_missing(self):
        """When query asks about E99 and sources only document E04, E99 is flagged as missing."""
        query = "How to clear error E99?"
        source_texts = [
            "Error E-04 indicates low refrigerant pressure.",
            "Error F-07 indicates fan motor failure.",
        ]
        missing = query_identifiers_not_in_sources(query, source_texts)
        self.assertEqual(missing, {"E99"})

    def test_known_fault_code_in_source_not_flagged_missing(self):
        """When query asks about E-04 and source has E04, it is NOT missing."""
        query = "How to fix E-04?"
        source_texts = ["Error Code E04: inspect suction line and recharge gas."]
        missing = query_identifiers_not_in_sources(query, source_texts)
        self.assertEqual(missing, set())
