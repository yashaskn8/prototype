"""Chunk headings for numbered sections and source checksums on re-index."""
import hashlib
from types import SimpleNamespace

from django.test import SimpleTestCase

from core.services.chunking import chunk_markdownish
from core.services.indexing import _source_checksum


class NumberedHeadingChunkingTests(SimpleTestCase):
    def test_numbered_upper_case_titles_become_chunk_headings(self):
        text = ("5. SYMPTOM: OUTDOOR MODULE NOT WORKING\nPossible causes:\n- Power is interrupted.\n\n"
                "6. POWER SUPPLY INTERRUPTION\nA tripped breaker interrupts the power supply.")
        chunks = chunk_markdownish(text)
        self.assertEqual([h for h, _ in chunks],
                         ["5. SYMPTOM: OUTDOOR MODULE NOT WORKING", "6. POWER SUPPLY INTERRUPTION"])
        # The cause-list label stays in its section's body so the list is still recognised.
        self.assertIn("Possible causes:", chunks[0][1])
        self.assertIn("tripped breaker", chunks[1][1])

    def test_ordinary_numbered_sentences_are_not_headings(self):
        chunks = chunk_markdownish("Steps:\n1. Check the filter is clean.\n2. Note any alarm code.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "Steps")

    def test_markdown_and_colon_headings_unchanged(self):
        chunks = chunk_markdownish("## Overview\nText one here.\nMaintenance:\nText two here.")
        self.assertEqual([h for h, _ in chunks], ["Overview", "Maintenance"])


class _File:
    def __init__(self, data=None):
        self.data = data

    def __bool__(self):
        return True

    def open(self, mode):
        if self.data is None:
            raise OSError("missing")
        import io
        return io.BytesIO(self.data)


class SourceChecksumTests(SimpleTestCase):
    def test_file_document_keeps_file_bytes_checksum(self):
        doc = SimpleNamespace(source_type="file", file=_File(b"%PDF original bytes"), checksum_sha256="")
        self.assertEqual(_source_checksum(doc, "extracted text"),
                         hashlib.sha256(b"%PDF original bytes").hexdigest())

    def test_unreadable_file_keeps_recorded_checksum(self):
        doc = SimpleNamespace(source_type="file", file=_File(None), checksum_sha256="ab" * 32)
        self.assertEqual(_source_checksum(doc, "extracted text"), "ab" * 32)

    def test_text_document_hashes_its_text(self):
        doc = SimpleNamespace(source_type="text", file=None, checksum_sha256="")
        self.assertEqual(_source_checksum(doc, "body"), hashlib.sha256(b"body").hexdigest())
