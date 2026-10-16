from django.test import TestCase
from unittest.mock import MagicMock
from core.services.retriever import RetrievedChunk
from core.services.llm import _extractive_answer, INSUFFICIENT_EVIDENCE_MESSAGE


class LLMAnswerRankingTests(TestCase):
    def setUp(self):
        self.mock_asset = MagicMock()
        self.mock_asset.name = "VRF Outdoor Module"

    def test_retriever_rank_overrides_lexical_token_overlap(self):
        """Top-ranked retriever passage must be selected as primary chunk even if a
        lower-ranked passage repeats more generic question words."""
        # Chunk 1: Highly relevant fault code procedure, ranked #1 by retriever
        chunk_1 = RetrievedChunk(
            source_kind="knowledge_document",
            source_id=1,
            document_id=10,
            chunk_id=101,
            title="VRF Alarm Guide",
            heading="E04 Inverter Board Failure",
            text="Step 1 - Inspect inverter board terminals for loose wiring.\nStep 2 - Verify DC bus voltage is within 280-320V.",
            score=0.85,
            doc_type="Manual",
            source_url="",
            confidential=False,
            final_rank=1,
            exact_identifier_match=True,
        )

        # Chunk 2: General introductory passage that happens to repeat customer words multiple times
        chunk_2 = RetrievedChunk(
            source_kind="knowledge_document",
            source_id=2,
            document_id=11,
            chunk_id=102,
            title="General Maintenance",
            heading="General Introduction",
            text="When your equipment has an issue or problem, check the unit carefully. Always inspect your unit.",
            score=0.45,
            doc_type="Manual",
            source_url="",
            confidential=False,
            final_rank=2,
            exact_identifier_match=False,
        )

        question = "What should I do when my unit has problem E04 and issue occurs?"
        retrieved = [chunk_1, chunk_2]

        answer = _extractive_answer(question, self.mock_asset, retrieved)

        # The primary diagnostic summary must correspond to Chunk 1 (E04 Inverter Board Failure), NOT Chunk 2
        summary_section = answer.split("CHECK FIRST")[0]
        self.assertIn("E04 Inverter Board Failure", summary_section)
        self.assertNotIn("General Introduction", summary_section)

        # Step 1 must be from Chunk 1
        self.assertIn("Inspect inverter board terminals", answer)

        # VERIFIED REFERENCES must list Chunk 1 before Chunk 2
        refs_section = answer.split("VERIFIED REFERENCES")[1]
        c1_idx = refs_section.find("VRF Alarm Guide")
        c2_idx = refs_section.find("General Maintenance")
        self.assertTrue(c1_idx != -1 and c2_idx != -1)
        self.assertLess(c1_idx, c2_idx, "Chunk 1 must appear before Chunk 2 in VERIFIED REFERENCES")

    def test_abstention_on_unsupported_identifier(self):
        """If query asks about an identifier missing from retrieved sources, answer must abstain."""
        chunk = RetrievedChunk(
            source_kind="knowledge_document",
            source_id=1,
            document_id=10,
            chunk_id=101,
            title="Split AC Manual",
            heading="Filter Maintenance",
            text="Clean filter mesh every 250 operational hours with warm water.",
            score=0.50,
            doc_type="Manual",
            source_url="",
            confidential=False,
        )
        answer = _extractive_answer("What causes error E-99 on split ac?", self.mock_asset, [chunk])
        self.assertIn("No approved documentation found for identifier E99", answer)

    def test_abstention_on_empty_retrieved(self):
        answer = _extractive_answer("Random question", self.mock_asset, [])
        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_MESSAGE)
