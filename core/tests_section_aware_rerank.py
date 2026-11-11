from types import SimpleNamespace

from django.conf import settings
from django.test import SimpleTestCase

from core.ml.reranker import build_contextual_candidate_text
from core.ml.section_aware_rerank import (
    classify_section_intents,
    experimental_evidence_section_rerank,
)


def make_candidate(chunk_id, title, heading, doc_type, text, score=1.0):
    return SimpleNamespace(
        source_kind="knowledge_document", source_id=chunk_id,
        document_id=chunk_id + 100, chunk_id=chunk_id,
        title=title, heading=heading, doc_type=doc_type, text=text,
        score=score, fusion_score=score, dense_score=score,
        sparse_score=score, final_rank=None,
    )


class SectionIntentClassificationTests(SimpleTestCase):
    def test_preinstallation_is_distinct_from_installation(self):
        self.assertEqual(
            classify_section_intents("What must be ready before installation?"),
            ("pre_installation",),
        )

    def test_multi_intent_commissioning_measurements_and_records(self):
        self.assertEqual(
            classify_section_intents(
                "During commissioning, which voltage measurements should be recorded?"
            ),
            ("commissioning", "measurements", "record_keeping"),
        )

    def test_safety_and_maintenance_are_detected(self):
        found = classify_section_intents(
            "What safety isolation is required during routine maintenance?"
        )
        self.assertIn("safety", found)
        self.assertIn("maintenance", found)

    def test_unknown_query_has_no_forced_section_label(self):
        self.assertEqual(classify_section_intents("Can you explain this?"), ())


class SectionAwareRankingTests(SimpleTestCase):
    def test_default_feature_flag_is_disabled(self):
        self.assertFalse(settings.SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK)

    def test_maintenance_candidate_can_out_rank_wrong_installation_section(self):
        installation = make_candidate(
            1, "Service Manual", "Installation procedure",
            "Installation", "Install the unit and connect the piping."
        )
        maintenance = make_candidate(
            2, "Service Manual", "Routine maintenance interval",
            "Maintenance", "Inspect and clean the filter at the stated interval."
        )
        ranked = experimental_evidence_section_rerank(
            [installation, maintenance],
            "What is the routine maintenance interval?",
            top_k=2,
        )
        self.assertEqual(ranked[0].chunk_id, 2)
        self.assertEqual(ranked[0].section_rerank_reason, "section_match:maintenance")
        self.assertIn("maintenance", ranked[0].section_candidate_intents)

    def test_unknown_query_does_not_invent_section_preference(self):
        first = make_candidate(
            1, "Manual", "General operation", "General",
            "The system contains a controller and a display."
        )
        second = make_candidate(
            2, "Manual", "System overview", "General",
            "The system contains a controller and a display."
        )
        ranked = experimental_evidence_section_rerank(
            [first, second], "Can you explain this?", top_k=2
        )
        self.assertEqual([item.chunk_id for item in ranked], [1, 2])
        self.assertEqual(ranked[0].section_rerank_adjustment, 0.0)

    def test_authorized_candidate_set_is_not_filtered_and_tail_is_preserved(self):
        items = [
            make_candidate(i, "Manual", "Routine maintenance", "Maintenance", "Inspect filter.")
            for i in range(1, 4)
        ]
        before_ids = {id(item) for item in items}
        ranked = experimental_evidence_section_rerank(
            items, "What is the maintenance interval?", top_k=2
        )
        self.assertEqual(len(ranked), 3)
        self.assertEqual({id(item) for item in ranked}, before_ids)
        self.assertIs(ranked[-1], items[-1])

    def test_contextual_candidate_text_contains_scoped_metadata_and_original_passage(self):
        candidate = make_candidate(
            7, "VRF Service Manual", "Pressure measurement",
            "Measurements", "Measure discharge pressure at the service port."
        )
        rendered = build_contextual_candidate_text(
            candidate, asset_context="VRF Outdoor Module | VRF-60"
        )
        self.assertIn("Asset: VRF Outdoor Module | VRF-60", rendered)
        self.assertIn("Document: VRF Service Manual", rendered)
        self.assertIn("Section: Pressure measurement", rendered)
        self.assertIn("Document type: Measurements", rendered)
        self.assertIn("Measure discharge pressure", rendered)
