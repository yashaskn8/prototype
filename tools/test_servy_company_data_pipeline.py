"""Unit tests for tools/servy_company_data_pipeline.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from servy_company_data_pipeline import (
    CompanyDataPipelineError,
    canonical_source_family,
    minimize_confidential_text,
    segment_document_sections,
    validate_and_quarantine_documents,
    validate_training_example_schema,
    verify_family_disjoint_splits,
)


class CompanyDataPipelineTest(unittest.TestCase):
    def test_document_validation_and_quarantine(self):
        docs = [
            {
                "document_id": 10,
                "tenant_id": 9,
                "title": "VoltCore 3kVA Inverter Service Manual v2.0",
                "product": "VoltCore 3kVA Inverter",
                "text": "## Operating Specifications\nContact tech@skydream.example or +1-555-234-5678. Voltage: 5V / 12V / 19V.",
                "is_rag_enabled": True,
                "is_confidential": False,
            },
            # Duplicate of doc 10
            {
                "document_id": 11,
                "tenant_id": 9,
                "title": "VoltCore 3kVA Copy",
                "product": "VoltCore 3kVA Inverter",
                "text": "## Operating Specifications\nContact tech@skydream.example or +1-555-234-5678. Voltage: 5V / 12V / 19V.",
                "is_rag_enabled": True,
                "is_confidential": False,
            },
            # Revoked doc
            {
                "document_id": 12,
                "tenant_id": 9,
                "title": "Revoked Manual",
                "product": "Old Unit",
                "text": "Obsolete procedure",
                "revoked_at": "2026-10-01T00:00:00Z",
            },
            # Confidential doc
            {
                "document_id": 13,
                "tenant_id": 9,
                "title": "Internal Board Schematics",
                "product": "VoltCore 3kVA",
                "text": "Internal schematic",
                "is_confidential": True,
            },
        ]
        res = validate_and_quarantine_documents(docs, allow_confidential=False)
        self.assertEqual(res["accepted_count"], 1)
        self.assertEqual(res["quarantined_count"], 3)
        acc = res["accepted_documents"][0]
        self.assertEqual(acc["document_id"], 10)
        self.assertIn("[REDACTED_EMAIL]", acc["text"])
        self.assertIn("[REDACTED_PHONE]", acc["text"])
        self.assertTrue(acc["source_family"].startswith("fam_voltcore_3kva"))

    def test_section_aware_segmentation(self):
        doc = {
            "document_id": 10,
            "tenant_id": 9,
            "title": "VoltCore 3kVA Manual",
            "equipment_family": "VoltCore 3kVA",
            "model_number": "VC-3000",
            "source_family": "fam_voltcore_3kva",
            "text": "## Operating Specifications\nOperating V specification 5/12/19.\n## Stop and escalate if\nStop immediately if smoke appears.",
        }
        chunks = segment_document_sections(doc, start_chunk_id=100)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], 100)
        self.assertEqual(chunks[0]["heading"], "Operating Specifications")
        self.assertEqual(chunks[1]["chunk_id"], 101)
        self.assertEqual(chunks[1]["heading"], "Stop and escalate if")

    def test_family_disjoint_splits_and_historical_leakage_rejection(self):
        examples = [
            {
                "query_id": "Q-TRAIN-1",
                "tenant_id": 9,
                "equipment_family": "VoltCore",
                "model_number": "VC-3000",
                "query_type": "voltage_electrical_spec",
                "query_text": "What is the 5V / 12V / 19V specification for VoltCore?",
                "positive_chunk_ids": [100],
                "positive_doc_ids": [10],
                "hard_negative_chunk_ids": [101],
                "source_family": "fam_voltcore",
                "expected_abstain": False,
                "review_status": "approved_by_sme",
                "reviewer": "sme_lead_1",
                "split": "train",
            },
            {
                "query_id": "Q-DEV-1",
                "tenant_id": 9,
                "equipment_family": "CryoMilk",
                "model_number": "CM-500",
                "query_type": "preventive_maintenance_schedule",
                "query_text": "What are the preventive maintenance intervals for CryoMilk?",
                "positive_chunk_ids": [200],
                "positive_doc_ids": [20],
                "hard_negative_chunk_ids": [201],
                "source_family": "fam_cryomilk",
                "expected_abstain": False,
                "review_status": "approved_by_sme",
                "reviewer": "sme_lead_1",
                "split": "dev",
            },
            {
                "query_id": "Q-TEST-1",
                "tenant_id": 9,
                "equipment_family": "NewChiller",
                "model_number": "NC-900",
                "query_type": "safety_escalation",
                "query_text": "When should a technician stop and escalate on NewChiller?",
                "positive_chunk_ids": [300],
                "positive_doc_ids": [30],
                "hard_negative_chunk_ids": [301],
                "source_family": "fam_newchiller_unseen",
                "expected_abstain": False,
                "review_status": "approved_by_sme",
                "reviewer": "sme_lead_2",
                "split": "independent_test",
            },
        ]
        summary = verify_family_disjoint_splits(
            examples,
            historical_families={"fam_voltcore", "fam_cryomilk"},
            historical_chunk_ids={100, 101, 200, 201},
        )
        self.assertEqual(summary["status"], "SPLITS_VERIFIED_FAMILY_DISJOINT")
        self.assertTrue(summary["all_sme_approved"])
        self.assertFalse(summary["production_promotion_allowed"])

        # Now test rejection when independent_test leaks a historical family
        with self.assertRaises(CompanyDataPipelineError):
            verify_family_disjoint_splits(
                examples,
                historical_families={"fam_voltcore", "fam_cryomilk", "fam_newchiller_unseen"},
            )


if __name__ == "__main__":
    unittest.main()
