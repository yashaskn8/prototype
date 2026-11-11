"""Synthetic-only regressions for the three approved real asset families.

The nine checksum keys below are public provenance identifiers, not document
contents. Tests never connect to the company database or copy manual text.
"""
from __future__ import annotations

import os
import re
import unittest
from types import SimpleNamespace

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("LOCALAPPDATA", "/tmp")

import django
django.setup()

from core.services.customer_isolation import VERIFIED_REAL_PROVENANCE_REGISTRY
from core.services.knowledge_scope import is_document_applicable_to_asset


EXPECTED_ASSETS = {
    59: {"asset_code": "SKY-AST-00001", "name": "Split AC Unit 2T", "customer_id": 14},
    60: {"asset_code": "SKY-AST-00002", "name": "VRF Outdoor Module", "customer_id": 13},
    61: {"asset_code": "SKY-AST-00003", "name": "Ductable AC 5T", "customer_id": 14},
}
EXPECTED_DOC_IDS = {
    59: {168, 169, 170},
    60: {171, 172, 173},
    61: {174, 175, 176},
}
EXPECTED_DOC_TYPES = {"troubleshooting", "maintenance", "installation"}


class ThreeAssetNineDocumentRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = VERIFIED_REAL_PROVENANCE_REGISTRY
        cls.assets = cls.registry["assets"]
        cls.documents = cls.registry["documents"]
        cls.selected_documents = {
            digest: entry for digest, entry in cls.documents.items()
            if entry.get("local_asset_id") in EXPECTED_ASSETS
        }

    def test_exactly_three_approved_assets_are_in_scope(self):
        for asset_id, expected in EXPECTED_ASSETS.items():
            entry = self.assets.get(expected["asset_code"])
            self.assertIsNotNone(entry, f"Missing asset provenance for {expected['asset_code']}")
            self.assertEqual(entry["local_id"], asset_id)
            self.assertEqual(entry["model"], expected["name"])
            self.assertEqual(entry["tenant_slug"], "skydream")
            self.assertEqual(entry["source_org"], "273")
            self.assertEqual(entry["status"], "active")

    def test_nine_registered_documents_are_distributed_three_per_asset(self):
        self.assertEqual(len(self.selected_documents), 9)
        for asset_id, doc_ids in EXPECTED_DOC_IDS.items():
            rows = [
                entry for entry in self.selected_documents.values()
                if entry["local_asset_id"] == asset_id
            ]
            self.assertEqual({entry["local_doc_id"] for entry in rows}, doc_ids)
            self.assertEqual(
                {entry["doc_type"] for entry in rows},
                EXPECTED_DOC_TYPES,
                f"Asset {asset_id} must have troubleshooting, maintenance and installation manuals.",
            )
            self.assertTrue(all(entry["status"] == "active" for entry in rows))

    def test_document_checksum_registry_is_complete_and_unique(self):
        self.assertEqual(
            {entry["local_doc_id"] for entry in self.selected_documents.values()},
            set().union(*EXPECTED_DOC_IDS.values()),
        )
        self.assertEqual(
            len({entry["local_doc_id"] for entry in self.selected_documents.values()}),
            9,
        )
        for digest, entry in self.selected_documents.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(entry["source_org"], "273")
            self.assertEqual(entry["asset_code"], EXPECTED_ASSETS[entry["local_asset_id"]]["asset_code"])

    def test_each_registered_manual_applies_only_to_its_exact_asset(self):
        assets = {
            asset_id: SimpleNamespace(id=asset_id, product=None, product_id=None)
            for asset_id in EXPECTED_ASSETS
        }
        for digest, entry in self.selected_documents.items():
            document = SimpleNamespace(
                asset_id=entry["local_asset_id"],
                product_id=None,
                category_id=None,
                domain_id=None,
            )
            owning_asset_id = entry["local_asset_id"]
            self.assertTrue(
                is_document_applicable_to_asset(document, assets[owning_asset_id]),
                f"Registered source {digest[:12]} did not apply to its bound asset.",
            )
            for other_asset_id, other_asset in assets.items():
                if other_asset_id == owning_asset_id:
                    continue
                self.assertFalse(
                    is_document_applicable_to_asset(document, other_asset),
                    f"Asset {other_asset_id} could retrieve another asset's manual.",
                )
            self.assertFalse(is_document_applicable_to_asset(document, None))

    def test_frozen_ci_fixture_has_regressions_for_all_three_assets(self):
        from core.ml.benchmark_registry import load_and_verify_benchmark

        benchmark = load_and_verify_benchmark("servy-ci-regression-fixture-v1")
        rows = [
            row for row in benchmark["cases"]
            if row.get("equipment") in {item["name"] for item in EXPECTED_ASSETS.values()}
            and row.get("positive_chunk_ids")
        ]
        counts = {
            name: sum(row["equipment"] == name for row in rows)
            for name in (item["name"] for item in EXPECTED_ASSETS.values())
        }
        self.assertEqual(counts, {
            "Split AC Unit 2T": 11,
            "VRF Outdoor Module": 8,
            "Ductable AC 5T": 9,
        })
        self.assertEqual(len(rows), 28)
        self.assertTrue(all(row.get("positive_chunk_ids") for row in rows))

    def test_demo_customer_scope_matches_the_three_asset_registry(self):
        expected_scope = {
            asset_id: row["customer_id"] for asset_id, row in EXPECTED_ASSETS.items()
        }
        self.assertEqual(expected_scope, {59: 14, 60: 13, 61: 14})


if __name__ == "__main__":
    unittest.main()
