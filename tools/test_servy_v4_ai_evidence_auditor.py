"""Unit tests for the autonomous AI evidence and label review auditor."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from servy_v4_ai_evidence_auditor import ServyAIEvidenceAuditor, norm_tokens, token_set


class TestServyAIEvidenceAuditor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        
        # Create SQLite DB
        self.db_path = self.root / "test_db.sqlite3"
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE core_tenant (id INTEGER PRIMARY KEY, name TEXT, slug TEXT);
        """)
        cur.execute("INSERT INTO core_tenant VALUES (6, 'Starlly Tester', 'starlly-tester');")
        
        cur.execute("""
            CREATE TABLE core_product (id INTEGER PRIMARY KEY, name TEXT, brand_id INTEGER, category_id INTEGER, tenant_id INTEGER);
        """)
        cur.execute("INSERT INTO core_product VALUES (10, 'Paper Shredder', 1, 10, 6);")
        
        cur.execute("""
            CREATE TABLE core_productcategory (id INTEGER PRIMARY KEY, name TEXT, domain_id INTEGER, tenant_id INTEGER);
        """)
        cur.execute("INSERT INTO core_productcategory VALUES (10, 'Office Equipment', 8, 6);")
        
        cur.execute("""
            CREATE TABLE core_brand (id INTEGER PRIMARY KEY, name TEXT);
        """)
        cur.execute("INSERT INTO core_brand VALUES (1, 'TestBrand');")
        
        cur.execute("""
            CREATE TABLE core_productdomain (id INTEGER PRIMARY KEY, name TEXT);
        """)
        cur.execute("INSERT INTO core_productdomain VALUES (8, 'Office & Utility');")
        
        cur.execute("""
            CREATE TABLE core_knowledgedocument (
                id INTEGER PRIMARY KEY, title TEXT, doc_type TEXT, source_type TEXT,
                version TEXT, checksum_sha256 TEXT, asset_id INTEGER, customer_id INTEGER,
                product_id INTEGER, category_id INTEGER, domain_id INTEGER, tenant_id INTEGER
            );
        """)
        cur.execute("""
            INSERT INTO core_knowledgedocument VALUES (
                135, 'CMC1000 Maintenance Guide', 'maintenance', 'text',
                '1.0', ?, NULL, NULL, 10, 10, 8, 6
            );
        """, ('a' * 64,))
        conn.commit()
        conn.close()
        
        # Create corpus
        self.corpus_path = self.root / "corpus.jsonl"
        self.corpus_path.write_text(json.dumps({
            "chunk_id": "58", "tenant_id": "6", "document_id": "135",
            "passage": "## Routine maintenance\nEmpty the waste container and lubricate blades."
        }) + "\n" + json.dumps({
            "chunk_id": "59", "tenant_id": "6", "document_id": "135",
            "passage": "## Expected result\nSmooth quiet operation."
        }) + "\n", encoding="utf-8")
        
        # Create bundle
        self.bundle_dir = self.root / "bundle"
        self.bundle_dir.mkdir(parents=True)
        (self.bundle_dir / "source_family_review_queue.jsonl").write_text(json.dumps({
            "tenant_id": "6", "document_id": "135", "chunk_count": 2,
            "document_versions": ["1.0"], "source_document_sha256": ["aa"*32],
            "unverified_declared_groups": ["doc-135"]
        }) + "\n", encoding="utf-8")
        
        (self.bundle_dir / "expert_review_queue.jsonl").write_text(json.dumps({
            "split": "train", "query_id": "Q1",
            "query_text": "CMC1000 paper shredder routine maintenance procedure",
            "references": [
                {
                    "role": "positive", "chunk_id": "58", "document_id": "135",
                    "passage": "## Routine maintenance\nEmpty the waste container and lubricate blades.",
                    "decision": "PENDING"
                },
                {
                    "role": "negative", "chunk_id": "59", "document_id": "135",
                    "passage": "## Expected result\nSmooth quiet operation.",
                    "decision": "PENDING"
                }
            ]
        }) + "\n", encoding="utf-8")

    def test_token_helpers(self):
        self.assertEqual(norm_tokens("CMC1000 & Shredder!"), ["cmc1000", "shredder"])
        self.assertIn("cmc1000", token_set("CMC1000 & Shredder!"))

    def test_provenance_audit_verified_evidence(self):
        auditor = ServyAIEvidenceAuditor(self.bundle_dir, self.corpus_path, self.db_path)
        try:
            recs, summary = auditor.audit_document_provenance()
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["evidence_status"], "VERIFIED_SOURCE_EVIDENCE")
            self.assertEqual(recs[0]["proposed_canonical_source_family"], "FAMILY-PAPER_SHREDDER")
            self.assertEqual(recs[0]["confidence"], "HIGH")
            self.assertEqual(summary["total_documents"], 1)
            self.assertEqual(summary["status_counts"]["VERIFIED_SOURCE_EVIDENCE"], 1)
        finally:
            auditor.close()

    def test_relevance_audit_and_export(self):
        auditor = ServyAIEvidenceAuditor(self.bundle_dir, self.corpus_path, self.db_path)
        out_dir = self.root / "ai_evidence_out"
        try:
            summary = auditor.run_full_audit(out_dir)
            self.assertTrue((out_dir / "ai_source_family_recommendations.jsonl").is_file())
            self.assertTrue((out_dir / "ai_label_recommendations.jsonl").is_file())
            self.assertTrue((out_dir / "ai_uncertain_cases.jsonl").is_file())
            self.assertTrue((out_dir / "ai_false_negative_candidates.jsonl").is_file())
            self.assertTrue((out_dir / "ai_source_evidence_manifest.json").is_file())
            self.assertTrue((out_dir / "ai_review_summary.json").is_file())
            self.assertEqual(summary["provenance_audit"]["total_documents"], 1)
            self.assertEqual(summary["relevance_audit"]["total_references_audited"], 2)
        finally:
            auditor.close()


if __name__ == "__main__":
    unittest.main()
