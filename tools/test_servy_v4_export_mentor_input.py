"""Synthetic tests of the private mentor-input ZIP exporter."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from servy_v4_export_mentor_input import pack
from servy_v4_prepare_research_artifacts import sha256


def dump(path, rows):
    path.write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")


class BundleExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)/"private_training_data"
        root.mkdir()
        self.split=root/"split"
        self.split.mkdir()
        self.corpus=root/"corpus.jsonl"
        self.families=root/"families.jsonl"
        self.target=root/"test_mentor_input.zip"
        chunks=[{"chunk_id":str(i),"tenant_id":"1","document_id":str(i),
                 "passage":"Technical step "+str(i),"source_version":"1",
                 "source_checksum_sha256":str(i)*64} for i in range(1,5)]
        dump(self.corpus,chunks)
        train=[{"tenant_id":"1","query_id":"T1","query_text":"technical step 1",
                "positive_chunk_ids":["1"],"negative_chunk_ids":["2"]}]
        dev=[{"tenant_id":"1","query_id":"D1","query_text":"technical step 3",
              "positive_chunk_ids":["3"],"negative_chunk_ids":["4"]}]
        dump(self.split/"train_unapproved_candidate.jsonl",train)
        dump(self.split/"dev_unapproved_candidate.jsonl",dev)
        dump(self.split/"quarantined_negatives.jsonl",[
            {"split":"train","query_id":"T1","negative_chunk_id":"4","review_status":"PENDING"}
        ])
        train[0]["quarantined_negative_chunk_ids"]=["4"]
        dump(self.split/"train_unapproved_candidate.jsonl",train)
        dump(self.families,[{"tenant_id":"1","document_id":str(i),
                             "proposed_canonical_source_family":"family"+str(i),
                             "evidence_status":"REGISTRY_METADATA_MATCH"} for i in range(1,5)])
        (self.split/"resplit_report.json").write_text(json.dumps({
            "status":"PROVISIONAL_RESPLIT_ONLY", "total_queries":2,
            "input_sha256":{"corpus":sha256(self.corpus),
                            "recommendations":sha256(self.families)},
            "output_sha256":{
                name:sha256(self.split/name)
                for name in ("train_unapproved_candidate.jsonl",
                             "dev_unapproved_candidate.jsonl",
                             "quarantined_negatives.jsonl")}
        }),encoding="utf-8")

    def test_export_integrity_verified(self):
        meta=pack(self.split,self.corpus,self.families,self.target)
        self.assertTrue(self.target.is_file())
        self.assertEqual(meta["train_queries"],1)
        self.assertEqual(meta["candidate_method"],"BM25")
        with zipfile.ZipFile(self.target) as z:
            self.assertEqual(z.testzip(),None)
            self.assertIn("manifest.json",z.namelist())
            self.assertIn("split/resplit_report.json",z.namelist())
            self.assertIn("candidates.jsonl",z.namelist())
            self.assertEqual(json.loads(z.read("manifest.json"))["expert_approval_claimed"],False)

    def test_no_overwrite_and_no_public_archive(self):
        pack(self.split,self.corpus,self.families,self.target)
        with self.assertRaises(FileExistsError):
            pack(self.split,self.corpus,self.families,self.target)
        with self.assertRaisesRegex(ValueError,"private_training_data"):
            pack(self.split,self.corpus,self.families,Path(self.tmp.name)/"public.zip")

    def test_invalid_frozen_source_manifest_blocks_before_zip(self):
        p=self.split/"resplit_report.json"
        report=json.loads(p.read_text())
        report["input_sha256"]["corpus"]="0"*64
        p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,"CORPUS_HASH_MISMATCH"):
            pack(self.split,self.corpus,self.families,self.target)
        self.assertFalse(self.target.exists())

if __name__=="__main__":
    unittest.main()
