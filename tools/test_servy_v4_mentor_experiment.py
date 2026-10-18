"""Synthetic, no-GPU tests for exploratory Servy reranker data admission."""
import json
import tempfile
import unittest
from pathlib import Path

from servy_v4_mentor_experiment import (
    ExplorationBlocked, provisional_pairs, audited_inputs,
)
from servy_v4_prepare_research_artifacts import sha256


def dump(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


class MentorExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "private_training_data"
        root.mkdir()
        self.split = root / "split"
        self.split.mkdir()
        self.corpus = root / "corpus.jsonl"
        self.families = root / "families.jsonl"
        self.candidates = root / "candidates.jsonl"
        corpus = [
            {"chunk_id": str(i), "tenant_id": "6", "document_id": str(i),
             "passage": "Procedure version " + str(i),
             "source_checksum_sha256": str(i)*64,
             "source_version": "1.0"}
            for i in range(1, 5)
        ]
        train = [{"tenant_id":"6","query_id":"q1","query_text":"first procedure",
                  "positive_chunk_ids":["1"],"negative_chunk_ids":["2"]}]
        dev = [{"tenant_id":"6","query_id":"q2","query_text":"second procedure",
                "positive_chunk_ids":["3"],"negative_chunk_ids":["4"]}]
        dump(self.corpus, corpus)
        dump(self.split / "train_unapproved_candidate.jsonl", train)
        dump(self.split / "dev_unapproved_candidate.jsonl", dev)
        dump(self.split / "quarantined_negatives.jsonl", [
            {"split":"train","query_id":"q1","negative_chunk_id":"4","review_status":"PENDING"}
        ])
        train[0]["quarantined_negative_chunk_ids"]=["4"]
        dump(self.split / "train_unapproved_candidate.jsonl", train)
        families = [{"tenant_id":"6","document_id":str(i),
                     "proposed_canonical_source_family":"family-"+str(i),
                     "evidence_status":"REGISTRY_METADATA_MATCH"}
                    for i in range(1, 5)]
        dump(self.families, families)
        dump(self.candidates, [{"query_id":"q2", "chunk_ids":["3","4"]}])
        report = {"status":"PROVISIONAL_RESPLIT_ONLY","total_queries":2,
                  "input_sha256":{
                      "corpus":sha256(self.corpus),
                      "recommendations":sha256(self.families),
                  },
                  "output_sha256":{
                      key:sha256(self.split / key)
                      for key in (
                          "train_unapproved_candidate.jsonl",
                          "dev_unapproved_candidate.jsonl",
                          "quarantined_negatives.jsonl")}
                  }
        (self.split / "resplit_report.json").write_text(json.dumps(report))

    def audit(self):
        return audited_inputs(self.split,self.corpus,self.families,self.candidates)

    def test_valid_source_split_and_pair_counts(self):
        result=self.audit()
        self.assertEqual(result["train_queries"],1)
        self.assertEqual(result["dev_queries"],1)
        self.assertEqual(len(result["train_pairs"]),2)
        self.assertEqual(len(result["dev_pairs"]),2)
        self.assertEqual(result["candidate_gold_present"],1)

    def test_corrupted_split_fails_sha(self):
        with (self.split/"train_unapproved_candidate.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ExplorationBlocked,"HASH_MISMATCH"):
            self.audit()

    def test_family_overlap_blocks_without_changing_source(self):
        families=[json.loads(l) for l in self.families.read_text().splitlines()]
        families[2]["proposed_canonical_source_family"]=families[0]["proposed_canonical_source_family"]
        dump(self.families,families)
        report=json.loads((self.split/"resplit_report.json").read_text())
        report["input_sha256"]["recommendations"]=sha256(self.families)
        (self.split/"resplit_report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ExplorationBlocked,"PROPOSED_FAMILY_SPLIT_LEAKAGE"):
            self.audit()

    def test_rejected_pseudo_relevance_not_converted_to_approval(self):
        chunk={"1":{"chunk_id":"1","tenant_id":"6","passage":"technical instruction"},
               "2":{"chunk_id":"2","tenant_id":"6","passage":"unrelated"}}
        rows=[{"tenant_id":"6","query_id":"t","query_text":"question",
               "review_status":"unverified",
               "positive_chunk_ids":["1"],"negative_chunk_ids":["2"]}]
        pairs=provisional_pairs(rows,chunk)
        self.assertEqual([x["label"] for x in pairs],[1.0,0.0])
        self.assertEqual(rows[0]["review_status"],"unverified")

    def test_cross_tenant_candidate_rejected(self):
        dump(self.candidates,[{"query_id":"q2","chunk_ids":["3","999"]}])
        with self.assertRaises(ValueError):
            self.audit()


if __name__ == "__main__":
    unittest.main()
