"""Offline synthetic tests for private artifact discovery and dataset integrity."""
import json
import tempfile
import unittest
from pathlib import Path

from servy_v5_artifact_audit import (
    PrivateArtifactError, audit_experiment, discover,
    validate_candidate_files, validate_selected_subset,
)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        split = (self.root / "v5_gemini_semantic_audit_v1" /
                 "input_snapshot" / "split")
        self.original = [
            {"query_id": "t1", "tenant_id": 2, "query_text": "Check pump", 
             "positive_chunk_ids": ["g1"], "negative_chunk_ids": ["n1"]},
            {"query_id": "t2", "tenant_id": 2, "query_text": "Pressure issue",
             "positive_chunk_ids": ["g2"], "negative_chunk_ids": ["n2"]},
        ]
        write_jsonl(split / "train_unapproved_candidate.jsonl", self.original)
        write_jsonl(split / "dev_unapproved_candidate.jsonl", [
            {"query_id": "d1", "tenant_id": 2, "query_text": "Dev check",
             "positive_chunk_ids": ["g3"], "negative_chunk_ids": ["n3"]}
        ])
        selected = (self.root / "v5_training_candidates_20261019T083815_575655Z" /
                    "train_keep_ai_provisional.jsonl")
        write_jsonl(selected, [self.original[0]])
        write_jsonl(self.root / "qwen3_candidates_top10.jsonl",
                    [{"query_id": "d1", "chunk_ids": ["g3", "n3"]}])
        write_jsonl(self.root / "hybrid_rrf60_candidates_top10.jsonl",
                    [{"query_id": "d1", "chunk_ids": ["n3", "g3"]}])

    def test_four_known_files_are_discovered(self):
        paths = discover(self.root)
        self.assertEqual(paths["original_train"].name, "train_unapproved_candidate.jsonl")
        self.assertEqual(paths["selected_train"].name, "train_keep_ai_provisional.jsonl")
        self.assertEqual(paths["qwen3_candidates"].name, "qwen3_candidates_top10.jsonl")
        self.assertEqual(paths["hybrid_candidates"].name, "hybrid_rrf60_candidates_top10.jsonl")

    def test_valid_structural_audit_does_not_approve_training(self):
        result = audit_experiment(self.root)
        self.assertEqual(result["status"], "STRUCTURAL_AUDIT_PASSED_RESEARCH_ONLY")
        self.assertFalse(result["fine_tuning_admitted"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["independent_holdout"])
        self.assertFalse(result["source_family_split_verified"])
        self.assertFalse(result["training"]["expert_labels_approved"])
        self.assertEqual(result["training"]["selected_queries"], 1)
        self.assertEqual(result["candidate_comparison"]["identical_top1"], 0)
        self.assertEqual(result["candidate_comparison"]["dev_query_count"], 1)
        self.assertFalse(result["candidate_comparison"]["gold_coverage_verified"])

    def test_modified_gold_label_is_blocked(self):
        other = {**self.original[0], "positive_chunk_ids": ["forged"]}
        with self.assertRaisesRegex(PrivateArtifactError, "MODIFIED"):
            validate_selected_subset(self.original, [other])

    def test_duplicate_selected_query_is_blocked(self):
        with self.assertRaisesRegex(PrivateArtifactError, "DUPLICATE_SELECTED"):
            validate_selected_subset(self.original, [self.original[0], self.original[0]])

    def test_candidate_query_sets_must_match(self):
        with self.assertRaisesRegex(PrivateArtifactError, "QUERY_SETS"):
            validate_candidate_files(
                [{"query_id": "a", "chunk_ids": ["1"]}],
                [{"query_id": "b", "chunk_ids": ["1"]}],
            )

    def test_duplicated_candidate_chunk_is_blocked(self):
        with self.assertRaisesRegex(PrivateArtifactError, "DUPLICATE_CANDIDATE"):
            validate_candidate_files(
                [{"query_id": "a", "chunk_ids": ["1", "1"]}],
                [{"query_id": "a", "chunk_ids": ["2"]}],
            )

    def test_all_training_rows_need_a_negative(self):
        rows = [
            self.original[0],
            {**self.original[1], "negative_chunk_ids": []},
        ]
        with self.assertRaisesRegex(PrivateArtifactError, "EACH_SELECTED_QUERY"):
            validate_selected_subset(rows, rows)

    def test_quarantined_negative_cannot_remain_active(self):
        row = {**self.original[0], "quarantined_negative_chunk_ids": ["n1"]}
        with self.assertRaisesRegex(PrivateArtifactError, "QUARANTINED_NEGATIVE"):
            validate_selected_subset([row], [row])

    def test_missing_candidate_snapshot_is_blocked(self):
        (self.root / "qwen3_candidates_top10.jsonl").unlink()
        with self.assertRaisesRegex(PrivateArtifactError, "MISSING_ARTIFACT"):
            audit_experiment(self.root)

    def test_duplicate_snapshot_directories_are_ambiguous(self):
        second = (self.root / "v5_training_candidates_20261019T999999" /
                  "train_keep_ai_provisional.jsonl")
        write_jsonl(second, [self.original[0]])
        with self.assertRaisesRegex(PrivateArtifactError, "AMBIGUOUS_ARTIFACT"):
            discover(self.root)

    def test_dev_query_set_mismatch_is_blocked(self):
        split = (self.root / "v5_gemini_semantic_audit_v1" /
                 "input_snapshot" / "split")
        write_jsonl(split / "dev_unapproved_candidate.jsonl", [
            {"query_id": "unexpected", "tenant_id": 2}
        ])
        with self.assertRaisesRegex(PrivateArtifactError, "DEV_AND_CANDIDATE"):
            audit_experiment(self.root)


if __name__ == "__main__":
    unittest.main()
