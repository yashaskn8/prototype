"""Synthetic no-private-data tests for the Servy V6 holdout leakage auditor."""
import json
import tempfile
import unittest
from pathlib import Path

from servy_v6_holdout_audit import (
    HoldoutAuditError, audit, audit_files,
)


def question(qid, pos, neg, *, tenant="1", text=None, **extra):
    return {
        "query_id": qid, "tenant_id": tenant,
        "query_text": text or qid, "positive_chunk_ids": pos,
        "negative_chunk_ids": neg, **extra,
    }


def chunk(cid, doc, tenant="1"):
    return {
        "chunk_id": cid, "tenant_id": tenant,
        "document_id": doc, "passage": "synthetic test only",
    }


def family(doc, name, tenant="1"):
    return {
        "tenant_id": tenant, "document_id": doc,
        "proposed_canonical_source_family": name,
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class HoldoutLeakageTests(unittest.TestCase):
    def setUp(self):
        self.train = [question("train", ["a"], ["b"])]
        self.original = list(self.train)
        self.dev = [question("dev", ["c"], ["d"])]
        self.holdout = [
            question("hold", ["e"], ["f"], review_status="approved", reviewer="placeholder")
        ]
        self.corpus = [
            chunk("a", "doc-train"), chunk("b", "doc-train"),
            chunk("c", "doc-dev"), chunk("d", "doc-dev"),
            chunk("e", "doc-hold"), chunk("f", "doc-hold"),
        ]
        self.families = [
            family("doc-train", "family-one"),
            family("doc-dev", "family-two"),
            family("doc-hold", "family-three"),
        ]

    def check(self):
        return audit(
            self.holdout, self.train, self.original,
            self.dev, self.corpus, self.families,
        )

    def test_source_isolated_new_family_is_still_not_certified_independent(self):
        result = self.check()
        self.assertEqual(
            result["status"], "STRUCTURAL_ISOLATION_ONLY_CHRONOLOGY_UNVERIFIED"
        )
        self.assertEqual(result["source_families"]["unused_families_after_original_train_and_dev"], 1)
        self.assertFalse(result["independent_holdout_verified"])
        self.assertFalse(result["valid_for_independent_accuracy_claims"])
        self.assertFalse(result["review_metadata_unverified"]["reviewer_identity_authenticated"])
        self.assertFalse(result["training_authorized"])

    def test_same_source_family_is_blocked_even_with_unique_queries_chunks(self):
        self.families[2]["proposed_canonical_source_family"] = "family-one"
        result = self.check()
        self.assertEqual(result["status"], "BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE")
        self.assertEqual(result["source_families"]["holdout_shared_with_selected_train"], 1)
        self.assertEqual(result["reference_overlap"]["distinct_shared_chunk_ids_with_selected_train"], 0)

    def test_holdout_negative_that_was_train_positive_is_detected(self):
        self.holdout[0]["negative_chunk_ids"] = ["a"]
        result = self.check()
        self.assertEqual(result["reference_overlap"]["holdout_negative_vs_train_positive"], 1)
        self.assertEqual(result["reference_overlap"]["holdout_questions_affected_by_selected_train"], 1)
        self.assertEqual(result["status"], "BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE")

    def test_holdout_positive_that_was_train_negative_is_detected(self):
        self.holdout[0]["positive_chunk_ids"] = ["b"]
        result = self.check()
        self.assertEqual(result["reference_overlap"]["holdout_positive_vs_train_negative"], 1)
        self.assertFalse(result["independent_holdout_verified"])

    def test_missing_gold_blocks_complete_scoring_without_mutating_labels(self):
        self.holdout[0]["positive_chunk_ids"] = ["unresolvable"]
        result = self.check()
        self.assertEqual(result["evidence_integrity"]["holdout_questions_with_missing_or_empty_gold"], 1)
        self.assertEqual(result["evidence_integrity"]["holdout_questions_with_no_positive_references"], 0)
        self.assertEqual(result["evidence_integrity"]["holdout_questions_with_unresolved_positive_references"], 1)
        self.assertEqual(result["evidence_integrity"]["holdout_questions_with_resolvable_positive_references"], 0)
        self.assertEqual(result["evidence_integrity"]["reference_errors_by_split"]["holdout"]["missing_reference_count"], 1)
        self.assertEqual(result["status"], "BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE")
        self.assertEqual(self.holdout[0]["positive_chunk_ids"], ["unresolvable"])

    def test_empty_gold_is_separated_from_unresolved_gold(self):
        self.holdout[0]["positive_chunk_ids"] = []
        result = self.check()
        evidence = result["evidence_integrity"]
        self.assertEqual(evidence["holdout_questions_with_no_positive_references"], 1)
        self.assertEqual(evidence["holdout_questions_with_unresolved_positive_references"], 0)
        self.assertEqual(evidence["holdout_questions_with_resolvable_positive_references"], 0)
        self.assertEqual(evidence["reference_errors_by_split"]["holdout"]["missing_reference_count"], 0)
        self.assertEqual(evidence["empty_gold_semantic_interpretation"],
                         "UNDETERMINED_POSSIBLE_ABSTENTION")
        self.assertEqual(result["status"], "BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE")

    def test_mixed_empty_and_unresolved_gold_counts_are_separate(self):
        self.holdout[0]["positive_chunk_ids"] = []
        self.holdout.append(question("second", ["unknown-chunk"], ["f"]))
        evidence = self.check()["evidence_integrity"]
        self.assertEqual(evidence["holdout_questions_with_missing_or_empty_gold"], 2)
        self.assertEqual(evidence["holdout_questions_with_no_positive_references"], 1)
        self.assertEqual(evidence["holdout_questions_with_unresolved_positive_references"], 1)
        self.assertEqual(evidence["holdout_questions_with_resolvable_positive_references"], 0)

    def test_wrong_tenant_positive_is_flagged(self):
        self.holdout[0]["positive_chunk_ids"] = ["a"]
        self.holdout[0]["tenant_id"] = "other-tenant"
        result = self.check()
        self.assertEqual(result["evidence_integrity"]["holdout_questions_with_cross_tenant_gold"], 1)
        self.assertEqual(result["status"], "BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE")

    def test_duplicate_query_ids_are_rejected(self):
        self.holdout.append(dict(self.holdout[0]))
        with self.assertRaisesRegex(HoldoutAuditError, "DUPLICATE_QUERY_ID"):
            self.check()

    def test_same_normalized_question_is_detected(self):
        self.holdout[0]["query_text"] = " TRAIN !!! "
        result = self.check()
        self.assertEqual(result["question_overlap"]["selected_train_normalized_text"], 1)
        self.assertFalse(result["independent_holdout_verified"])

    def test_selected_must_be_part_of_original_query_set(self):
        self.train.append(question("new", ["a"], ["b"]))
        with self.assertRaisesRegex(HoldoutAuditError, "SELECTED_QUERIES_NOT_IN_ORIGINAL"):
            self.check()

    def test_incomplete_family_map_rejected(self):
        self.families.pop()
        with self.assertRaisesRegex(HoldoutAuditError, "CORPUS_SOURCE_FAMILY_MAP_INCOMPLETE"):
            self.check()

    def test_original_train_history_flags_contamination_even_if_selected_differs(self):
        self.original.append(question("historical", ["e"], ["f"]))
        result = self.check()
        self.assertEqual(result["source_families"]["holdout_shared_with_selected_train"], 0)
        self.assertEqual(result["source_families"]["holdout_shared_with_historical_train_or_dev"], 1)
        self.assertEqual(result["reference_overlap"]["shared_with_original_train_chunk_ids"], 2)
        self.assertFalse(result["independent_holdout_verified"])

    def test_report_contains_only_aggregate_not_private_question_text(self):
        data = json.dumps(self.check())
        self.assertNotIn("synthetic test only", data)
        self.assertNotIn('"query_text"', data)
        self.assertNotIn('"query_id"', data)

    def test_file_audit_includes_hashes_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = {}
            for name, rows in {
                "holdout": self.holdout, "selected_train": self.train,
                "original_train": self.original, "dev": self.dev,
                "corpus": self.corpus, "families": self.families,
            }.items():
                path = root / (name + ".jsonl")
                write_jsonl(path, rows)
                args[name] = path
            report = audit_files(**args)
            self.assertEqual(len(report["input_sha256"]), 6)
            self.assertEqual(report["question_counts"]["holdout"], 1)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             sorted(name + ".jsonl" for name in args))


if __name__ == "__main__":
    unittest.main()
