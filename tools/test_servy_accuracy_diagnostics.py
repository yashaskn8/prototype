"""Pure CPU-only diagnostics tests: no private data or model weights."""
import json
from pathlib import Path
import tempfile
import unittest

from servy_accuracy_diagnostics import compare_frozen_pool, write_jsonl_exclusive


def case(qid, frozen, ordered, gold):
    original = next((i for i, cid in enumerate(frozen, 1) if cid in gold), None)
    rank = next((i for i, cid in enumerate(ordered, 1) if cid in gold), None)
    return {
        "query_id": qid,
        "candidate_chunk_ids": list(frozen),
        "ranked_chunk_ids": list(ordered),
        "gold_chunk_ids": list(gold),
        "original_gold_rank": original,
        "gold_rank": rank,
    }


class PairedAccuracyDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.frozen = {
            "q1": ["a", "b", "c"],
            "q2": ["d", "e", "f"],
            "q3": ["g", "h", "i"],
            "q4": ["j", "k"],
        }
        self.before = {"queries": [
            case("q1", self.frozen["q1"], ["a", "b", "c"], {"b"}),
            case("q2", self.frozen["q2"], ["e", "d", "f"], {"e"}),
            case("q3", self.frozen["q3"], ["g", "h", "i"], {"i"}),
            case("q4", self.frozen["q4"], ["j", "k"], {"not-in-candidates"}),
        ]}
        self.after = {"queries": [
            case("q1", self.frozen["q1"], ["b", "a", "c"], {"b"}),
            case("q2", self.frozen["q2"], ["d", "e", "f"], {"e"}),
            case("q3", self.frozen["q3"], ["g", "h", "i"], {"i"}),
            case("q4", self.frozen["q4"], ["j", "k"], {"not-in-candidates"}),
        ]}

    def test_ceiling_separates_retrieval_failures_from_reranking(self):
        result = compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)
        s = result["summary"]
        self.assertEqual(s["queries"], 4)
        self.assertEqual(s["head_pool_oracle_hit_at_1_ceiling_count"], 2)
        self.assertEqual(s["candidate_recall_at_10_count"], 3)
        self.assertEqual(s["queries_with_gold_outside_head_but_inside_candidates"], 1)
        self.assertEqual(s["queries_missing_gold_from_candidates"], 1)
        self.assertEqual(s["queries_improved_to_top1"], 1)
        self.assertEqual(s["queries_regressed_from_top1"], 1)
        self.assertEqual(s["before_top1_count"], s["after_top1_count"])
        self.assertEqual(s["query_gold_rank_movements"]["up"], 1)
        self.assertEqual(s["query_gold_rank_movements"]["down"], 1)
        self.assertEqual(s["query_gold_rank_movements"]["both_missing"], 1)
        self.assertFalse(s["production_promotion_allowed"])

    def test_top1_same_does_not_prove_same_rankings(self):
        result = compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)
        self.assertEqual(result["summary"]["before_top1_count"], 1)
        self.assertEqual(result["summary"]["after_top1_count"], 1)
        self.assertEqual(result["summary"]["queries_with_different_top_chunk"], 2)

    def test_pool_change_changes_oracle_not_metrics(self):
        a = compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)
        b = compare_frozen_pool(self.before, self.after, self.frozen, pool_size=3)
        self.assertEqual(a["summary"]["head_pool_oracle_hit_at_1_ceiling_count"], 2)
        self.assertEqual(b["summary"]["head_pool_oracle_hit_at_1_ceiling_count"], 3)
        self.assertEqual(a["summary"]["before_top1_count"], b["summary"]["before_top1_count"])

    def test_frozen_candidates_cannot_change_between_runs(self):
        self.after["queries"][0]["candidate_chunk_ids"] = ["b", "a", "c"]
        with self.assertRaisesRegex(ValueError, "candidate scope"):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)

    def test_reported_rank_must_match_actual_order(self):
        self.after["queries"][0]["gold_rank"] = 2
        with self.assertRaisesRegex(ValueError, "contradicts"):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)

    def test_gold_cannot_change(self):
        self.after["queries"][0]["gold_chunk_ids"] = ["a"]
        self.after["queries"][0]["gold_rank"] = 2
        with self.assertRaisesRegex(ValueError, "Gold labels changed"):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)

    def test_missing_query_is_fatal_not_dropped(self):
        self.after["queries"].pop()
        with self.assertRaisesRegex(ValueError, "missing query"):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)

    def test_same_candidates_cannot_lose_gold(self):
        self.after["queries"][0]["ranked_chunk_ids"] = ["a", "b", "bad"]
        with self.assertRaisesRegex(ValueError, "candidate scope"):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=2)

    def test_negative_or_zero_pool_rejected(self):
        with self.assertRaises(ValueError):
            compare_frozen_pool(self.before, self.after, self.frozen, pool_size=0)

    def test_private_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paired.jsonl"
            write_jsonl_exclusive(path, [{"query_id": "private", "rank": 1}])
            self.assertEqual(json.loads(path.read_text().strip())["rank"], 1)
            with self.assertRaises(FileExistsError):
                write_jsonl_exclusive(path, [{"query_id": "private", "rank": 9}])
            self.assertEqual(json.loads(path.read_text().strip())["rank"], 1)


if __name__ == "__main__":
    unittest.main()
