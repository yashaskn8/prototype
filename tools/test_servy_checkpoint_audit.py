"""Synthetic CPU-only regression tests; no database or external model."""
import unittest
import csv
import json
import tempfile
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from servy_checkpoint_audit import (
    normalize_inputs, validate_candidates, check_split_overlap,
    rerank_fixed, evaluate_one, checkpoint_fingerprint, load_training_input,
    run,
)


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": 1, "tenant_id": 9, "document_id": 31, "passage": "pressure valve"},
            {"chunk_id": 2, "tenant_id": 9, "document_id": 32, "passage": "temperature alert"},
            {"chunk_id": 3, "tenant_id": 9, "document_id": 31, "passage": "valve details"},
            {"chunk_id": 4, "tenant_id": 8, "document_id": 99, "passage": "other tenant"},
        ]
        self.dev = [{"query_id": "q-1", "tenant_id": 9,
                     "query_text": "pressure valve", "positive_chunk_ids": [1, 3]}]
        self.cases, self.chunks = normalize_inputs(self.dev, self.corpus)

    def test_fixed_candidate_comparison(self):
        candidates = validate_candidates(self.cases, self.chunks,
            [{"query_id": "q-1", "chunk_ids": [2, 1, 3]}])
        before = evaluate_one(self.cases, self.chunks, candidates,
                              lambda pairs: [4.0, 1.0, 0.5], pool_size=2)
        after = evaluate_one(self.cases, self.chunks, candidates,
                             lambda pairs: [-2.0, 9.0, 0.5], pool_size=2)
        self.assertEqual(before["queries"][0]["gold_rank"], 2)
        self.assertEqual(after["queries"][0]["gold_rank"], 1)
        self.assertEqual(before["metrics"]["hit_at_1"], 0)
        self.assertEqual(after["metrics"]["hit_at_1"], 100)
        self.assertEqual(after["queries"][0]["ranked_chunk_ids"][-1], "3")

    def test_reject_cross_tenant_candidate(self):
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            validate_candidates(self.cases, self.chunks,
                                [{"query_id": "q-1", "chunk_ids": [4, 1]}])

    def test_reject_cross_tenant_gold(self):
        with self.assertRaisesRegex(ValueError, "different tenant"):
            normalize_inputs([{"query_id": "q", "tenant_id": 9,
                               "query_text": "x", "chunk_id": 4}], self.corpus)

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate corpus"):
            normalize_inputs(self.dev, self.corpus + self.corpus[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            normalize_inputs(self.dev * 2, self.corpus)

    def test_reject_missing_or_duplicate_candidates(self):
        with self.assertRaisesRegex(ValueError, "Candidate query IDs differ"):
            validate_candidates(self.cases, self.chunks, [])
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            validate_candidates(self.cases, self.chunks,
                                [{"query_id": "q-1", "chunk_ids": [1, 1]}])

    def test_overlap_checks(self):
        collisions, unknown = check_split_overlap(self.cases, [
            {"query_text": " Pressure  Valve ", "positive_chunk_ids": [1]}
        ], self.chunks)
        self.assertEqual(unknown, [])
        self.assertEqual(collisions[0]["collision_types"],
                         ["query", "gold_chunk", "source_document"])

    def test_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "score count"):
            rerank_fixed("q", ["1", "2"], self.chunks, lambda pairs: [1])
        with self.assertRaisesRegex(ValueError, "Invalid reranker score"):
            rerank_fixed("q", ["1"], self.chunks,
                         lambda pairs: [float("nan")])

    def test_ties_and_untouched_tail(self):
        ranks, scores = rerank_fixed("q", ["2", "1", "3"],
                                      self.chunks, lambda pairs: [0, 0, 999], 2)
        self.assertEqual(ranks, ["2", "1", "3"])
        self.assertEqual(scores, {"2": 0, "1": 0, "3": 999})


    def test_fail_closed_if_gold_missing_from_corpus(self):
        with self.assertRaisesRegex(ValueError, "absent from exported corpus"):
            normalize_inputs([{"query_id": "missing", "tenant_id": 9,
                               "query": "pressure", "positive_chunk_ids": [12345]}],
                             self.corpus)

    def test_fail_closed_if_candidate_list_empty(self):
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            validate_candidates(self.cases, self.chunks,
                                [{"query_id": "q-1", "chunk_ids": []}])

    def test_negative_chunk_causes_document_overlap(self):
        collisions, unknown = check_split_overlap(
            self.cases,
            [{"query_text": "unrelated", "positive_chunk_ids": [2],
              "negative_chunk_ids": [3]}],
            self.chunks)
        self.assertEqual(unknown, [])
        self.assertIn("source_document", collisions[0]["collision_types"])
        self.assertIn("gold_chunk", collisions[0]["collision_types"])

    def test_stage2_text_only_matches_both_passages(self):
        rows = [{"query": "unrelated question",
                 "positive": "temperature alert",
                 "candidate_negative": "valve details"}]
        collisions, unknown = check_split_overlap(self.cases, rows, self.chunks)
        self.assertEqual(unknown, [])
        self.assertIn("source_document", collisions[0]["collision_types"])

    def test_unknown_stage2_passage_fails_provenance(self):
        collisions, unknown = check_split_overlap(
            self.cases,
            [{"query": "unrelated", "positive": "temperature alert",
              "candidate_negative": "UNMAPPED PROPRIETARY GUIDE"}],
            self.chunks)
        self.assertEqual(collisions, [])
        self.assertTrue(any("no exact passage match" in x["reason"] for x in unknown))

    def test_checkpoints_have_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "checkpoint"
            folder.mkdir()
            (folder / "weights.bin").write_bytes(b"abc")
            one = checkpoint_fingerprint(str(folder))
            self.assertEqual(one["status"], "LOCAL_DIRECTORY")
            self.assertEqual(len(one["sha256"]), 64)
            (folder / "weights.bin").write_bytes(b"xyz")
            two = checkpoint_fingerprint(str(folder))
            self.assertNotEqual(one["sha256"], two["sha256"])
            with self.assertRaises(FileNotFoundError):
                checkpoint_fingerprint(str(folder / "missing"))
        self.assertEqual(
            checkpoint_fingerprint("BAAI/bge-reranker-v2-m3")["status"],
            "REMOTE_MODEL_REVISION_UNPINNED")

    def test_csv_training_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screened.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["query", "positive", "candidate_negative"])
                writer.writeheader()
                writer.writerow({"query": "x", "positive": "y", "candidate_negative": "z"})
            self.assertEqual(load_training_input(path)[0]["candidate_negative"], "z")

    def test_end_to_end_report_no_gpu_or_models(self):
        """Dry-run real I/O with mock model; detects broken CLI/report schema."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def dump(name, rows):
                path = root / name
                path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                encoding="utf-8")
                return str(path)
            corpus_path = dump("corpus.jsonl", self.corpus)
            cases_path = dump("cases.jsonl", self.dev)
            candidates_path = dump("candidates.jsonl", [
                {"query_id": "q-1", "chunk_ids": [2, 1, 3]}
            ])
            train_path = dump("train.jsonl", [
                {"query": "different query", "positive_chunk_ids": [2]}
            ])
            model_path = root / "mock_checkpoint"
            model_path.mkdir()
            (model_path / "model.bin").write_bytes(b"safe test model")

            class FakeEncoder:
                num_labels = 1
                def __init__(self, *args, **kwargs):
                    pass
                def predict(self, pairs, **kwargs):
                    return [3 if "pressure" in passage else 0
                            for _, passage in pairs]

            fake_transformers = types.SimpleNamespace(CrossEncoder=FakeEncoder)
            fake_torch = types.SimpleNamespace(
                nn=types.SimpleNamespace(Identity=lambda: object()),
                cuda=types.SimpleNamespace(is_available=lambda: False))
            args = SimpleNamespace(
                cases=cases_path, corpus=corpus_path, train=[train_path],
                candidates=candidates_path,
                model=["mock=" + str(model_path)],
                device="cpu", top_k=10, pool_size=2,
                out_dir=str(root / "reports"), embedding_model="unused",
                allow_untraceable_training=False)
            with patch.dict(sys.modules, {
                "sentence_transformers": fake_transformers,
                "torch": fake_torch,
            }):
                report = run(args)
            self.assertEqual(report["results"]["mock"]["hit_at_1"], 100)
            self.assertEqual(report["train_overlap_check"],
                             "PARTIAL_EXACT_PROVENANCE_CHECK")
            data = [json.loads(line) for line in
                    (root / "reports" / "per_query.jsonl").read_text().splitlines()
                    if line.strip()]
            row = data[0]
            self.assertEqual(row["mock"]["ranked_chunk_ids"], ["1", "2", "3"])
            self.assertEqual(len(row["mock"]["reranker_scores"]), 3)
            with (root / "reports" / "per_query.csv").open(newline="") as stream:
                cells = list(csv.DictReader(stream))
            self.assertEqual(len(cells), 1)
            self.assertIn("mock_rank_change_vs_baseline", cells[0])

    def test_no_training_input_is_not_verified(self):
        """The tool must not suggest holdout independence without a training file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def dump(name, rows):
                path = root / name
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return str(path)
            data = dump("c.jsonl", self.corpus)
            qs = dump("q.jsonl", self.dev)
            candidates = dump("k.jsonl", [
                {"query_id": "q-1", "chunk_ids": [2, 1]}
            ])
            class FakeEncoder:
                num_labels = 1
                def __init__(self, *a, **kw):
                    pass
                def predict(self, pairs, **kw):
                    return list(range(len(pairs)))
            fake_torch = types.SimpleNamespace(
                nn=types.SimpleNamespace(Identity=lambda: object()),
                cuda=types.SimpleNamespace(is_available=lambda: False))
            args = SimpleNamespace(
                cases=qs, corpus=data, train=None,
                candidates=candidates, model=["base=BAAI/mock"],
                device="cpu", top_k=10, pool_size=2,
                out_dir=str(root / "out"), embedding_model="unused",
                allow_untraceable_training=False)
            with patch.dict(sys.modules, {
                "sentence_transformers": types.SimpleNamespace(CrossEncoder=FakeEncoder),
                "torch": fake_torch,
            }):
                result = run(args)
            self.assertEqual(result["train_overlap_check"], "NOT_PERFORMED")
            self.assertFalse(result["holdout_independent"])


if __name__ == "__main__":
    unittest.main()
