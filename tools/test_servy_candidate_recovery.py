"""Synthetic research-only tests. No model, network or proprietary content."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import servy_candidate_recovery as recovery
from servy_candidate_recovery import rescue_candidates, lexical_terms
from servy_checkpoint_audit import normalize_inputs, contextual_passage, rerank_fixed


class CandidateRecoveryTests(unittest.TestCase):
    def test_generate_only_cli_verifies_inputs_preserves_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private_training_data"
            root.mkdir()
            def write(name, rows):
                path = root / name
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                return path
            corpus = write("corpus", [{"chunk_id": "x", "document_id": "d", "tenant_id": "t", "passage": "Inspect cable"}])
            cases = write("cases", [{"query_id": str(i), "tenant_id": "t", "query": "inspect cable", "positive_chunk_id": "x"} for i in range(42)])
            candidates = write("candidates", [{"query_id": str(i), "chunk_ids": ["x"]} for i in range(42)])
            from servy_checkpoint_audit import evaluate_one
            normalized, chunks = normalize_inputs(recovery.load_jsonl(cases), recovery.load_jsonl(corpus))
            frozen = {str(i): ["x"] for i in range(42)}
            result = evaluate_one(normalized, chunks, frozen, lambda pairs: [1.0])
            recorded = write("recorded", result["queries"])
            paths = dict(cases=cases, corpus=corpus, candidates=candidates, recorded=recorded)
            pin = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
            manifest = root / "manifest"
            manifest.write_text(json.dumps({"input_sha256": {"corpus": pin["corpus"],
                "dev_split": pin["cases"], "frozen_candidates": pin["candidates"]},
                "v6_weights_sha256": recovery.V6_SHA}))
            families = write("families", [{"document_id": "d", "family": "f"}])
            args = argparse.Namespace(**{k: str(v) for k, v in paths.items()},
                families=str(families), manifest=str(manifest), out_dir=str(root / "new"),
                generate_only=True, enable_lexical_rescue=False, enable_context_rerank=False,
                model=None, device="cpu")
            with patch.dict(recovery.PINNED, pin), patch("builtins.print"):
                report = recovery.run(args)
                self.assertFalse(report["new_accuracy_measured"])
                self.assertEqual(report["baseline"]["hit_at_1"], 100)
                self.assertTrue((root / "new" / "experimental_candidates.jsonl").is_file())
                self.assertEqual(pin, {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
                with self.assertRaisesRegex(ValueError, "NEW_PRIVATE"):
                    recovery.run(args)
                # Exercise the real scoring orchestration with a synthetic
                # local checkpoint/scorer. Never represent this as V6 accuracy.
                model_dir = root / "checkpoint"
                model_dir.mkdir()
                (model_dir / "model.safetensors").write_bytes(b"synthetic weights")
                digest = hashlib.sha256(b"synthetic weights").hexdigest()
                m = json.loads(manifest.read_text())
                m["v6_weights_sha256"] = digest
                manifest.write_text(json.dumps(m))
                args.model = str(model_dir)
                args.generate_only = False
                args.enable_context_rerank = True
                args.out_dir = str(root / "scored")
                calls = []
                class Scores(list):
                    def reshape(self, *args): return self
                    def tolist(self): return list(self)
                class Model:
                    num_labels = 1
                    def __init__(self, *args, **kwargs):
                        self_outer.assertTrue(kwargs["local_files_only"])
                    def predict(self, pairs, **kwargs):
                        calls.extend(pairs)
                        return Scores([1.0] * len(pairs))
                self_outer = self
                with patch.object(recovery, "V6_SHA", digest), patch.dict("sys.modules", {
                    "sentence_transformers": SimpleNamespace(CrossEncoder=Model),
                    "torch": SimpleNamespace(nn=SimpleNamespace(Identity=lambda: None))}), patch.dict("os.environ"):
                    scored = recovery.run(args)
                self.assertTrue(scored["new_accuracy_measured"])
                self.assertEqual(scored["comparisons"]["context_only"]["decision"], "NO_TOP1_GAIN")
                self.assertEqual(len(calls), 84)  # baseline AND separate contextual arm
                args.out_dir = str(root / "other")
                corpus.write_text("changed")
                with self.assertRaisesRegex(ValueError, "ORIGINAL_INPUT_SHA"):
                    recovery.run(args)

    def setUp(self):
        self.chunks = {str(i): {"chunk_id": str(i), "document_id": "d", "tenant_id": "a",
                                "passage": "unrelated overview"} for i in range(12)}
        self.chunks["10"]["passage"] = "Battery rated 24 V DC."
        self.chunks["11"] = {**self.chunks["10"], "tenant_id": "b", "passage": "24 V battery battery"}
        self.cases = {"q": {"query": "battery voltage specification 24", "tenant_id": "a"}}
        self.frozen = {"q": [str(i) for i in range(10)]}

    def test_rescue_without_gold_preserves_prefix_identity_and_inputs(self):
        snapshot = copy.deepcopy((self.cases, self.chunks, self.frozen))
        result = rescue_candidates(self.cases, self.chunks, self.frozen)["q"]
        self.assertEqual(result[:3], self.frozen["q"][:3])
        self.assertIn("10", result[:5])
        self.assertNotIn("11", result)
        self.assertEqual(len(set(result)), 10)
        self.assertEqual(snapshot, (self.cases, self.chunks, self.frozen))

    def test_narrower_authorization_excludes_restricted_source(self):
        allowed = {"q": set(self.frozen["q"])}  # source 10 is confidential to this caller
        result = rescue_candidates(self.cases, self.chunks, self.frozen, allowed_by_query=allowed)
        self.assertNotIn("10", result["q"])
        allowed["q"].remove("0")
        with self.assertRaisesRegex(ValueError, "ineligible"):
            rescue_candidates(self.cases, self.chunks, self.frozen, allowed_by_query=allowed)

    def test_cross_tenant_baseline_rejected(self):
        self.frozen["q"][0] = "11"
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            rescue_candidates(self.cases, self.chunks, self.frozen)

    def test_gold_labels_cannot_drive_generation(self):
        a = rescue_candidates(self.cases, self.chunks, self.frozen)
        self.cases["q"]["gold_chunk_ids"] = ["9"]
        self.cases["q"]["gold_document_ids"] = ["invented"]
        self.assertEqual(a, rescue_candidates(self.cases, self.chunks, self.frozen))

    def test_voltage_normalization_does_not_invent_units_for_steps(self):
        self.assertIn("24volt", lexical_terms("operating v specification 24"))
        self.assertIn("24volt", lexical_terms("24 volts"))
        self.assertNotIn("24volt", lexical_terms("step 24 check battery"))

    def test_context_only_from_corpus_not_gold_document_title(self):
        rows = [{"query_id": "q", "query": "inspect", "tenant_id": "a", "positive_chunk_id": "x",
                 "doc_title": "Forbidden label-derived title"}]
        corpus = [{"chunk_id": "x", "document_id": "d", "tenant_id": "a", "passage": "## Inspection\nCheck cable"}]
        _, chunks = normalize_inputs(rows, corpus)
        self.assertEqual(contextual_passage(chunks["x"]), corpus[0]["passage"])
        corpus[0]["document_title"] = "Verified device manual"
        _, chunks = normalize_inputs(rows, corpus)
        self.assertTrue(contextual_passage(chunks["x"]).startswith("Verified device manual"))

    def test_context_is_opt_in_and_tail_cannot_be_promoted(self):
        self.chunks["0"]["document_title"] = "Source title"
        seen = []
        def predict(pairs):
            seen.append(pairs)
            return [0, 1, 100]
        ids = ["0", "1", "2"]
        ranked, _ = rerank_fixed("inspect", ids, self.chunks, predict, pool_size=2)
        self.assertEqual(ranked, ["1", "0", "2"])
        self.assertNotIn("Source title", seen[-1][0][1])
        rerank_fixed("inspect", ids, self.chunks, predict, pool_size=2, contextual=True)
        self.assertIn("Source title", seen[-1][0][1])


if __name__ == "__main__":
    unittest.main()
