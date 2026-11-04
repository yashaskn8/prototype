"""Offline research experiment. No training, downloads, or serving integration.

Generate-only uses recorded baseline scores solely to verify the OLD ranking.
New candidate/context accuracy is reported only after actual local V6 inference.
All outputs must be in a new private_training_data directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re

from servy_checkpoint_audit import (
    load_jsonl, normalize_inputs, validate_candidates, contextual_passage,
    evaluate_one, _metrics,
)
from servy_v4_prepare_research_artifacts import build_bm25_baseline, words
from servy_accuracy_diagnostics import (
    compare_frozen_pool, sha256_file, write_jsonl_exclusive, compare_candidate_runs,
)

# Identities from the original verified Drive experiment, not relevance labels.
PINNED = {
    "candidates": "6b05c3489a7b64e9435589a1aefd179b6a6791d67f12a853d5cec4d0019d0ba1",
    "corpus": "50792737f0b297d2deb63e6a4222fd75fa569fd1ea7d104395517d60b39bd2fe",
    "cases": "c8f391b9591da9e3a7e4a97aedc117254a4c90169b382ffab550183620a4559c",
    "recorded": "9e462fd5f98928a2c0740e490ed4bcc718408e7c7287fa2939f0dcac3847386f",
}
V6_SHA = "b2d5a5e028b9488d6e2097e58f7cb6efd469bb2f774ee4aa061f3ddf3fcec25f"


def lexical_terms(text):
    """Canonical voltage tokens in either source or query, without synonyms/labels."""
    text = text.casefold()
    # Explicit voltage units, including reversed 'voltage specification 24'.
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:volts?|vdc|vac|v)\b", r" \1volt ", text)
    text = re.sub(r"\b(?:operating\s+v|voltage)\s+(?:specification\s+)?(\d+(?:\.\d+)?)\b",
                  r" \1volt ", text)
    return words(text)


def rescue_candidates(cases, chunks, frozen, *, allowed_by_query=None):
    """One predeclared strategy: retain Qwen top 3, admit up to 2 lexical
    top-5 candidates absent from Qwen's head, then preserve Qwen tail order.
    Deduplicate and truncate to 10. No gold fields are read. This is NOT RRF.

    An optional narrower eligibility set is enforced before lexical ranking;
    it cannot widen the mandatory tenant boundary. Callers must not mistake
    this research tenant universe for customer authorization.
    """
    validate_candidates(cases, chunks, [
        {"query_id": q, "chunk_ids": ids} for q, ids in frozen.items()
    ])
    context_chunks = {cid: {**c, "passage": contextual_passage(c)} for cid, c in chunks.items()}
    result = {}
    for qid, case in cases.items():
        eligible = {cid: c for cid, c in context_chunks.items()
                    if c["tenant_id"] == case["tenant_id"]
                    and (allowed_by_query is None or cid in allowed_by_query[qid])}
        if any(cid not in eligible for cid in frozen[qid]):
            raise ValueError("Baseline contains ineligible candidate")
        # Candidate generation receives only query and tenant, never positives.
        query = {qid: {"query": case["query"], "tenant_id": case["tenant_id"]}}
        lexical = build_bm25_baseline(query, eligible, pool=5, tokenizer=lexical_terms)[qid]
        original = frozen[qid]
        query_terms = set(lexical_terms(case["query"]))
        extra = [cid for cid in lexical if cid not in original[:3]
                 and query_terms.intersection(lexical_terms(eligible[cid]["passage"]))][:2]
        result[qid] = list(dict.fromkeys(original[:3] + extra + original[3:]))[:10]
    return result


def run(args):
    output = Path(args.out_dir).resolve()
    if "private_training_data" not in {p.casefold() for p in output.parts} or output.exists():
        raise ValueError("NEW_PRIVATE_OUTPUT_DIRECTORY_REQUIRED")
    paths = {key: Path(getattr(args, key)).resolve() for key in PINNED}
    paths.update(manifest=Path(args.manifest).resolve(), families=Path(args.families).resolve())
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    for key, expected in PINNED.items():
        if hashes[key] != expected:
            raise ValueError("ORIGINAL_INPUT_SHA_MISMATCH:" + key)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for key, field in (("candidates", "frozen_candidates"), ("corpus", "corpus"), ("cases", "dev_split")):
        if manifest["input_sha256"][field] != hashes[key]:
            raise ValueError("MANIFEST_INPUT_MISMATCH:" + key)
    if manifest["v6_weights_sha256"] != V6_SHA:
        raise ValueError("MANIFEST_V6_MISMATCH")
    cases, chunks = normalize_inputs(load_jsonl(paths["cases"]), load_jsonl(paths["corpus"]))
    frozen = validate_candidates(cases, chunks, load_jsonl(paths["candidates"]))
    recorded = {"queries": load_jsonl(paths["recorded"])}
    compare_frozen_pool(recorded, recorded, frozen, pool_size=5)
    if len(cases) != 42:
        raise ValueError("EXPECTED_COMPLETE_42_QUERY_DEVELOPMENT_SET")
    for row in recorded["queries"]:
        if row["gold_chunk_ids"] != cases[row["query_id"]]["gold_chunk_ids"]:
            raise ValueError("RECORDED_LABEL_MISMATCH")
    # Verify recorded logits reproduce every baseline ordering without a model.
    for row in recorded["queries"]:
        qid = row["query_id"]
        replay = evaluate_one({qid: cases[qid]}, chunks, {qid: frozen[qid]},
                              lambda pairs: [row["reranker_scores"][cid] for cid in frozen[qid]])
        if replay["queries"][0]["ranked_chunk_ids"] != row["ranked_chunk_ids"]:
            raise ValueError("RECORDED_LOGIT_REPLAY_MISMATCH")
    proposed = rescue_candidates(cases, chunks, frozen) if args.enable_lexical_rescue else copy.deepcopy(frozen)
    baseline = {**recorded, "metrics": _metrics(cases, recorded["queries"])}
    report = {"input_sha256": hashes, "baseline": baseline["metrics"],
              "new_accuracy_measured": False, "customer_authorization_tested": False,
              "independent_accuracy": False, "production_promotion_allowed": False,
              "strategy": "qwen3_keep3_lexical2" if args.enable_lexical_rescue else "unchanged",
              "context_fields_available": sorted({k for c in chunks.values() for k in
                                                  ("document_title", "title", "heading", "section") if k in c})}
    retrieval = []
    for qid, case in cases.items():
        gold = set(case["gold_chunk_ids"])
        rank = lambda ids: next((i for i, cid in enumerate(ids, 1) if cid in gold), None)
        retrieval.append({"query_id": qid, "original_rank": rank(frozen[qid]),
                          "experimental_rank": rank(proposed[qid])})
    report["retrieval_only"] = {
        "per_query": retrieval,
        **{name: {f"recall_at_{k}": round(100 * sum(r[field] is not None and r[field] <= k
                       for r in retrieval) / len(retrieval), 2) for k in (5, 10)}
           for name, field in (("original", "original_rank"), ("experimental", "experimental_rank"))},
    }
    head_ceiling = sum(r["experimental_rank"] is not None and r["experimental_rank"] <= 5
                       for r in retrieval)
    baseline_hits = sum(r["gold_rank"] == 1 for r in recorded["queries"])
    report["candidate_gate"] = ("REJECT_NO_POSSIBLE_TOP1_GAIN" if args.enable_lexical_rescue
                                 and head_ceiling <= baseline_hits else "UNMEASURED_REQUIRES_V6_SCORING")
    report["head5_oracle_hit1_ceiling_count"] = head_ceiling
    outputs = {}
    if not args.generate_only:
        if not args.model or not Path(args.model).is_dir():
            raise ValueError("EXISTING_LOCAL_V6_REQUIRED_NO_DOWNLOADS")
        model_path = Path(args.model).resolve()
        weights = model_path / "model.safetensors"
        if sha256_file(weights) != V6_SHA:
            raise ValueError("V6_CHECKPOINT_SHA_MISMATCH")
        if model_path == output or model_path in output.parents or output in model_path.parents:
            raise ValueError("OUTPUT_OVERLAPS_CHECKPOINT")
        paths["weights"] = weights
        hashes["weights"] = V6_SHA
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(str(model_path), device=args.device, max_length=256, local_files_only=True)
        if model.num_labels != 1:
            raise ValueError("EXPECTED_SINGLE_RANKING_LOGIT")
        def predict(pairs):
            return model.predict(pairs, activation_fn=torch.nn.Identity(), show_progress_bar=False).reshape(-1).tolist()
        actual = evaluate_one(cases, chunks, frozen, predict, pool_size=5)
        compare_frozen_pool(recorded, actual, frozen, pool_size=5)
        recorded_by_id = {r["query_id"]: r for r in recorded["queries"]}
        if any(r["ranked_chunk_ids"] != recorded_by_id[r["query_id"]]["ranked_chunk_ids"]
               for r in actual["queries"]):
            raise ValueError("ACTUAL_V6_BASELINE_ORDER_DRIFT")
        outputs["baseline"] = actual
        if args.enable_lexical_rescue:
            # Fresh inference includes every new candidate. No borrowed logits.
            outputs["lexical_rescue"] = evaluate_one(cases, chunks, proposed, predict, pool_size=5)
        if args.enable_context_rerank:
            outputs["context_only"] = evaluate_one(cases, chunks, frozen, predict, pool_size=5, contextual=True)
        report["comparisons"] = {name: compare_candidate_runs(actual, result, frozen,
            proposed if name == "lexical_rescue" else frozen)
            for name, result in outputs.items() if name != "baseline"}
        report["new_accuracy_measured"] = bool(report["comparisons"])
    if {k: sha256_file(p) for k, p in paths.items()} != hashes:
        raise ValueError("INPUT_DRIFT_NO_VALID_REPORT")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_exclusive(output / "experimental_candidates.jsonl", [
        {"query_id": q, "chunk_ids": ids} for q, ids in proposed.items()])
    for name, result in outputs.items():
        write_jsonl_exclusive(output / (name + "_per_query.jsonl"), result["queries"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "baseline": report["baseline"],
                      "new_accuracy_measured": report["new_accuracy_measured"]}))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (*PINNED, "families", "manifest", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--model", help="Existing local V6 checkpoint directory; never downloaded")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--enable-lexical-rescue", action="store_true")
    p.add_argument("--enable-context-rerank", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
