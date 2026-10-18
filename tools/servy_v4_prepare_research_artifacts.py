"""Prepare *provisional* Servy V4 research artifacts without approvals or training.

Offline/CPU only. Outputs include public model metadata and an optional frozen
tenant-filtered BM25 retrieval baseline. BM25 is NOT Qwen retrieval; use
--retrieval-candidates to freeze real retriever output instead.

Never upgrades weak/provisional labels or claims research admission.
No Docker, network, GPU, database, private upload or destructive operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from servy_checkpoint_audit import (
    load_jsonl, normalize_inputs, validate_candidates,
)

BGE_MODEL = "BAAI/bge-reranker-v2-m3"
# Public Hugging Face model repository commit (not a placeholder).
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_REVISION_URL = ("https://huggingface.co/" + BGE_MODEL +
                      "/tree/" + BGE_REVISION)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def words(value):
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def make_config(pool=10):
    if not isinstance(pool, int) or isinstance(pool, bool) or not (5 <= pool <= 25):
        raise ValueError("candidate_pool must be 5-25")
    return {
        "base_model_id": BGE_MODEL,
        "base_revision": BGE_REVISION,
        "tokenizer_id": BGE_MODEL,
        "tokenizer_revision": BGE_REVISION,
        "sentence_transformers_version": "5.1.0",
        "loss_name": "BinaryCrossEntropyLoss",
        "max_seq_length": 256,
        "candidate_pool": pool,
        "query_format": "raw-question",
        "passage_format": "heading-body",
    }


def build_bm25_baseline(cases, chunks, pool=10):
    """Deterministic unsupervised tenant-filtered BM25, never uses gold labels.

    Gold IDs do not appear in this algorithm; never insert them into a pool.
    Corpus chunk identities remain stable and ties resolve by chunk ID.
    """
    if not (5 <= pool <= 25):
        raise ValueError("pool must be 5-25")
    docs = {}
    counts = {}
    df_by_tenant = defaultdict(Counter)
    tokens_by_tenant = defaultdict(int)
    n_by_tenant = Counter()
    for cid, chunk in chunks.items():
        tenant = chunk["tenant_id"]
        terms = words(chunk["passage"])
        if not terms:
            raise ValueError("Empty candidate passage: " + cid)
        docs[cid] = terms
        counts[cid] = Counter(terms)
        df_by_tenant[tenant].update(set(terms))
        tokens_by_tenant[tenant] += len(terms)
        n_by_tenant[tenant] += 1
    result = {}
    for qid, case in cases.items():
        tenant = case["tenant_id"]
        eligible = sorted(cid for cid in chunks if chunks[cid]["tenant_id"] == tenant)
        if not eligible:
            raise ValueError("No tenant-approved candidates for query " + qid)
        n = n_by_tenant[tenant]
        avgdl = tokens_by_tenant[tenant] / n
        query_terms = Counter(words(case["query"]))
        scored = []
        for cid in eligible:
            tf = counts[cid]
            dl = len(docs[cid])
            score = 0.0
            for word, q_weight in query_terms.items():
                freq = tf.get(word, 0)
                if not freq:
                    continue
                document_freq = df_by_tenant[tenant][word]
                idf = math.log(1.0 + (n - document_freq + 0.5) /
                               (document_freq + 0.5))
                score += q_weight * idf * (2.2 * freq) / (
                    freq + 1.2 * (0.25 + 0.75 * dl / avgdl))
            scored.append((-score, cid))
        scored.sort()
        result[qid] = [cid for _, cid in scored[:pool]]
    return result


def freeze(dev, corpus, output, pool=10, method="bm25", retrieval_candidates=None):
    output = Path(output).resolve()
    if "private_training_data" not in {p.casefold() for p in output.parts}:
        raise ValueError("Artifacts must be written under private_training_data")
    if output.exists():
        raise FileExistsError("Refusing overwrite of any research artifact")
    if not Path(dev).is_file() or not Path(corpus).is_file():
        raise FileNotFoundError("Actual dev/corpus JSONL files required")
    if method == "bm25" and retrieval_candidates is not None:
        raise ValueError("Specify qwen-export with --retrieval-candidates")
    if method == "qwen-export" and not retrieval_candidates:
        raise ValueError("Qwen export path is required; no model output is invented")
    cases, chunks = normalize_inputs(load_jsonl(dev), load_jsonl(corpus))
    if method == "bm25":
        candidate_map = build_bm25_baseline(cases, chunks, pool)
    elif method == "qwen-export":
        candidate_map = validate_candidates(
            cases, chunks, load_jsonl(retrieval_candidates), pool)
    else:
        raise ValueError("Unsupported candidate retrieval method")
    rows = [{"query_id": qid, "chunk_ids": candidate_map[qid]} for qid in cases]
    checked = validate_candidates(cases, chunks, rows, pool)
    if checked != candidate_map:
        raise ValueError("Frozen candidates differ from computed ranking")
    # Explicitly report, never repair a missing gold candidate.
    covered = sum(bool(set(case["gold_chunk_ids"]) & set(checked[qid]))
                  for qid, case in cases.items())
    model = make_config(pool)
    metadata = {
        "schema": "servy-v4-provisional-research-artifacts-1",
        "status": "NOT_APPROVED_FOR_TRAINING",
        "model_id": BGE_MODEL,
        "model_revision_url": MODEL_REVISION_URL,
        "model_revision": BGE_REVISION,
        "candidate_method": method,
        "method_warning": ("BM25 lexical retrieval baseline; NOT the Qwen "
                           "embedding retrieval baseline used in earlier "
                           "research runs." if method == "bm25" else
                           "Imported actual Qwen retrieval export; provenance "
                           "of the external ranking must be independently reviewed."),
        "evaluation_questions": len(cases),
        "tenant_filtered": True,
        "candidate_pool": pool,
        "gold_candidate_coverage_diagnostic_only": {
            "present": covered, "absent": len(cases) - covered,
        },
        "no_gold_injection": True,
        "human_review_complete": False,
        "company_authorization_supplied": False,
        "training_performed": False,
        "input_sha256": {"dev": sha256(dev), "corpus": sha256(corpus)},
    }
    if retrieval_candidates:
        metadata["input_sha256"]["retrieval_candidates"] = sha256(retrieval_candidates)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "model.json"
    candidates_path = output / "frozen_candidates.jsonl"
    config_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    with candidates_path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata["output_sha256"] = {
        "model": sha256(config_path),
        "candidates": sha256(candidates_path),
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--candidate-pool", type=int, default=10)
    p.add_argument("--method", choices=["bm25", "qwen-export"], default="bm25")
    p.add_argument("--retrieval-candidates")
    args = p.parse_args()
    result = freeze(args.dev, args.corpus, args.out_dir,
                    pool=args.candidate_pool, method=args.method,
                    retrieval_candidates=args.retrieval_candidates)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
