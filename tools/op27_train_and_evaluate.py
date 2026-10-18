"""Op27 Phase 4 — Intelligent Fine-Tuning & Multi-Track Evaluation.

Executes scientifically controlled fine-tuning for both models:
1. MiniLM (all-MiniLM-L6-v2) using repaired dataset (no query prefix).
2. BGE-small (BAAI/bge-small-en-v1.5) using repaired dataset (with BGE instruction prefix).

For each model:
- Trains with MultipleNegativesRankingLoss (BatchSamplers.NO_DUPLICATES).
- Saves checkpoint and computes SHA-256 fingerprint.
- Builds an isolated, fully stamped Chroma collection (IndexSchema 2.0).
- Runs fresh evaluation on 43 relevance queries (pure dense + full production hybrid).
- Compares against Phase 0 baselines.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("SERVY_DB_ENGINE", "mysql")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import django
django.setup()

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import chromadb
from core.models import Tenant, KnowledgeChunk
from core.ml.benchmark_manifest import resolve_query_asset
from core.ml.embeddings import (
    PROFILES,
    EmbeddingProfile,
    cached_checkpoint_fingerprint,
    INDEX_SCHEMA_VERSION,
    PASSAGE_FORMAT_VERSION,
)
from core.services.retriever import retrieve
import core.services.retriever as retriever_module

from tools.op27_passages import op27_passage_text
from tools.op27_split_integrity import enforce_train_dev_separation

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_dataset(split: str):
    path = BASE_DIR / "private_training_data" / f"op27_{split}_repaired.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    return rows


def load_holdout():
    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]
    return rows, relevance, abstention


def resolve_chunk_texts(rows, *, content_first=False):
    """Resolve every source exactly; abort before training on any missing label."""
    all_ids = {
        cid for row in rows for cid in
        row.get("positive_chunk_ids", []) + row.get("negative_chunk_ids", [])
    }
    chunks = {
        c.id: c for c in KnowledgeChunk.objects.filter(
            id__in=all_ids
        ).select_related("document", "document__asset", "document__product",
                         "document__category", "document__domain")
    }
    resolved = []
    for row in rows:
        qid = row.get("query_id", "?")
        positive = row.get("positive_chunk_ids", [])
        negative = row.get("negative_chunk_ids", [])
        if not positive:
            raise ValueError(f"{qid}: no positive chunk IDs")
        missing = [cid for cid in positive + negative if cid not in chunks]
        if missing:
            raise ValueError(f"{qid}: missing training chunk IDs: {missing}")
        for cid in positive + negative:
            c = chunks[cid]
            if str(c.tenant_id) != str(row.get("tenant_id")):
                raise ValueError(f"{qid}: cross-tenant chunk {cid}")
            if c.is_quarantined or not c.document.is_rag_enabled:
                raise ValueError(f"{qid}: disallowed training chunk {cid}")
        pos_texts = [
            op27_passage_text(chunks[cid], content_first=content_first)
            for cid in positive
        ]
        neg_texts = [
            op27_passage_text(chunks[cid], content_first=content_first)
            for cid in negative
        ]
        if any(not text.strip() for text in pos_texts + neg_texts):
            raise ValueError(f"{qid}: empty passage")
        if set(pos_texts) & set(neg_texts):
            raise ValueError(f"{qid}: passage labeled both positive and negative")
        resolved.append({
            **row, "positive_text": pos_texts[0],
            "negative_text": neg_texts[0] if neg_texts else "",
            "positive_document_ids": sorted({chunks[cid].document_id for cid in positive}),
            "negative_document_ids": sorted({chunks[cid].document_id for cid in negative}),
            "positive_texts": pos_texts,
            "negative_texts": neg_texts,
        })
    if not resolved:
        raise ValueError("No training examples after source validation")
    return resolved

def train_single_model(model_id, query_prefix, output_dir, train_resolved, val_resolved,
                       epochs=3, batch_size=16, lr=2e-5, revision=None):
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_id}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'='*80}")

    if not val_resolved or len(val_resolved) < 8:
        raise ValueError("At least eight separate development questions required before training")
    if (output_dir / "final").exists():
        raise FileExistsError("Existing model checkpoint is protected: " + str(output_dir / "final"))
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"device": "cpu"}
    if revision:
        kwargs["revision"] = revision
    model = SentenceTransformer(model_id, **kwargs)
    model.max_seq_length = 256

    has_neg = [r for r in train_resolved if r.get("negative_text")]
    no_neg = [r for r in train_resolved if not r.get("negative_text")]

    train_data = {}
    if has_neg:
        train_data["triplets"] = Dataset.from_dict({
            "anchor": [query_prefix + r["query_text"] for r in has_neg],
            "positive": [r["positive_text"] for r in has_neg],
            "negative": [r["negative_text"] for r in has_neg],
        })
    if no_neg:
        train_data["pairs"] = Dataset.from_dict({
            "anchor": [query_prefix + r["query_text"] for r in no_neg],
            "positive": [r["positive_text"] for r in no_neg],
        })

    if not train_data:
        raise ValueError("Empty resolved training dataset")
    train_dataset = next(iter(train_data.values())) if len(train_data) == 1 else train_data

    # All validation questions must contribute, including positive-only examples.
    # Do not use the repeated final benchmark for checkpoint selection.
    if not val_resolved or len(val_resolved) < 8:
        raise ValueError("At least eight resolved development questions required")
    eval_dataset = Dataset.from_dict({
        "anchor": [query_prefix + r["query_text"] for r in val_resolved],
        "positive": [r["positive_text"] for r in val_resolved],
    })

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        seed=42,
        report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        use_cpu=True,
        fp16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    t0 = time.time()
    trainer.train()
    print(f"Selected checkpoint: {trainer.state.best_model_checkpoint}; "
          f"validation loss: {trainer.state.best_metric}")
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    final_dir = output_dir / "final"
    model.save(str(final_dir))

    # Compute fingerprint
    h = hashlib.sha256()
    for p in sorted(final_dir.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    ckpt_hash = h.hexdigest()
    print(f"Final checkpoint saved to {final_dir} (SHA-256: {ckpt_hash})")
    return final_dir, ckpt_hash


def build_and_stamp_chroma_collection(model_path, collection_name, query_prefix, content_first):
    print(f"\nBuilding and stamping Chroma collection: {collection_name}")
    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

    PROTECTED_COLLECTIONS = {
        "servy_dense_kb",
        "servy_dense_bge_small_v1",
        "servy_company_dense_kb",
        "servy_company_bge_small_v1",
        "servy_company_bge_ft_v1",
    }
    if collection_name in PROTECTED_COLLECTIONS:
        raise ValueError(f"Cannot delete or overwrite protected production collection: {collection_name}")

    # NEVER destructively replace an experimental vector collection during evaluation.
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        raise FileExistsError("Existing research vector collection is protected: " + collection_name)

    model = SentenceTransformer(str(model_path), device="cpu")

    chunks = list(KnowledgeChunk.objects.filter(
        is_quarantined=False,
        document__is_rag_enabled=True,
        document__index_status="INDEXED",
    ).exclude(text="").select_related(
        "document", "document__asset", "document__product",
        "document__category", "document__domain"
    ).order_by("id"))

    passages = [op27_passage_text(c, content_first=content_first) for c in chunks]

    embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

    fp = cached_checkpoint_fingerprint(str(model_path))
    meta = {
        "hnsw:space": "cosine",
        "servy_model": str(model_path),
        "servy_revision": "",
        "servy_dim": 384,
        "servy_query_prefix": query_prefix,
        "servy_passage_format": PASSAGE_FORMAT_VERSION if content_first else "legacy_metadata_v0",
        "servy_schema_version": INDEX_SCHEMA_VERSION,
        "servy_checkpoint_fp": fp,
    }

    coll = client.create_collection(name=collection_name, metadata=meta)
    batch_sz = 100
    for i in range(0, len(chunks), batch_sz):
        bc = chunks[i:i+batch_sz]
        be = embeddings[i:i+batch_sz].tolist()
        coll.add(
            ids=[str(c.id) for c in bc],
            embeddings=be,
            metadatas=[{"chunk_id": c.id, "document_id": c.document_id,
                        "tenant_id": c.tenant_id, "chunk_index": c.chunk_index,
                        "heading": c.heading or ""} for c in bc],
            documents=[passages[i+j] for j in range(len(bc))],
        )
    print(f"Indexed {coll.count()} chunks in {collection_name}")
    return coll


def evaluate_dense(model_path, collection_name, query_prefix, relevance_rows):
    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    coll = client.get_collection(collection_name)
    model = SentenceTransformer(str(model_path), device="cpu")

    r1 = r3 = r5 = r10 = r20 = mrr = ndcg = 0.0
    for row in relevance_rows:
        query = query_prefix + row["query_text"]
        tid = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))

        emb = model.encode([query], normalize_embeddings=True)[0].tolist()
        results = coll.query(
            query_embeddings=[emb], n_results=20,
            where={"tenant_id": tid}, include=["metadatas"],
        )
        ranked = [m["chunk_id"] for m in results["metadatas"][0]] if results["metadatas"][0] else []
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]

        r1 += any(h < 1 for h in hits)
        r3 += any(h < 3 for h in hits)
        r5 += any(h < 5 for h in hits)
        r10 += any(h < 10 for h in hits)
        r20 += any(h < 20 for h in hits)
        rr = 1 / (hits[0] + 1) if hits else 0
        mrr += rr
        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), 10)))
        ndcg += dcg / idcg if idcg else 0

    n = len(relevance_rows)
    return {
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
    }


def evaluate_hybrid(model_path, collection_name, query_prefix, relevance_rows, profile_key):
    # Register profile
    PROFILES[profile_key] = EmbeddingProfile(
        key=profile_key,
        model_id=str(model_path),
        revision=None,
        dim=384,
        collection=collection_name,
        local_path=str(model_path),
        query_prefix=query_prefix,
        content_first=False,
        promotion_eligible=True,
    )

    orig_dense = retriever_module._dense_score_candidates
    def patched_dense(*args, **kwargs):
        import core.ml.embeddings as emb_mod
        orig_gp = emb_mod.get_profile
        def mock_gp(key):
            if key in ("company-minilm", "minilm", "company-bge-small", "bge-small", "company-bge-ft", profile_key):
                return orig_gp(profile_key)
            return orig_gp(key)
        emb_mod.get_profile = mock_gp
        try:
            return orig_dense(*args, **kwargs)
        finally:
            emb_mod.get_profile = orig_gp

    retriever_module._dense_score_candidates = patched_dense

    tenants = {t.id: t for t in Tenant.objects.all()}
    r1 = r3 = r5 = r10 = r20 = mrr = ndcg = 0.0

    try:
        for row in relevance_rows:
            query = row["query_text"]
            tid = int(row["tenant_id"])
            pos_ids = set(row.get("positive_chunk_ids", []))
            equip = row.get("equipment", "")

            tenant = tenants[tid]
            asset = resolve_query_asset(tid, equip)

            results = retrieve(
                tenant=tenant, question=query, asset=asset, customer=None,
                include_confidential=False, top_k=20, staff_mode=False,
            )
            ranked = [r.chunk_id for r in results if r.chunk_id]
            hits = [i for i, c in enumerate(ranked) if c in pos_ids]

            r1 += any(h < 1 for h in hits)
            r3 += any(h < 3 for h in hits)
            r5 += any(h < 5 for h in hits)
            r10 += any(h < 10 for h in hits)
            r20 += any(h < 20 for h in hits)
            rr = 1 / (hits[0] + 1) if hits else 0
            mrr += rr
            dcg = sum(1 / math.log2(h + 2) for h in hits)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), 10)))
            ndcg += dcg / idcg if idcg else 0
    finally:
        retriever_module._dense_score_candidates = orig_dense

    n = len(relevance_rows)
    return {
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
    }


def main():
    print("=" * 80)
    print("OPERATION 27 PHASE 4 — TRAINING & EVALUATION OF OP27 REPAIRED MODELS")
    print("=" * 80)

    train_rows = load_dataset("train")
    dev_rows = load_dataset("dev")
    _, relevance_rows, _ = load_holdout()

    print(f"Loaded {len(train_rows)} train rows, {len(dev_rows)} dev rows, {len(relevance_rows)} relevance test rows")

    # The Op27 Chroma collections are the legacy metadata-rich passage profile.
    train_resolved = resolve_chunk_texts(train_rows, content_first=False)
    val_resolved = resolve_chunk_texts(dev_rows, content_first=False)

    integrity_report = enforce_train_dev_separation(train_resolved, val_resolved)
    print("Train/dev structural isolation:", integrity_report)
    # Operation 27's labels are source-derived, not authenticated expert truth.
    # Training must be deliberately opted into; production promotion stays disabled.
    if os.getenv("SERVY_OP27_WEAK_RESEARCH_ACK") != "acknowledge":
        raise RuntimeError(
            "TRAINING_BLOCKED: Op27 uses weak supervision. "
            "Set SERVY_OP27_WEAK_RESEARCH_ACK=acknowledge for an isolated "
            "research run only after inspecting the integrity report."
        )

    all_eval_results = {}

    # === MODEL 1: Op27 MiniLM ===
    minilm_out = BASE_DIR / "ml_models" / "checkpoints" / "op27_minilm_ft"
    minilm_ckpt, minilm_hash = train_single_model(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        query_prefix="",
        output_dir=minilm_out,
        train_resolved=train_resolved,
        val_resolved=val_resolved,
        epochs=3, batch_size=16, lr=2e-5,
    )
    build_and_stamp_chroma_collection(
        model_path=minilm_ckpt,
        collection_name="servy_op27_minilm_ft_v1",
        query_prefix="",
        content_first=False,
    )
    print("\nEvaluating Op27 MiniLM Pure Dense...")
    all_eval_results["op27_minilm_dense"] = evaluate_dense(
        model_path=minilm_ckpt,
        collection_name="servy_op27_minilm_ft_v1",
        query_prefix="",
        relevance_rows=relevance_rows,
    )
    print(f"Op27 MiniLM Dense: R@1={all_eval_results['op27_minilm_dense']['recall@1']:.4f}, R@5={all_eval_results['op27_minilm_dense']['recall@5']:.4f}")

    print("\nEvaluating Op27 MiniLM Hybrid...")
    all_eval_results["op27_minilm_hybrid"] = evaluate_hybrid(
        model_path=minilm_ckpt,
        collection_name="servy_op27_minilm_ft_v1",
        query_prefix="",
        relevance_rows=relevance_rows,
        profile_key="op27-minilm-ft",
    )
    print(f"Op27 MiniLM Hybrid: R@1={all_eval_results['op27_minilm_hybrid']['recall@1']:.4f}, R@5={all_eval_results['op27_minilm_hybrid']['recall@5']:.4f}")

    # === MODEL 2: Op27 BGE-small ===
    bge_out = BASE_DIR / "ml_models" / "checkpoints" / "op27_bge_ft"
    bge_ckpt, bge_hash = train_single_model(
        model_id="BAAI/bge-small-en-v1.5",
        query_prefix=BGE_QUERY_PREFIX,
        output_dir=bge_out,
        train_resolved=train_resolved,
        val_resolved=val_resolved,
        epochs=3, batch_size=16, lr=2e-5,
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    )
    build_and_stamp_chroma_collection(
        model_path=bge_ckpt,
        collection_name="servy_op27_bge_ft_v1",
        query_prefix=BGE_QUERY_PREFIX,
        content_first=False,
    )
    print("\nEvaluating Op27 BGE-small Pure Dense...")
    all_eval_results["op27_bge_dense"] = evaluate_dense(
        model_path=bge_ckpt,
        collection_name="servy_op27_bge_ft_v1",
        query_prefix=BGE_QUERY_PREFIX,
        relevance_rows=relevance_rows,
    )
    print(f"Op27 BGE Dense: R@1={all_eval_results['op27_bge_dense']['recall@1']:.4f}, R@5={all_eval_results['op27_bge_dense']['recall@5']:.4f}")

    print("\nEvaluating Op27 BGE-small Hybrid...")
    all_eval_results["op27_bge_hybrid"] = evaluate_hybrid(
        model_path=bge_ckpt,
        collection_name="servy_op27_bge_ft_v1",
        query_prefix=BGE_QUERY_PREFIX,
        relevance_rows=relevance_rows,
        profile_key="op27-bge-ft",
    )
    print(f"Op27 BGE Hybrid: R@1={all_eval_results['op27_bge_hybrid']['recall@1']:.4f}, R@5={all_eval_results['op27_bge_hybrid']['recall@5']:.4f}")

    # Summary table
    print("\n" + "=" * 90)
    print("OP27 TRAINED MODELS EVALUATION SUMMARY (relevance denominator N=43)")
    print("=" * 90)
    header = f"{'Configuration':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>8} {'nDCG@10':>8}"
    print(header)
    print("-" * len(header))
    for k, m in all_eval_results.items():
        print(f"{k:<30} {m['recall@1']:>6.4f} {m['recall@3']:>6.4f} {m['recall@5']:>6.4f} {m['recall@10']:>6.4f} {m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f}")

    # Save results
    out_file = BASE_DIR / "private_training_data" / "op27_phase4_trained_models_evaluation.json"
    out_file.write_text(json.dumps(all_eval_results, indent=2), encoding="utf-8")
    print(f"\nSaved Op27 evaluation to: {out_file}")


if __name__ == "__main__":
    main()
