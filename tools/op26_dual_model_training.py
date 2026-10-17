"""Operation 26 â€” Dual-Model Training Runner.

Fine-tunes BOTH models sequentially using the Op26 training dataset,
then evaluates all configurations (base + finetuned) against the frozen holdout.

Model A: sentence-transformers/all-MiniLM-L6-v2  (no query prefix)
Model B: BAAI/bge-small-en-v1.5                  (BGE query instruction prefix)

Each model gets its own:
  - Checkpoint directory
  - Chroma collection
  - Evaluation track
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
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("SERVY_DB_ENGINE", "mysql")

import django
django.setup()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ MODEL CONFIGURATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODEL_CONFIGS = {
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": None,
        "dim": 384,
        "query_prefix": "",  # MiniLM uses NO prefix
        "content_first": False,
        "collection_name": "servy_op26_minilm_ft_v1",
        "checkpoint_dir": "ml_models/checkpoints/op26_minilm_ft",
        "profile_key": "op26-minilm-ft",
    },
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "dim": 384,
        "query_prefix": BGE_QUERY_PREFIX,
        "content_first": True,
        "collection_name": "servy_op26_bge_ft_v1",
        "checkpoint_dir": "ml_models/checkpoints/op26_bge_ft",
        "profile_key": "op26-bge-ft",
    },
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ DATA LOADING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_op26_dataset(split: str):
    """Load Op26 train or dev split from JSONL."""
    path = BASE_DIR / "private_training_data" / f"op26_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_holdout():
    """Load frozen independent holdout benchmark."""
    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Holdout not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_chunk_texts(rows):
    """Resolve positive and negative chunk IDs to passage text."""
    from core.models import KnowledgeChunk

    all_chunk_ids = set()
    for r in rows:
        for cid in r.get("positive_chunk_ids", []):
            all_chunk_ids.add(cid)
        for cid in r.get("negative_chunk_ids", []):
            all_chunk_ids.add(cid)

    chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=all_chunk_ids).select_related(
        "document", "document__asset", "document__product",
        "document__category", "document__domain"
    )}

    resolved = []
    skipped = 0
    for r in rows:
        pos_ids = r.get("positive_chunk_ids", [])
        neg_ids = r.get("negative_chunk_ids", [])

        pos_texts = []
        for cid in pos_ids:
            c = chunks.get(cid)
            if c:
                heading = (c.heading or "").strip()
                body = c.text or ""
                pos_texts.append(f"{heading}\n{body}".strip() if heading and heading not in body else body)

        neg_texts = []
        for cid in neg_ids:
            c = chunks.get(cid)
            if c:
                heading = (c.heading or "").strip()
                body = c.text or ""
                neg_texts.append(f"{heading}\n{body}".strip() if heading and heading not in body else body)

        if not pos_texts:
            skipped += 1
            continue

        resolved_row = dict(r)
        resolved_row["positive_text"] = pos_texts[0]
        resolved_row["negative_text"] = neg_texts[0] if neg_texts else ""
        resolved_row["query_text"] = r["query_text"]
        resolved_row["query_id"] = r.get("query_id", "")
        resolved_row["tenant_id"] = r.get("tenant_id", 0)
        resolved_row["review_status"] = r.get("review_status", "weakly_supervised")
        resolved.append(resolved_row)

    if skipped:
        print(f"  WARNING: Skipped {skipped} rows with unresolvable chunk IDs")

    return resolved


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ TRAINING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_model(model_name: str, train_resolved, val_resolved,
                epochs=3, batch_size=16, lr=2e-5, max_seq_length=256, seed=42):
    """Fine-tune a single model using MultipleNegativesRankingLoss."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

    config = MODEL_CONFIGS[model_name]
    out_dir = BASE_DIR / config["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {model_name.upper()}")
    print(f"Model ID: {config['model_id']}")
    print(f"Query prefix: {'<empty>' if not config['query_prefix'] else repr(config['query_prefix'][:40])}")
    print(f"Output: {out_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Train examples: {len(train_resolved)}, Val examples: {len(val_resolved)}")
    print(f"{'='*80}")

    torch.manual_seed(seed)

    # Load base model
    kwargs = {"device": "cpu"}
    if config["revision"]:
        kwargs["revision"] = config["revision"]
    model = SentenceTransformer(config["model_id"], **kwargs)
    model.max_seq_length = max_seq_length

    prefix = config["query_prefix"]  # MODEL-SPECIFIC prefix

    # Build datasets â€” key difference: query prefix is model-specific
    triplets_with_neg = [r for r in train_resolved if r.get("negative_text")]
    triplets_without_neg = [r for r in train_resolved if not r.get("negative_text")]

    print(f"  With hard negatives: {len(triplets_with_neg)}")
    print(f"  Without hard negatives (in-batch only): {len(triplets_without_neg)}")

    def build_triplet_ds(rows):
        has_neg = [r for r in rows if r.get("negative_text")]
        if not has_neg:
            return None
        return Dataset.from_dict({
            "anchor": [prefix + r["query_text"] for r in has_neg],
            "positive": [r["positive_text"] for r in has_neg],
            "negative": [r["negative_text"] for r in has_neg],
        })

    def build_pair_ds(rows):
        no_neg = [r for r in rows if not r.get("negative_text")]
        if not no_neg:
            return None
        return Dataset.from_dict({
            "anchor": [prefix + r["query_text"] for r in no_neg],
            "positive": [r["positive_text"] for r in no_neg],
        })

    # Build multi-dataset training (mixed mode)
    ds_triplets = build_triplet_ds(train_resolved)
    ds_pairs = build_pair_ds(train_resolved)

    if ds_triplets and ds_pairs:
        train_dataset = {"triplets": ds_triplets, "pairs": ds_pairs}
    elif ds_triplets:
        train_dataset = ds_triplets
    else:
        # All pairs only
        train_dataset = Dataset.from_dict({
            "anchor": [prefix + r["query_text"] for r in train_resolved],
            "positive": [r["positive_text"] for r in train_resolved],
        })

    # Eval dataset
    eval_dataset = None
    if val_resolved:
        eval_ds_triplets = build_triplet_ds(val_resolved)
        eval_ds_pairs = build_pair_ds(val_resolved)
        if eval_ds_triplets and eval_ds_pairs:
            eval_dataset = {"triplets": eval_ds_triplets, "pairs": eval_ds_pairs}
        elif eval_ds_triplets:
            eval_dataset = eval_ds_triplets
        else:
            eval_dataset = Dataset.from_dict({
                "anchor": [prefix + r["query_text"] for r in val_resolved],
                "positive": [r["positive_text"] for r in val_resolved],
            })

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        seed=seed,
        report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch" if val_resolved else "no",
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
    result = trainer.train()
    elapsed = time.time() - t0

    # Save final checkpoint
    final_dir = out_dir / "final"
    model.save(str(final_dir))

    # Compute deterministic checkpoint hash
    h = hashlib.sha256()
    for p in sorted(final_dir.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    ckpt_hash = h.hexdigest()

    # Reload verification
    reloaded = SentenceTransformer(str(final_dir), device="cpu")
    vec = reloaded.encode(["reload verification check"], normalize_embeddings=True)

    report = {
        "model_name": model_name,
        "model_id": config["model_id"],
        "query_prefix": config["query_prefix"],
        "training_completed": True,
        "reload_succeeded": True,
        "reload_dim": int(vec.shape[1]),
        "optimizer_steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_train": len(train_resolved),
        "n_with_negatives": len(triplets_with_neg),
        "n_without_negatives": len(triplets_without_neg),
        "n_val": len(val_resolved),
        "seed": seed,
        "checkpoint": str(final_dir),
        "checkpoint_sha256": ckpt_hash,
        "seconds": round(elapsed, 1),
    }

    report_path = out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  âœ“ Training complete: {elapsed:.1f}s")
    print(f"  âœ“ Loss: {result.training_loss:.6f}")
    print(f"  âœ“ Steps: {result.global_step}")
    print(f"  âœ“ Checkpoint: {final_dir}")
    print(f"  âœ“ SHA256: {ckpt_hash[:16]}...")

    return report


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ INDEXING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_experimental_index(model_name: str, checkpoint_path: str):
    """Index all authorized chunks into an experimental Chroma collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from core.models import KnowledgeChunk

    config = MODEL_CONFIGS[model_name]
    collection_name = config["collection_name"]
    prefix = config["query_prefix"]

    print(f"\n{'='*80}")
    print(f"BUILDING EXPERIMENTAL INDEX: {model_name.upper()}")
    print(f"Collection: {collection_name}")
    print(f"{'='*80}")

    # Load fine-tuned model
    model = SentenceTransformer(checkpoint_path, device="cpu")
    print(f"  Model loaded from: {checkpoint_path}")
    print(f"  Dimensions: {model.get_sentence_embedding_dimension()}")

    # Get all eligible chunks
    chunks = list(KnowledgeChunk.objects.filter(
        is_quarantined=False,
        document__is_rag_enabled=True,
        document__index_status="INDEXED",
    ).exclude(text="").select_related(
        "document", "document__asset", "document__product",
        "document__category", "document__domain"
    ).order_by("id"))

    print(f"  Chunks to index: {len(chunks)}")

    # Prepare passage texts
    if config["content_first"]:
        # BGE-style: content only (no metadata preamble)
        passages = []
        for c in chunks:
            heading = (c.heading or "").strip()
            body = c.text or ""
            text = f"{heading}\n{body}".strip() if heading and heading not in body else body
            passages.append(text)
    else:
        # MiniLM-style: full passage with metadata
        passages = []
        for c in chunks:
            d = c.document
            parts = filter(None, [
                d.title, d.description, d.tags, d.doc_type,
                getattr(d.domain, 'name', '') if d.domain else "",
                getattr(d.category, 'name', '') if d.category else "",
                getattr(d.product, 'name', '') if d.product else "",
                getattr(d.asset, 'name', '') if d.asset else "",
                getattr(d.asset, 'model_number', '') if d.asset else "",
            ])
            prefix_text = " ".join(parts)
            heading = (c.heading or "").strip()
            text = f"{prefix_text} {heading}\n{c.text}".strip()
            passages.append(text)

    # Encode passages (no query prefix for passages â€” passages never get prefix)
    print(f"  Encoding {len(passages)} passages...")
    t0 = time.time()
    embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True,
                              batch_size=32)
    encode_time = time.time() - t0
    print(f"  Encoding complete: {encode_time:.1f}s ({len(passages)/encode_time:.1f} passages/s)")

    # Create/replace Chroma collection
    chroma_dir = str(BASE_DIR / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete if exists
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")
    except Exception:
        pass

    coll = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size].tolist()
        batch_ids = [str(c.id) for c in batch_chunks]
        batch_metadatas = [{
            "chunk_id": c.id,
            "document_id": c.document_id,
            "tenant_id": c.tenant_id,
            "chunk_index": c.chunk_index,
            "heading": c.heading or "",
        } for c in batch_chunks]
        batch_docs = [passages[i+j] for j in range(len(batch_chunks))]

        coll.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_docs,
        )

    print(f"  âœ“ Indexed {coll.count()} chunks into {collection_name}")
    return {"collection": collection_name, "count": coll.count(), "encode_seconds": round(encode_time, 1)}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ EVALUATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate_model_on_holdout(model_name: str, checkpoint_path: str, holdout_rows, k_max=10):
    """Evaluate a fine-tuned model against the frozen holdout benchmark."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np

    config = MODEL_CONFIGS[model_name]
    collection_name = config["collection_name"]
    prefix = config["query_prefix"]

    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name.upper()} (fine-tuned)")
    print(f"Collection: {collection_name}")
    print(f"Holdout queries: {len(holdout_rows)}")
    print(f"{'='*80}")

    model = SentenceTransformer(checkpoint_path, device="cpu")

    chroma_dir = str(BASE_DIR / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    coll = client.get_collection(collection_name)

    r1 = r3 = r5 = r10 = mrr = ndcg = 0.0
    latencies = []
    per_query_results = []

    for row in holdout_rows:
        query = row["query_text"]
        tenant_id = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))

        # Apply model-specific prefix to query
        prefixed_query = prefix + query

        t0 = time.perf_counter()
        emb = model.encode([prefixed_query], normalize_embeddings=True)[0].tolist()
        results = coll.query(
            query_embeddings=[emb],
            n_results=k_max,
            where={"tenant_id": tenant_id},
            include=["metadatas"],
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        ranked = [m["chunk_id"] for m in results["metadatas"][0]]
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]

        hit1 = any(h < 1 for h in hits)
        hit3 = any(h < 3 for h in hits)
        hit5 = any(h < 5 for h in hits)
        hit10 = bool(hits)

        r1 += hit1
        r3 += hit3
        r5 += hit5
        r10 += hit10
        rr = 1 / (hits[0] + 1) if hits else 0
        mrr += rr

        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), k_max)))
        ndcg += dcg / idcg if idcg else 0

        per_query_results.append({
            "query_id": row.get("query_id", ""),
            "query_text": query,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "hit@10": hit10,
            "rr": rr,
            "ranked": ranked[:5],
            "positive_ids": list(pos_ids),
        })

    n = max(len(holdout_rows), 1)
    latencies.sort()

    metrics = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "collection": collection_name,
        "n_queries": len(holdout_rows),
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "latency_p50_ms": round(latencies[len(latencies)//2], 1) if latencies else None,
        "latency_p95_ms": round(latencies[int(len(latencies)*0.95)] if latencies else 0, 1),
    }

    print(f"\n  Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.4f}")
    print(f"  Recall@3:  {metrics['recall@3']:.4f}")
    print(f"  Recall@5:  {metrics['recall@5']:.4f}")
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  MRR@10:    {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")
    print(f"  Latency p50: {metrics['latency_p50_ms']}ms")

    return metrics, per_query_results


def evaluate_base_model(model_name: str, holdout_rows, k_max=10):
    """Evaluate a BASE (not fine-tuned) model against holdout using its own collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from core.models import KnowledgeChunk

    config = MODEL_CONFIGS[model_name]
    prefix = config["query_prefix"]

    # Use a temporary experimental collection for the base model
    base_collection_name = f"servy_op26_{model_name.replace('-', '_')}_base_v1"

    print(f"\n{'='*80}")
    print(f"EVALUATING BASE MODEL: {model_name.upper()}")
    print(f"Model ID: {config['model_id']}")
    print(f"Collection: {base_collection_name}")
    print(f"{'='*80}")

    # Load base model
    kwargs = {"device": "cpu"}
    if config["revision"]:
        kwargs["revision"] = config["revision"]
    model = SentenceTransformer(config["model_id"], **kwargs)

    # Index chunks into base collection
    chunks = list(KnowledgeChunk.objects.filter(
        is_quarantined=False,
        document__is_rag_enabled=True,
        document__index_status="INDEXED",
    ).exclude(text="").select_related(
        "document", "document__asset", "document__product",
        "document__category", "document__domain"
    ).order_by("id"))

    # Build passages
    if config["content_first"]:
        passages = []
        for c in chunks:
            heading = (c.heading or "").strip()
            body = c.text or ""
            text = f"{heading}\n{body}".strip() if heading and heading not in body else body
            passages.append(text)
    else:
        passages = []
        for c in chunks:
            d = c.document
            parts = filter(None, [
                d.title, d.description, d.tags, d.doc_type,
                getattr(d.domain, 'name', '') if d.domain else "",
                getattr(d.category, 'name', '') if d.category else "",
                getattr(d.product, 'name', '') if d.product else "",
                getattr(d.asset, 'name', '') if d.asset else "",
                getattr(d.asset, 'model_number', '') if d.asset else "",
            ])
            prefix_text = " ".join(parts)
            heading = (c.heading or "").strip()
            text = f"{prefix_text} {heading}\n{c.text}".strip()
            passages.append(text)

    # Encode and index
    print(f"  Encoding {len(passages)} passages for base model...")
    embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    chroma_dir = str(BASE_DIR / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        client.delete_collection(base_collection_name)
    except Exception:
        pass

    coll = client.create_collection(
        name=base_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size].tolist()
        batch_ids = [str(c.id) for c in batch_chunks]
        batch_metadatas = [{
            "chunk_id": c.id,
            "document_id": c.document_id,
            "tenant_id": c.tenant_id,
            "chunk_index": c.chunk_index,
            "heading": c.heading or "",
        } for c in batch_chunks]
        batch_docs = [passages[i+j] for j in range(len(batch_chunks))]
        coll.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas, documents=batch_docs)

    print(f"  âœ“ Indexed {coll.count()} chunks into {base_collection_name}")

    # Now evaluate
    r1 = r3 = r5 = r10 = mrr = ndcg = 0.0
    latencies = []

    for row in holdout_rows:
        query = row["query_text"]
        tenant_id = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))

        prefixed_query = prefix + query

        t0 = time.perf_counter()
        emb = model.encode([prefixed_query], normalize_embeddings=True)[0].tolist()
        results = coll.query(
            query_embeddings=[emb],
            n_results=k_max,
            where={"tenant_id": tenant_id},
            include=["metadatas"],
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        ranked = [m["chunk_id"] for m in results["metadatas"][0]]
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]

        r1 += any(h < 1 for h in hits)
        r3 += any(h < 3 for h in hits)
        r5 += any(h < 5 for h in hits)
        r10 += bool(hits)
        mrr += 1 / (hits[0] + 1) if hits else 0

        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), k_max)))
        ndcg += dcg / idcg if idcg else 0

    n = max(len(holdout_rows), 1)
    latencies.sort()

    metrics = {
        "model": f"{model_name}-base",
        "model_id": config["model_id"],
        "collection": base_collection_name,
        "n_queries": len(holdout_rows),
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "latency_p50_ms": round(latencies[len(latencies)//2], 1) if latencies else None,
        "latency_p95_ms": round(latencies[int(len(latencies)*0.95)] if latencies else 0, 1),
    }

    print(f"\n  Base Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.4f}")
    print(f"  Recall@3:  {metrics['recall@3']:.4f}")
    print(f"  Recall@5:  {metrics['recall@5']:.4f}")
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  MRR@10:    {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")

    return metrics


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ MAIN RUNNER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print("=" * 80)
    print("OPERATION 26 â€” DUAL-MODEL DEEP ACCURACY TRAINING")
    print("=" * 80)

    # Phase 1: Load and resolve datasets
    print("\n[PHASE 1] Loading datasets...")
    train_rows = load_op26_dataset("train")
    dev_rows = load_op26_dataset("dev")
    holdout_rows = load_holdout()

    print(f"  Train: {len(train_rows)} rows")
    print(f"  Dev:   {len(dev_rows)} rows")
    print(f"  Holdout: {len(holdout_rows)} rows")

    print("\n[PHASE 1b] Resolving chunk texts...")
    train_resolved = resolve_chunk_texts(train_rows)
    dev_resolved = resolve_chunk_texts(dev_rows)

    print(f"  Resolved train: {len(train_resolved)}")
    print(f"  Resolved dev:   {len(dev_resolved)}")

    # Phase 2: Evaluate BASE models
    print("\n[PHASE 2] Evaluating BASE models on holdout...")
    base_results = {}
    for model_name in ["minilm", "bge-small"]:
        base_results[model_name] = evaluate_base_model(model_name, holdout_rows)

    # Phase 3: Train both models
    print("\n[PHASE 3] Training Model A (MiniLM)...")
    minilm_report = train_model(
        "minilm", train_resolved, dev_resolved,
        epochs=4, batch_size=16, lr=3e-5, seed=42
    )

    print("\n[PHASE 3] Training Model B (BGE-small)...")
    bge_report = train_model(
        "bge-small", train_resolved, dev_resolved,
        epochs=4, batch_size=16, lr=2e-5, seed=42
    )

    # Phase 4: Build experimental indexes
    print("\n[PHASE 4] Building experimental indexes...")
    minilm_index = build_experimental_index("minilm", minilm_report["checkpoint"])
    bge_index = build_experimental_index("bge-small", bge_report["checkpoint"])

    # Phase 5: Evaluate fine-tuned models on holdout
    print("\n[PHASE 5] Evaluating fine-tuned models on holdout...")
    minilm_ft_metrics, minilm_per_query = evaluate_model_on_holdout(
        "minilm", minilm_report["checkpoint"], holdout_rows
    )
    bge_ft_metrics, bge_per_query = evaluate_model_on_holdout(
        "bge-small", bge_report["checkpoint"], holdout_rows
    )

    # Phase 6: Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON â€” ALL CONFIGURATIONS")
    print("=" * 80)

    all_results = {
        "minilm_base": base_results["minilm"],
        "bge_small_base": base_results["bge-small"],
        "minilm_finetuned": minilm_ft_metrics,
        "bge_small_finetuned": bge_ft_metrics,
    }

    header = f"{'Configuration':<25} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'nDCG@10':>8}"
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        print(f"{name:<25} {m['recall@1']:>8.4f} {m.get('recall@3', 0):>8.4f} {m['recall@5']:>8.4f} {m['recall@10']:>8.4f} {m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f}")

    # Determine winner
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["mrr@10"])
    best = all_results[best_name]
    print(f"\nâœ“ BEST MODEL: {best_name} (MRR@10 = {best['mrr@10']:.4f})")

    # Check if any fine-tuned model improved over its base
    minilm_delta = minilm_ft_metrics["mrr@10"] - base_results["minilm"]["mrr@10"]
    bge_delta = bge_ft_metrics["mrr@10"] - base_results["bge-small"]["mrr@10"]

    print(f"\n  MiniLM fine-tuning delta: {minilm_delta:+.4f} MRR@10")
    print(f"  BGE-small fine-tuning delta: {bge_delta:+.4f} MRR@10")

    # Save comprehensive results
    final_report = {
        "operation": "OPERATION_26",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "train": len(train_resolved),
            "dev": len(dev_resolved),
            "holdout": len(holdout_rows),
        },
        "training": {
            "minilm": minilm_report,
            "bge_small": bge_report,
        },
        "indexing": {
            "minilm": minilm_index,
            "bge_small": bge_index,
        },
        "evaluation": all_results,
        "deltas": {
            "minilm_ft_vs_base": minilm_delta,
            "bge_small_ft_vs_base": bge_delta,
        },
        "winner": best_name,
        "per_query": {
            "minilm_ft": minilm_per_query,
            "bge_small_ft": bge_per_query,
        },
    }

    report_path = BASE_DIR / "private_training_data" / "op26_final_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nFull report saved: {report_path}")

    return final_report


if __name__ == "__main__":
    main()

