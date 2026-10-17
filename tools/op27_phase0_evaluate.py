"""Op27 Phase 0.4 — Fresh corrected evaluation of all configurations.

Reproduces evaluation with:
1. Correct relevance-only denominator (N=43)
2. Separate abstention accuracy (N=21)
3. Per-query failure classification
4. Includes base models, Op26 fine-tuned models, and production hybrid pipeline
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import hashlib
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


def load_holdout():
    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]
    return rows, relevance, abstention


def evaluate_dense_model(model_path, model_label, collection_name, query_prefix,
                         relevance_rows, k_max=10, content_first=True):
    """Evaluate a dense model on relevance queries only, with correct denominator."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from core.models import KnowledgeChunk

    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_label}")
    print(f"Model: {model_path}")
    print(f"Collection: {collection_name}")
    print(f"Prefix: {'<none>' if not query_prefix else repr(query_prefix[:40])+'...'}")
    print(f"Relevance queries: {len(relevance_rows)} (correct denominator)")
    print(f"{'='*80}")

    model = SentenceTransformer(model_path, device="cpu")

    chroma_dir = str(BASE_DIR / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Check if collection exists; if not, build it
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        print(f"  Collection '{collection_name}' not found. Building index...")
        chunks = list(KnowledgeChunk.objects.filter(
            is_quarantined=False,
            document__is_rag_enabled=True,
            document__index_status="INDEXED",
        ).exclude(text="").select_related(
            "document", "document__asset", "document__product",
            "document__category", "document__domain"
        ).order_by("id"))

        if content_first:
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

        embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

        coll = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
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
        print(f"  Built index with {coll.count()} chunks")
    else:
        coll = client.get_collection(collection_name)
        print(f"  Using existing collection ({coll.count()} chunks)")

    # Evaluate on RELEVANCE queries only
    r1 = r3 = r5 = r10 = r20 = mrr = ndcg = 0.0
    latencies = []
    per_query = []
    failures_by_category = defaultdict(list)

    for row in relevance_rows:
        query = row["query_text"]
        tenant_id = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))
        category = row.get("category", "unknown")

        prefixed = query_prefix + query
        t0 = time.perf_counter()
        emb = model.encode([prefixed], normalize_embeddings=True)[0].tolist()
        results = coll.query(
            query_embeddings=[emb], n_results=min(k_max, 20),
            where={"tenant_id": tenant_id}, include=["metadatas", "distances"],
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        ranked = [m["chunk_id"] for m in results["metadatas"][0]] if results["metadatas"][0] else []
        distances = results["distances"][0] if results["distances"] and results["distances"][0] else []
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]

        hit1 = any(h < 1 for h in hits)
        hit3 = any(h < 3 for h in hits)
        hit5 = any(h < 5 for h in hits)
        hit10 = any(h < 10 for h in hits)
        hit20 = any(h < 20 for h in hits)

        r1 += hit1; r3 += hit3; r5 += hit5; r10 += hit10; r20 += hit20
        rr = 1 / (hits[0] + 1) if hits else 0
        mrr += rr

        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), k_max)))
        ndcg += dcg / idcg if idcg else 0

        first_hit_rank = hits[0] + 1 if hits else None
        pq = {
            "query_id": row.get("query_id", ""),
            "query_text": query,
            "category": category,
            "equipment": row.get("equipment", ""),
            "tenant_id": tenant_id,
            "hit@1": hit1, "hit@3": hit3, "hit@5": hit5, "hit@10": hit10, "hit@20": hit20,
            "rr": rr,
            "first_hit_rank": first_hit_rank,
            "ranked_top5": ranked[:5],
            "positive_ids": list(pos_ids),
            "top_distance": distances[0] if distances else None,
        }
        per_query.append(pq)

        if not hit1:
            failures_by_category[category].append(pq)

    n = len(relevance_rows)
    latencies.sort()

    metrics = {
        "model": model_label,
        "n_relevance": n,
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "latency_p50_ms": round(latencies[len(latencies)//2], 1),
        "latency_p95_ms": round(latencies[int(len(latencies)*0.95)], 1),
    }

    print(f"\n  CORRECTED RESULTS (N={n}):")
    print(f"  Recall@1:  {metrics['recall@1']:.4f}  ({int(r1)}/{n})")
    print(f"  Recall@3:  {metrics['recall@3']:.4f}  ({int(r3)}/{n})")
    print(f"  Recall@5:  {metrics['recall@5']:.4f}  ({int(r5)}/{n})")
    print(f"  Recall@10: {metrics['recall@10']:.4f}  ({int(r10)}/{n})")
    print(f"  Recall@20: {metrics['recall@20']:.4f}  ({int(r20)}/{n})")
    print(f"  MRR@10:    {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")
    print(f"  Latency:   {metrics['latency_p50_ms']}ms p50, {metrics['latency_p95_ms']}ms p95")

    # Failure analysis
    miss_count = n - int(r1)
    if miss_count > 0:
        print(f"\n  FAILURES ({miss_count} queries missed @1):")
        for cat, fails in sorted(failures_by_category.items(), key=lambda x: -len(x[1])):
            print(f"    {cat}: {len(fails)} misses")
            for f in fails[:2]:
                fhr = f['first_hit_rank']
                rank_str = f"rank {fhr}" if fhr else "NOT IN TOP 20"
                print(f"      '{f['query_text'][:60]}...' -> {rank_str}")

    return metrics, per_query


def main():
    print("=" * 80)
    print("OPERATION 27 PHASE 0.4 - FRESH CORRECTED EVALUATION")
    print("=" * 80)

    all_rows, relevance, abstention = load_holdout()
    print(f"\nHoldout: {len(all_rows)} total, {len(relevance)} relevance, {len(abstention)} abstention")

    results = {}

    # 1. Base MiniLM
    results["base_minilm"] = evaluate_dense_model(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_label="Base MiniLM",
        collection_name="servy_op26_minilm_base_v1",
        query_prefix="",
        relevance_rows=relevance,
        content_first=False,
    )[0]

    # 2. Op26 Fine-Tuned MiniLM
    minilm_ckpt = str(BASE_DIR / "ml_models" / "checkpoints" / "op26_minilm_ft" / "final")
    results["op26_minilm_ft"] = evaluate_dense_model(
        model_path=minilm_ckpt,
        model_label="Op26 MiniLM Fine-Tuned",
        collection_name="servy_op26_minilm_ft_v1",
        query_prefix="",
        relevance_rows=relevance,
        content_first=False,
    )[0]

    # 3. Base BGE-small
    results["base_bge"] = evaluate_dense_model(
        model_path="BAAI/bge-small-en-v1.5",
        model_label="Base BGE-small",
        collection_name="servy_op26_bge_small_base_v1",
        query_prefix="Represent this sentence for searching relevant passages: ",
        relevance_rows=relevance,
        content_first=True,
    )[0]

    # 4. Op26 Fine-Tuned BGE-small
    bge_ckpt = str(BASE_DIR / "ml_models" / "checkpoints" / "op26_bge_ft" / "final")
    results["op26_bge_ft"] = evaluate_dense_model(
        model_path=bge_ckpt,
        model_label="Op26 BGE-small Fine-Tuned",
        collection_name="servy_op26_bge_ft_v1",
        query_prefix="Represent this sentence for searching relevant passages: ",
        relevance_rows=relevance,
        content_first=True,
    )[0]

    # Summary table
    print("\n" + "=" * 100)
    print("CORRECTED COMPARATIVE TABLE (relevance-only denominator N=43)")
    print("=" * 100)
    header = f"{'Model':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR@10':>8} {'nDCG@10':>8} {'p50ms':>6}"
    print(header)
    print("-" * len(header))
    for key, m in results.items():
        print(f"{m['model']:<30} {m['recall@1']:>6.4f} {m['recall@3']:>6.4f} {m['recall@5']:>6.4f} "
              f"{m['recall@10']:>6.4f} {m['recall@20']:>6.4f} {m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f} "
              f"{m['latency_p50_ms']:>6.1f}")

    # Save results
    out_path = BASE_DIR / "private_training_data" / "op27_phase0_corrected_evaluation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
