"""Offline, read-only cross-encoder audit on already authorized candidate lists."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import django
django.setup()

import numpy as np
from sentence_transformers import CrossEncoder
from core.models import KnowledgeChunk

parser = argparse.ArgumentParser()
parser.add_argument("population", choices=["dev", "holdout"])
args = parser.parse_args()

snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2" / "snapshots"
snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir()) if snapshot_root.exists() else []
if len(snapshots) != 1 or not (snapshots[0] / "model.safetensors").exists():
    raise RuntimeError("A unique complete local cross-encoder checkpoint is required; no download is allowed")
started = time.perf_counter()
model = CrossEncoder(str(snapshots[0]), device="cpu")
load_seconds = time.perf_counter() - started

private = ROOT / "private_training_data" / "op21_accuracy_audit"
if args.population == "dev":
    data = json.loads((private / "dev_experiments.json").read_text(encoding="utf-8"))["per_query"]
    questions = {r["query_id"]: r["query_text"] for r in [
        json.loads(line) for line in (ROOT / "private_training_data" / "company_val_weakly_supervised.jsonl").read_text(encoding="utf-8").splitlines()
    ]}
    rows = [{"query_id": r["query_id"], "query_text": questions[r["query_id"]],
             "positive_chunk_ids": [r["positive_chunk_id"]], "dense": r["dense"]} for r in data]
else:
    data = [json.loads(line) for line in (private / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [{"query_id": r["query_id"], "query_text": r["query_text"],
             "positive_chunk_ids": r["positive_chunk_ids"], "dense": r["fair_op20_bge_ft"]}
            for r in data if r["role"] == "RELEVANCE_EVALUATION"]

needed = {cid for r in rows for cid in r["dense"]}
chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=needed)}
depths = [8, 10, 20, 50]
records = []
latencies = []
for row in rows:
    ids = row["dense"]
    pairs = [[row["query_text"], f"{chunks[cid].heading}\n{chunks[cid].text}"] for cid in ids]
    started = time.perf_counter()
    logits = np.asarray(model.predict(pairs, batch_size=8, show_progress_bar=False), dtype=float).reshape(-1)
    latencies.append((time.perf_counter() - started) * 1000)
    result = {"query_id": row["query_id"], "positive_chunk_ids": row["positive_chunk_ids"]}
    for depth in depths:
        selected = ids[:depth]
        scores = logits[:len(selected)]
        ranked = sorted(range(len(selected)), key=lambda i: (-float(scores[i]), i))
        result[f"reranked_{depth}"] = [selected[i] for i in ranked]
    records.append(result)

metrics = {}
for depth in depths:
    ranked = [r[f"reranked_{depth}"] for r in records]
    ranks = [next((i + 1 for i, cid in enumerate(ids) if cid in set(r["positive_chunk_ids"])), None)
             for r, ids in zip(records, ranked)]
    n = len(ranks)
    metrics[str(depth)] = {
        "n": n,
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / n, 4),
        "hit_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / n, 4),
        "mrr_at_10": round(sum(1/rank for rank in ranks if rank and rank <= 10) / n, 4),
        "ndcg_at_10": round(sum(1/np.log2(rank + 1) for rank in ranks if rank and rank <= 10) / n, 4),
        "positive_candidate_recall": round(sum(rank is not None for rank in ranks) / n, 4),
    }

payload = {
    "population": args.population,
    "checkpoint": str(snapshots[0]),
    "model_load_seconds": round(load_seconds, 2),
    "p95_rerank_ms": round(float(np.percentile(latencies, 95)), 2),
    "scope": "warm CPU cross-encoder scoring of authorized dense candidates; excludes dense, DB and Chroma latency",
    "metrics": metrics,
    "per_query": records,
}
(private / f"rerank_{args.population}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in payload.items() if k != "per_query"}, indent=2))
