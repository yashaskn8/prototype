"""Measure warm, in-memory fair retrieval latency without training or index writes."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

import numpy as np
from core.ml.embeddings import BGE_QUERY_PREFIX, EmbeddingProfile, get_profile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.models import Tenant
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores, compute_char_ngram_scores

OUT = ROOT / "private_training_data" / "op21_accuracy_audit"
rows = [json.loads(line) for line in (OUT / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
chunks = list(_eligible_chunks())
tenants = {t.id: t for t in Tenant.objects.all()}
allowed = []
for row in rows:
    asset = resolve_query_asset(row["tenant_id"], row["equipment"])
    ids = {c.id for c in _knowledge_candidates(
        tenant=tenants[row["tenant_id"]], asset=asset, customer=None,
        include_confidential=False, staff_mode=False,
    )}
    allowed.append([i for i, c in enumerate(chunks) if c.id in ids])

profiles = {
    "fair_minilm": get_profile("minilm"),
    "fair_base_bge": get_profile("bge-small"),
    "fair_op20_bge_ft": EmbeddingProfile(
        key="op21-op20-bge-ft-latency",
        model_id=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
        revision=None, dim=384, collection="unused_op21_in_memory",
        query_prefix=BGE_QUERY_PREFIX, content_first=True,
        local_path=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
    ),
}

report = {}
for name, profile in profiles.items():
    print(f"Measuring {name}...", flush=True)
    start_load = time.perf_counter()
    model = load_model(profile)
    corpus = np.asarray(model.encode(
        [passage_text(profile, c) for c in chunks], normalize_embeddings=True, batch_size=8
    ), dtype=np.float32)
    load_and_index_ms = round((time.perf_counter() - start_load) * 1000, 2)
    durations = []
    rank_mismatches = 0
    for row, indices in zip(rows, allowed):
        start = time.perf_counter()
        q = np.asarray(model.encode(
            [profile.query_prefix + row["query_text"]], normalize_embeddings=True
        )[0], dtype=np.float32)
        scores = corpus[indices] @ q if indices else np.asarray([], dtype=np.float32)
        ranking = [chunks[indices[j]].id for j in sorted(range(len(indices)), key=lambda j: (-float(scores[j]), chunks[indices[j]].id))]
        durations.append((time.perf_counter() - start) * 1000)
        if ranking[:10] != row[name][:10]:
            rank_mismatches += 1
    ordered = sorted(durations)
    report[name] = {
        "n": len(durations), "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
        "model_load_and_corpus_encode_ms": load_and_index_ms,
        "ranking_mismatches_vs_batch_run": rank_mismatches,
        "scope": "warm model; authorized in-memory dense dot product; excludes DB and Chroma I/O",
    }
    del model, corpus

durations = []
for row, indices in zip(rows, allowed):
    texts = [f"{chunks[i].heading}\n{chunks[i].text}" for i in indices]
    start = time.perf_counter()
    bm25 = compute_bm25_scores(row["query_text"], texts)
    char = compute_char_ngram_scores(row["query_text"], texts)
    _ = sorted(range(len(indices)), key=lambda j: (-max(bm25[j], 0.5 * char[j]), chunks[indices[j]].id))
    durations.append((time.perf_counter() - start) * 1000)
ordered = sorted(durations)
report["lexical"] = {
    "n": len(durations), "p50_ms": round(statistics.median(ordered), 2),
    "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
    "scope": "warm in-memory BM25 + character ngrams; excludes DB I/O",
}
(OUT / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
