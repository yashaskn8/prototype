"""Warm offline retrieval latency, including DB scope, query encoding and ranking."""
import json
import os
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
import psutil
from core.ml.embeddings import get_profile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.ml.op22_technical_evidence import parse_technical_query, rank_authorized_candidates
from core.models import Asset, Tenant
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores

PRIVATE = ROOT / "private_training_data"
rows = [json.loads(line) for line in (PRIVATE / "company_val_weakly_supervised.jsonl").read_text(encoding="utf-8").splitlines()]
from core.models import KnowledgeChunk
cases = []
for row in rows:
    chunk = KnowledgeChunk.objects.select_related("document").get(id=row["positive_chunk_ids"][0])
    asset = chunk.document.asset or Asset.objects.filter(tenant_id=row["tenant_id"], product_id=chunk.document.product_id).order_by("id").first()
    cases.append((row["query_text"], Tenant.objects.get(id=row["tenant_id"]), asset))

profile = get_profile("minilm")
model = load_model(profile)
chunks = list(_eligible_chunks())
matrix = np.asarray(model.encode([passage_text(profile, c) for c in chunks], normalize_embeddings=True, batch_size=8), dtype=np.float32)
chunk_vectors = {c.id: matrix[i] for i, c in enumerate(chunks)}
rss_loaded = psutil.Process().memory_info().rss
durations = {"fair_minilm": [], "evidence_ranked": []}
for repeat in range(6):
    for query_text, tenant, asset in cases:
        for method in durations:
            start = time.perf_counter()
            authorized = [c for c in _knowledge_candidates(tenant, asset, None, False) if c.id in chunk_vectors]
            qvector = np.asarray(model.encode([profile.query_prefix + query_text], normalize_embeddings=True))[0]
            dense_scores = [float(chunk_vectors[c.id] @ qvector) for c in authorized]
            dense_order = sorted(range(len(authorized)), key=lambda i: (-dense_scores[i], authorized[i].id))
            if method == "evidence_ranked":
                lexical_scores = compute_bm25_scores(query_text, [f"{c.heading}\n{c.text}" for c in authorized])
                lexical_order = sorted(range(len(authorized)), key=lambda i: (-float(lexical_scores[i]), authorized[i].id))
                _ = rank_authorized_candidates(
                    parse_technical_query(query_text), authorized,
                    {authorized[i].id: rank + 1 for rank, i in enumerate(dense_order)},
                    {authorized[i].id: rank + 1 for rank, i in enumerate(lexical_order)},
                    verified_document_ids={c.document_id for c in authorized}, asset=asset,
                )
            durations[method].append((time.perf_counter() - start) * 1000)

report = {
    "population": "six weakly supervised development queries repeated six times",
    "measurement_scope": "warm offline retrieval: SQLite authorization/candidate query, MiniLM query encoding, in-memory vectors, lexical score for evidence ranker, feature ranking; excludes corpus encoding, Chroma I/O, LLM generation, network and citations",
    "model_and_corpus_loaded_rss_mib": round(rss_loaded / 1024**2, 2),
    "methods": {name: {"n": len(samples), "p50_ms": round(float(np.percentile(samples, 50)), 2),
                        "p95_ms": round(float(np.percentile(samples, 95)), 2)}
                for name, samples in durations.items()},
}
(PRIVATE / "op22_review" / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
