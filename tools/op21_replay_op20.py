"""Replay Operation 20's tenant-only dense lookup without changing its index."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from sentence_transformers import SentenceTransformer
from core.ml.embeddings import BGE_QUERY_PREFIX
from core.services.chroma_store import get_chroma_client

OUT = ROOT / "private_training_data" / "op21_accuracy_audit"
rows = [json.loads(x) for x in (OUT / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
model = SentenceTransformer(str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"), device="cpu", local_files_only=True)
collection = get_chroma_client().get_collection("servy_dense_op20_bge_ft_v1")
vectors = model.encode([BGE_QUERY_PREFIX + r["query_text"] for r in rows], normalize_embeddings=True, batch_size=8)
replay = []
for row, vector in zip(rows, vectors):
    raw = collection.query(query_embeddings=[vector.tolist()], n_results=50,
                           where={"tenant_id": row["tenant_id"]},
                           include=["metadatas", "distances"])
    items = [{"chunk_id": int(m["chunk_id"]), "cosine_similarity": round(1 - float(dist), 5)}
             for m, dist in zip(raw["metadatas"][0], raw["distances"][0])]
    positive = set(row["positive_chunk_ids"])
    replay.append({
        "query_id": row["query_id"], "role": row["role"],
        "positive_chunk_ids": row["positive_chunk_ids"],
        "top_10": items[:10], "top_50_chunk_ids": [x["chunk_id"] for x in items],
        "positive_rank": next((i + 1 for i, x in enumerate(items) if x["chunk_id"] in positive), None),
    })
with (OUT / "op20_replay.jsonl").open("w", encoding="utf-8") as f:
    for row in replay:
        f.write(json.dumps(row) + "\n")
rel = [r for r in replay if r["role"] == "RELEVANCE_EVALUATION"]
print(json.dumps({
    "relevance_queries": len(rel),
    "hit_at_1": sum(r["positive_rank"] == 1 for r in rel),
    "hit_at_5": sum(r["positive_rank"] is not None and r["positive_rank"] <= 5 for r in rel),
    "hit_at_10": sum(r["positive_rank"] is not None and r["positive_rank"] <= 10 for r in rel),
    "recall_at_50": sum(r["positive_rank"] is not None and r["positive_rank"] <= 50 for r in rel),
}, indent=2))
