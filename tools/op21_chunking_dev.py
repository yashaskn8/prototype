"""Read-only, parent-mapped chunking ablations on six weak validation queries."""
import json
import os
import re
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
from core.ml.embeddings import BGE_QUERY_PREFIX, EmbeddingProfile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.models import Asset, KnowledgeChunk, Tenant
from core.services.retriever import _knowledge_candidates

private = ROOT / "private_training_data"
output = private / "op21_accuracy_audit" / "chunking_dev.json"
rows = [json.loads(line) for line in (private / "company_val_weakly_supervised.jsonl").read_text(encoding="utf-8").splitlines()]
chunks = list(_eligible_chunks())
by_id = {c.id: c for c in KnowledgeChunk.objects.select_related("document__asset").all()}
tenants = {t.id: t for t in Tenant.objects.all()}

profile = EmbeddingProfile(
    key="op21-chunking-dev", model_id=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
    revision=None, dim=384, collection="unused_op21_chunking_dev", query_prefix=BGE_QUERY_PREFIX,
    content_first=True, local_path=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
)
model = load_model(profile)
query_vectors = np.asarray(model.encode([BGE_QUERY_PREFIX + r["query_text"] for r in rows], normalize_embeddings=True, batch_size=8))


def children(chunk, mode):
    text = chunk.text.strip()
    heading = chunk.heading.strip()
    if mode == "original":
        return [passage_text(profile, chunk)]
    if mode == "paragraph":
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    elif mode == "procedure":
        blocks = [b.strip() for b in re.split(
            r"(?im)(?=^(?:Step\s+\d+\b|Expected result:|Create a service call|NOTE\b|WARNING\b))", text
        ) if b.strip()]
    elif mode == "short_child":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = ["\n".join(lines[i:i + 2]) for i in range(len(lines))] or [text]
    else:
        raise ValueError(mode)
    return [f"{heading}\n{block}" if heading and heading.lower() not in block.lower() else block for block in blocks]


allowed = []
for row in rows:
    positive = row["positive_chunk_ids"][0]
    source_doc = by_id[positive].document
    asset = source_doc.asset or Asset.objects.filter(
        tenant_id=row["tenant_id"], product_id=source_doc.product_id,
    ).order_by("id").first()
    if asset is None:
        raise RuntimeError(f"No matching asset for {row['query_id']}")
    ids = {c.id for c in _knowledge_candidates(
        tenant=tenants[row["tenant_id"]], asset=asset, customer=None,
        include_confidential=False, staff_mode=False,
    )}
    if positive not in ids:
        raise RuntimeError(f"Positive excluded from authorized candidates: {row['query_id']}")
    allowed.append(ids)

results = {}
for mode in ("original", "paragraph", "procedure", "short_child"):
    child_texts = []
    parent_ids = []
    for chunk in chunks:
        for child in children(chunk, mode):
            child_texts.append(child)
            parent_ids.append(chunk.id)
    started = time.perf_counter()
    vectors = np.asarray(model.encode(child_texts, normalize_embeddings=True, batch_size=16))
    encoding_seconds = time.perf_counter() - started
    records = []
    for row, query_vector, ids in zip(rows, query_vectors, allowed):
        cosines = vectors @ query_vector
        best = {}
        for parent, score in zip(parent_ids, cosines):
            if parent in ids:
                best[parent] = max(best.get(parent, -2.0), float(score))
        ranked = sorted(best, key=lambda cid: (-best[cid], cid))
        positive = row["positive_chunk_ids"][0]
        records.append({"query_id": row["query_id"], "positive_rank": ranked.index(positive) + 1, "ranked_chunk_ids": ranked})
    ranks = [r["positive_rank"] for r in records]
    results[mode] = {
        "child_vectors": len(child_texts), "corpus_encode_seconds": round(encoding_seconds, 2),
        "hit_at_1": sum(rank == 1 for rank in ranks),
        "hit_at_5": sum(rank <= 5 for rank in ranks),
        "mrr_at_10": round(sum(1/rank for rank in ranks if rank <= 10) / len(ranks), 4),
        "records": records,
    }

payload = {
    "population": "six weakly supervised validation queries; same authorized parent chunks; exploratory",
    "method": "max cosine score across child passages, then rank parent chunks; full parent retained for answering",
    "results": results,
}
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({key: {k: v for k, v in value.items() if k != "records"} for key, value in results.items()}, indent=2))
