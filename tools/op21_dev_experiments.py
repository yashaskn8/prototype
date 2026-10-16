"""Small, exploratory retrieval ablations on the six weakly supervised validation cases."""
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

import numpy as np
from core.ml.embeddings import BGE_QUERY_PREFIX, EmbeddingProfile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.models import Asset, KnowledgeChunk, Tenant
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores, compute_char_ngram_scores

private = ROOT / "private_training_data"
out = private / "op21_accuracy_audit"
validation = [json.loads(line) for line in (private / "company_val_weakly_supervised.jsonl").read_text(encoding="utf-8").splitlines()]
chunks = list(_eligible_chunks())
by_id = {c.id: c for c in KnowledgeChunk.objects.select_related("document__asset").all()}
tenants = {t.id: t for t in Tenant.objects.all()}
profile = EmbeddingProfile(
    key="op21-dev-ft", model_id=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
    revision=None, dim=384, collection="unused_op21_dev", query_prefix=BGE_QUERY_PREFIX,
    content_first=True, local_path=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
)
model = load_model(profile)
corpus = np.asarray(model.encode([passage_text(profile, c) for c in chunks], normalize_embeddings=True, batch_size=8))
queries = np.asarray(model.encode([BGE_QUERY_PREFIX + r["query_text"] for r in validation], normalize_embeddings=True, batch_size=8))
results = []
for row, query in zip(validation, queries):
    positive = row["positive_chunk_ids"][0]
    source_document = by_id[positive].document
    asset = source_document.asset or Asset.objects.filter(
        tenant_id=row["tenant_id"], product_id=source_document.product_id,
    ).order_by("id").first()
    if asset is None:
        raise RuntimeError(f"No matching asset context for {row['query_id']}")
    allowed_ids = {c.id for c in _knowledge_candidates(
        tenant=tenants[row["tenant_id"]], asset=asset, customer=None,
        include_confidential=False, staff_mode=False,
    )}
    if positive not in allowed_ids:
        raise RuntimeError(f"Labeled positive is outside the authorized pool for {row['query_id']}")
    indices = [i for i, c in enumerate(chunks) if c.id in allowed_ids]
    cosine = corpus[indices] @ query
    original = [chunks[indices[j]].id for j in sorted(range(len(indices)), key=lambda j: (-float(cosine[j]), chunks[indices[j]].id))]
    non_overview = [cid for cid in original if
                    "general overview" not in by_id[cid].heading.lower() and
                    "purpose" not in by_id[cid].heading.lower()]
    texts = [f"{chunks[i].heading}\n{chunks[i].text}" for i in indices]
    bm25 = compute_bm25_scores(row["query_text"], texts)
    char = compute_char_ngram_scores(row["query_text"], texts)
    lexical = [chunks[indices[j]].id for j in sorted(
        range(len(indices)), key=lambda j: (-max(bm25[j], 0.5 * char[j]), chunks[indices[j]].id)
    )]
    results.append({
        "query_id": row["query_id"], "positive_chunk_id": positive,
        "dense": original, "dense_without_overview": non_overview,
        "lexical": lexical,
    })

def score(key):
    ranks = [next((i + 1 for i, cid in enumerate(r[key]) if cid == r["positive_chunk_id"]), None) for r in results]
    return {
        "hit_at_1": sum(x == 1 for x in ranks),
        "hit_at_5": sum(x is not None and x <= 5 for x in ranks),
        "mrr_at_10": round(sum(1 / x for x in ranks if x and x <= 10) / len(ranks), 4),
    }

summary = {
    "population": "six weakly supervised validation queries; exploratory only",
    "n": len(results),
    "dense": score("dense"),
    "dense_without_overview": score("dense_without_overview"),
    "lexical": score("lexical"),
}
(out / "dev_experiments.json").write_text(json.dumps({"summary": summary, "per_query": results}, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
