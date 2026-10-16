"""Read-only base BGE similarity audit of weakly supervised training negatives."""
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
from core.ml.embeddings import BGE_QUERY_PREFIX, get_profile, load_model, passage_text
from core.models import KnowledgeChunk

private = ROOT / "private_training_data"
out = private / "op21_accuracy_audit"
rows = []
for split, filename in (("train", "company_train_weakly_supervised.jsonl"),
                        ("validation", "company_val_weakly_supervised.jsonl")):
    for line in (private / filename).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append({"split": split, **json.loads(line)})
ids = {cid for row in rows for cid in row.get("positive_chunk_ids", []) + row.get("negative_chunk_ids", [])}
chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=ids).select_related("document")}
profile = get_profile("bge-small")
model = load_model(profile)
questions = [BGE_QUERY_PREFIX + r["query_text"] for r in rows]
positives = [passage_text(profile, chunks[r["positive_chunk_ids"][0]]) for r in rows]
negatives = [passage_text(profile, chunks[r["negative_chunk_ids"][0]]) for r in rows]
qv = np.asarray(model.encode(questions, normalize_embeddings=True, batch_size=8))
pv = np.asarray(model.encode(positives, normalize_embeddings=True, batch_size=8))
nv = np.asarray(model.encode(negatives, normalize_embeddings=True, batch_size=8))
records = []
for i, row in enumerate(rows):
    pos = float(qv[i] @ pv[i]); neg = float(qv[i] @ nv[i])
    records.append({
        "query_id": row["query_id"], "split": row["split"],
        "positive_chunk_id": row["positive_chunk_ids"][0],
        "negative_chunk_id": row["negative_chunk_ids"][0],
        "base_bge_positive_similarity": round(pos, 5),
        "base_bge_negative_similarity": round(neg, 5),
        "positive_minus_negative": round(pos - neg, 5),
        "same_document": chunks[row["positive_chunk_ids"][0]].document_id == chunks[row["negative_chunk_ids"][0]].document_id,
    })
with (out / "training_negative_margins.jsonl").open("w", encoding="utf-8") as file:
    for row in records:
        file.write(json.dumps(row) + "\n")
margins = sorted(r["positive_minus_negative"] for r in records)
summary = {
    "n": len(records),
    "mean_margin": round(sum(margins) / len(margins), 5),
    "median_margin": margins[len(margins) // 2],
    "negative_outranks_positive": sum(m < 0 for m in margins),
    "easy_margin_over_0_15": sum(m > 0.15 for m in margins),
    "caveat": "Embedding margin is a proxy; human relevance review is still required.",
}
(out / "training_negative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
