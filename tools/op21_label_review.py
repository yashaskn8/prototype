"""Write local private passage comparisons for manual Op21 label review."""
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

from core.models import KnowledgeChunk

OUT = ROOT / "private_training_data" / "op21_accuracy_audit"
rows = [json.loads(line) for line in (OUT / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
chunks = {c.id: c for c in KnowledgeChunk.objects.select_related("document").all()}
comparisons = []
for row in rows:
    if row["role"] != "RELEVANCE_EVALUATION":
        continue
    positive_ids = row["positive_chunk_ids"]
    top_ids = row["fair_op20_bge_ft"][:3]
    comparisons.append({
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "category": row["category"],
        "equipment": row["equipment"],
        "positive_passages": [
            {"chunk_id": cid, "document_id": chunks[cid].document_id,
             "document_title": chunks[cid].document.title, "heading": chunks[cid].heading,
             "text": chunks[cid].text}
            for cid in positive_ids if cid in chunks
        ],
        "top_passages": [
            {"chunk_id": cid, "document_id": chunks[cid].document_id,
             "document_title": chunks[cid].document.title, "heading": chunks[cid].heading,
             "text": chunks[cid].text}
            for cid in top_ids if cid in chunks
        ],
    })
(OUT / "label_review.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
print(f"Wrote {len(comparisons)} relevance comparisons to private file")
terms = ["belt", "crankcase", "flare nut", "torque", "capacitor", "manifold", "branch selector", "siphon", "e-04", "e04", "f-07"]
print(json.dumps({term: [c.id for c in chunks.values() if term in (c.heading + " " + c.text).lower()] for term in terms}, indent=2))
