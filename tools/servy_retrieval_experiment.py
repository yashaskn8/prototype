"""Embedding-text experiment on the real manuals, in a temporary Chroma directory.

Compares how chunk text is embedded with the approved BGE-small model:
  A  heading + body                      (the current serving format)
  B  document title + heading + body     (contextualized)
Queries: the 46 documented-answer questions of the protected three-asset
dataset. Each is scored by the rank of the first chunk from its expected manual
section, among that asset's chunks only (dense retrieval, asset-filtered).
Nothing is written to production or to the development index.

Usage: SERVY_DB_ENGINE=mysql SERVY_MYSQL_ENV_FILE=.env.mysql.servy_dev SERVY_MEDIA_ROOT=... \\
       .venv\\Scripts\\python.exe -B tools\\servy_retrieval_experiment.py <report.json>
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django  # noqa: E402
django.setup()
import chromadb  # noqa: E402
import numpy as np  # noqa: E402
from core.ml.embeddings import BGE_QUERY_PREFIX, get_profile, load_model  # noqa: E402
from core.models import Asset, KnowledgeChunk  # noqa: E402
from tools import servy_three_asset_customer_eval as base  # noqa: E402

DEV_CODES = {59: "DEV-AST-059", 60: "DEV-AST-060", 61: "DEV-AST-061"}
STRATEGIES = {
    "A_heading_body": lambda c: f"{c.heading}\n{c.text}" if c.heading and c.heading not in c.text else c.text,
    "B_title_heading_body": lambda c: f"{c.document.title}\n{c.heading}\n{c.text}",
}


def main(report_path):
    model = load_model(get_profile("bge-small"))
    assets = {k: Asset.objects.get(asset_code=v) for k, v in DEV_CODES.items()}
    chunks = list(KnowledgeChunk.objects.filter(document__asset__in=assets.values(), is_quarantined=False)
                  .select_related("document").order_by("document_id", "chunk_index"))
    cases = [(asset, t) for _, asset, _, turns in base.CASES for t in turns[:1] if t["kind"] == base.ANSWER]
    results = {}
    with tempfile.TemporaryDirectory(prefix="servy_retrieval_exp_", ignore_cleanup_errors=True) as tmp:
        client = chromadb.PersistentClient(path=tmp)
        for name, render in STRATEGIES.items():
            coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})
            texts = [render(c) for c in chunks]
            started = time.perf_counter()
            vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
            embed_s = time.perf_counter() - started
            coll.add(ids=[str(c.id) for c in chunks], embeddings=vectors.tolist(),
                     metadatas=[{"asset": c.document.asset_id, "heading": c.heading} for c in chunks])
            ranks, query_ms = [], []
            for asset, turn in cases:
                started = time.perf_counter()
                q = model.encode([BGE_QUERY_PREFIX + turn["question"]], normalize_embeddings=True)
                hit = coll.query(query_embeddings=q.tolist(), n_results=10,
                                 where={"asset": assets[asset].id}, include=["metadatas"])
                query_ms.append((time.perf_counter() - started) * 1000)
                headings = [m["heading"] for m in hit["metadatas"][0]]
                rank = next((i + 1 for i, h in enumerate(headings) if turn["heading"].casefold() in h.casefold()), None)
                ranks.append(rank)
            n = len(ranks)
            results[name] = {
                "queries": n,
                "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
                "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
                "hit@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
                "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
                "missed@10": sum(r is None for r in ranks),
                "index_embed_seconds": round(embed_s, 2),
                "mean_query_ms": round(float(np.mean(query_ms)), 1),
            }
            print(name, json.dumps(results[name]))
    Path(report_path).write_text(json.dumps({"chunks": len(chunks), "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
