"""Read-only dataset, source, tokenizer, and Chroma integrity inventory for Op21."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from core.ml.embeddings import compute_checkpoint_fingerprint, get_profile, passage_text
from core.ml.indexer import _chunk_key, _eligible_chunks
from core.models import KnowledgeChunk, KnowledgeDocument, SourceProvenanceRecord
from core.services.chroma_store import get_chroma_client
from tokenizers import Tokenizer


OUT = ROOT / "private_training_data" / "op21_accuracy_audit"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def main():
    private = ROOT / "private_training_data"
    train = read_jsonl(private / "company_train_weakly_supervised.jsonl")
    val = read_jsonl(private / "company_val_weakly_supervised.jsonl")
    holdout = read_jsonl(private / "frozen_independent_holdout_benchmark.jsonl")
    all_chunks = {c.id: c for c in KnowledgeChunk.objects.select_related("document", "document__asset").all()}
    eligible = list(_eligible_chunks())
    docs = list(KnowledgeDocument.objects.select_related("tenant").all())
    provenance = list(SourceProvenanceRecord.objects.all())

    training = train + val
    train_pos = {x for r in training for x in r.get("positive_chunk_ids", [])}
    holdout_pos = {x for r in holdout for x in r.get("positive_chunk_ids", [])}
    train_docs = {all_chunks[x].document_id for x in train_pos if x in all_chunks}
    holdout_docs = {all_chunks[x].document_id for x in holdout_pos if x in all_chunks}
    train_norm = {norm(r["query_text"]) for r in train}
    val_norm = {norm(r["query_text"]) for r in val}
    holdout_norm = {norm(r["query_text"]) for r in holdout}
    negative_pairs = [(all_chunks[p], all_chunks[n]) for r in training
                      for p in r.get("positive_chunk_ids", [])
                      for n in r.get("negative_chunk_ids", []) if p in all_chunks and n in all_chunks]

    tokenizer = Tokenizer.from_file(str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final" / "tokenizer.json"))
    bge = get_profile("bge-small")
    token_lengths = {c.id: len(tokenizer.encode(passage_text(bge, c)).ids)
                     for c in eligible}
    hist = Counter()
    for n in token_lengths.values():
        hist["0-64" if n <= 64 else "65-128" if n <= 128 else "129-256" if n <= 256 else "257-512" if n <= 512 else ">512"] += 1

    checkpoint = ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"
    op20_report = json.loads((ROOT / "ml_models" / "op20_run_report.json").read_text(encoding="utf-8"))
    cp_full = compute_checkpoint_fingerprint(str(checkpoint), short=False)
    cp_training_hash = hashlib.sha256()
    for f in sorted(checkpoint.rglob("*")):
        if f.is_file():
            cp_training_hash.update(f.name.encode())
            cp_training_hash.update(f.read_bytes())

    collection_info = {}
    client = get_chroma_client()
    for collection in client.list_collections():
        info = {"count": collection.count(), "metadata": collection.metadata}
        try:
            got = collection.get(include=["metadatas"])
            metas = got.get("metadatas") or []
            ids = got.get("ids") or []
            info["sample_metadata"] = metas[0] if metas else None
            info["tenant_counts"] = dict(Counter(str((m or {}).get("tenant_id")) for m in metas))
            info["unresolvable_chunk_id_count"] = sum(
                bool((m or {}).get("chunk_id")) and (m or {}).get("chunk_id") not in all_chunks
                for m in metas
            )
            info["id_chunk_mismatch_count"] = sum(
                ((m or {}).get("chunk_id") in all_chunks and
                 ids[i] != _chunk_key(all_chunks[(m or {})["chunk_id"]]))
                for i, m in enumerate(metas)
                if (m or {}).get("document_version") is not None
            )
            info["metadata_chunk_identity_mismatch_count"] = sum(
                ((m or {}).get("chunk_id") in all_chunks and
                 ((m or {}).get("document_id") != all_chunks[(m or {})["chunk_id"]].document_id or
                  (m or {}).get("chunk_index") != all_chunks[(m or {})["chunk_id"]].chunk_index or
                  (m or {}).get("tenant_id") != all_chunks[(m or {})["chunk_id"]].tenant_id))
                for m in metas
            )
            if collection.name == "servy_dense_op20_bge_ft_v1":
                present = {int(m.get("chunk_id")) for m in metas if m and m.get("chunk_id")}
                info["eligible_chunk_coverage"] = len(present & {c.id for c in eligible})
                info["eligible_chunks_missing"] = sorted({c.id for c in eligible} - present)
                info["content_hash_lengths"] = dict(Counter(len((m or {}).get("content_sha256") or "") for m in metas))
                info["current_content_hash_mismatch_count"] = sum(
                    (not m.get("content_sha256")) or
                    m.get("content_sha256") != hashlib.sha256(passage_text(bge, all_chunks[m["chunk_id"]]).encode()).hexdigest()[:len(m["content_sha256"])]
                    for m in metas if m and m.get("chunk_id") in all_chunks
                )
                info["current_content_hash_mismatch_chunk_ids"] = [
                    m["chunk_id"] for m in metas if m and m.get("chunk_id") in all_chunks and
                    ((not m.get("content_sha256")) or
                     m.get("content_sha256") != hashlib.sha256(passage_text(bge, all_chunks[m["chunk_id"]]).encode()).hexdigest()[:len(m["content_sha256"])])
                ]
                info["mismatch_matches_raw_body"] = [
                    m["chunk_id"] for m in metas if m and m.get("chunk_id") in all_chunks and
                    m["chunk_id"] in info["current_content_hash_mismatch_chunk_ids"] and
                    m.get("content_sha256") == hashlib.sha256(all_chunks[m["chunk_id"]].text.encode()).hexdigest()[:len(m.get("content_sha256") or "")]
                ]
        except Exception as exc:
            info["read_error"] = type(exc).__name__ + ": " + str(exc)
        collection_info[collection.name] = info

    source_docs = [p for p in provenance if p.local_object_type == "KnowledgeDocument"]
    summary = {
        "sqlite_scope": "local snapshot; live MySQL provenance unverified",
        "database": {
            "documents_total": len(docs), "chunks_total": len(all_chunks),
            "documents_by_tenant": dict(Counter(d.tenant.slug for d in docs)),
            "eligible_documents": len({c.document_id for c in eligible}),
            "eligible_chunks": len(eligible),
            "eligible_chunks_by_tenant": dict(Counter(c.tenant_id for c in eligible)),
            "source_provenance_total": len(provenance),
            "source_provenance_object_types": dict(Counter(p.external_object_type for p in provenance)),
            "source_document_records": len(source_docs),
            "source_document_verification_status": dict(Counter(p.verification_status for p in source_docs)),
            "unique_source_document_external_ids": len({(p.source_system, p.source_org, p.external_id) for p in source_docs}),
        },
        "training": {
            "train_rows": len(train), "val_rows": len(val),
            "review_status_counts": dict(Counter(r.get("review_status") for r in training)),
            "approved_with_reviewer": sum(r.get("review_status") == "approved" and bool(r.get("reviewer")) for r in training),
            "rows_by_tenant": dict(Counter(str(r["tenant_id"]) for r in training)),
            "source_group_count": len({r.get("source_group") for r in training}),
            "positive_document_count": len(train_docs),
            "negative_pairs": len(negative_pairs),
            "negative_same_document": sum(p.document_id == n.document_id for p, n in negative_pairs),
            "negative_same_tenant": sum(p.tenant_id == n.tenant_id for p, n in negative_pairs),
            "positive_ids_also_used_as_negatives": len(train_pos & {x for r in training for x in r.get("negative_chunk_ids", [])}),
            "train_val_exact_query_overlap": len(train_norm & val_norm),
            "train_holdout_exact_query_overlap": len(train_norm & holdout_norm),
            "val_holdout_exact_query_overlap": len(val_norm & holdout_norm),
            "train_holdout_positive_chunk_overlap": len(train_pos & holdout_pos),
            "holdout_positive_chunk_count": len(holdout_pos),
            "train_holdout_document_overlap": len(train_docs & holdout_docs),
            "holdout_positive_document_count": len(holdout_docs),
        },
        "tokens": {
            "passage_lengths": dict(hist),
            "passages_over_256": sum(n > 256 for n in token_lengths.values()),
            "positive_train_val_over_256": sum(token_lengths.get(x, 0) > 256 for x in train_pos),
            "positive_holdout_over_256": sum(token_lengths.get(x, 0) > 256 for x in holdout_pos),
            "max_length": max(token_lengths.values()),
            "median_length": sorted(token_lengths.values())[len(token_lengths) // 2],
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "files_present": checkpoint.is_dir(),
            "current_collection_fingerprint": cp_full,
            "current_training_algorithm_hash": cp_training_hash.hexdigest(),
            "reported_training_hash": op20_report["phases"]["phase_5"]["checkpoint_sha256"],
            "reported_collection_fingerprint": op20_report["phases"]["phase_6"]["checkpoint_fingerprint"],
            "report_mismatch_flag": op20_report["phases"]["phase_6"]["fingerprint_matches_report"],
            "older_company_ft_fingerprint": compute_checkpoint_fingerprint(str(ROOT / "ml_models" / "checkpoints" / "servy_company_bge_v1" / "final")),
        },
        "collections": collection_info,
    }
    (OUT / "integrity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["database", "training", "tokens", "checkpoint"]}, indent=2))
    print(json.dumps({k: {"count": v["count"], "metadata": v["metadata"], "eligible_chunk_coverage": v.get("eligible_chunk_coverage"), "eligible_chunks_missing": v.get("eligible_chunks_missing"), "current_content_hash_mismatch_count": v.get("current_content_hash_mismatch_count")} for k, v in collection_info.items()}, indent=2))


if __name__ == "__main__":
    main()
