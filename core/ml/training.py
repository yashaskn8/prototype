"""Embedding evaluation + guarded fine-tuning for Servy retrieval.

Evaluation reports Recall@1/5/10, MRR@10, nDCG@10 and query latency against
APPROVED labels only. Fine-tuning refuses to run on unapproved labels unless
explicitly invoked as a synthetic smoke test, whose output is tagged so it can
never be registered as a production model.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def evaluate_profile(profile_key: str, rows, k_max: int = 10) -> dict:
    from .embeddings import get_profile, load_model, get_collection, encode_queries
    p = get_profile(profile_key)
    model, coll = load_model(p), get_collection(p, create=False)
    r1 = r5 = r10 = mrr = ndcg = 0.0
    lat = []
    for r in rows:
        t = time.perf_counter()
        emb = encode_queries(p, model, [r["query_text"]])[0].tolist()
        res = coll.query(query_embeddings=[emb], n_results=k_max,
                         where={"tenant_id": int(r["tenant_id"])}, include=["metadatas"])
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [m["chunk_id"] for m in res["metadatas"][0]]
        rel = set(r["positive_chunk_ids"])
        hits = [i for i, c in enumerate(ranked) if c in rel]
        r1 += any(h < 1 for h in hits); r5 += any(h < 5 for h in hits); r10 += bool(hits)
        mrr += 1 / (hits[0] + 1) if hits else 0
        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k_max)))
        ndcg += dcg / idcg if idcg else 0
    n = max(len(rows), 1)
    lat.sort()
    return {"profile": p.key, "n_queries": len(rows), "recall@1": r1 / n, "recall@5": r5 / n,
            "recall@10": r10 / n, "mrr@10": mrr / n, "ndcg@10": ndcg / n,
            "latency_ms_p50": lat[len(lat) // 2] if lat else None}


def finetune(train_rows, val_rows, out_dir, base_revision, *, epochs=1, batch_size=8,
             lr=2e-5, max_seq_length=256, seed=42, synthetic_smoke_test=False) -> dict:
    """Fine-tune BGE-small with MultipleNegativesRankingLoss (+ explicit hard negatives).

    train_rows must contain query_text plus resolved 'positive_text' and optional
    'negative_text'. Unapproved rows are rejected unless synthetic_smoke_test=True.
    """
    if not synthetic_smoke_test and any(r.get("review_status") != "approved" for r in train_rows):
        raise ValueError("Refusing to train on labels that are not human-approved.")
    if not train_rows:
        raise ValueError("No training rows: BLOCKED - VERIFIED LABELS REQUIRED.")

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from .embeddings import BGE_QUERY_PREFIX

    torch.manual_seed(seed)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", revision=base_revision,
                                device="cpu", local_files_only=True)
    model.max_seq_length = max_seq_length

    has_neg = all(r.get("negative_text") for r in train_rows)
    def to_ds(rows):
        d = {"anchor": [BGE_QUERY_PREFIX + r["query_text"] for r in rows],
             "positive": [r["positive_text"] for r in rows]}
        if has_neg:
            d["negative"] = [r["negative_text"] for r in rows]
        return Dataset.from_dict(d)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, seed=seed, report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES, dataloader_num_workers=0,
        save_strategy="epoch", eval_strategy="epoch" if val_rows else "no",
        logging_steps=1, use_cpu=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=to_ds(train_rows),
        eval_dataset=to_ds(val_rows) if val_rows else None,
        loss=MultipleNegativesRankingLoss(model),
    )
    t = time.time()
    result = trainer.train()
    final = out_dir / "final"
    model.save(str(final))
    reloaded = SentenceTransformer(str(final), device="cpu")
    vec = reloaded.encode(["reload check"], normalize_embeddings=True)
    report = {
        "kind": "SYNTHETIC TRAINING PIPELINE SMOKE TEST" if synthetic_smoke_test else "verified-label fine-tune",
        "optimizer_steps": int(result.global_step), "train_loss": float(result.training_loss),
        "epochs": epochs, "n_train": len(train_rows), "n_val": len(val_rows or []),
        "hard_negatives": has_neg, "seed": seed, "lr": lr, "batch_size": batch_size,
        "base_revision": base_revision, "checkpoint": str(final),
        "reload_dim": int(vec.shape[1]), "seconds": round(time.time() - t, 1),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    return report
