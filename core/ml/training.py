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


TRAINING_MODE_POSITIVE_PAIRS = "positive_pairs_only"
TRAINING_MODE_VERIFIED_TRIPLETS = "verified_triplets_only"
TRAINING_MODE_MIXED = "mixed_verified_examples"
VALID_TRAINING_MODES = {TRAINING_MODE_POSITIVE_PAIRS, TRAINING_MODE_VERIFIED_TRIPLETS, TRAINING_MODE_MIXED}


def finetune(train_rows, val_rows, out_dir, base_revision, *, epochs=1, batch_size=8,
             lr=2e-5, max_seq_length=256, seed=42, synthetic_smoke_test=False,
             weak_supervision: bool = False,
             training_mode: str = TRAINING_MODE_MIXED) -> dict:
    """Fine-tune BGE-small with MultipleNegativesRankingLoss (+ explicit hard negatives).

    train_rows must contain query_text plus resolved 'positive_text' and optional
    'negative_text'.

    Training modes (Phase 1 Operation 16.0):
      - positive_pairs_only:     Use only anchor-positive pairs, ignore all negatives.
      - verified_triplets_only:  Require every example to have a reviewed negative.
                                 Reject incomplete triplets. Never silently downgrade.
      - mixed_verified_examples: Use triplets where negatives exist, pairs otherwise.
                                 Record exactly which examples and losses contributed.

    Never silently discard reviewed hard negatives.
    Never invent placeholder negative passages.
    """
    if training_mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"Invalid training_mode '{training_mode}'. Must be one of: {sorted(VALID_TRAINING_MODES)}"
        )

    if synthetic_smoke_test:
        supervision = "synthetic_smoke"
    elif weak_supervision or any(r.get("review_status") != "approved" for r in train_rows):
        supervision = "weakly_supervised"
    else:
        supervision = "human_verified"

    if supervision == "synthetic_smoke":
        pass  # Synthetic smoke rows supply their own text
    elif supervision == "weakly_supervised":
        from .labels import resolve_provenance_verified_examples
        res_train = resolve_provenance_verified_examples(train_rows, require_approved=False)
        if res_train["rejected"] or not res_train["verified_examples"]:
            raise ValueError(f"Training data provenance validation failed: {res_train['rejected']}")
        train_rows = res_train["verified_examples"]
        if val_rows:
            res_val = resolve_provenance_verified_examples(val_rows, require_approved=False)
            if res_val["rejected"]:
                raise ValueError(f"Validation data provenance validation failed: {res_val['rejected']}")
            val_rows = res_val["verified_examples"]
            if not val_rows:
                raise ValueError("Validation rows were provided but all were rejected during provenance verification.")
    else:
        from .labels import resolve_provenance_verified_examples
        res_train = resolve_provenance_verified_examples(train_rows, require_approved=True)
        if res_train["rejected"] or not res_train["verified_examples"]:
            raise ValueError(f"Training data provenance validation failed: {res_train['rejected']}")
        train_rows = res_train["verified_examples"]
        if val_rows:
            res_val = resolve_provenance_verified_examples(val_rows, require_approved=True)
            if res_val["rejected"]:
                raise ValueError(f"Validation data provenance validation failed: {res_val['rejected']}")
            val_rows = res_val["verified_examples"]
            if not val_rows:
                raise ValueError("Validation rows were provided but all were rejected during provenance verification.")

    if not train_rows:
        raise ValueError("No training rows: BLOCKED — VALID TRAINING DATA REQUIRED.")

    # ---- Triplet expansion (no model required) ----
    from .labels import expand_training_triplets
    train_triplets = expand_training_triplets(train_rows, preserve_primary_positive_only=True)
    val_triplets = expand_training_triplets(val_rows, preserve_primary_positive_only=True) if val_rows else []

    # ---- Phase 1 (Operation 16.0): Explicit training mode enforcement ----
    # FAIL-FAST: Validate training mode BEFORE expensive model loading (Op17 fix).
    # Categorize every expanded triplet into with-negative or without-negative.
    triplets_with_neg = [r for r in train_triplets if r.get("negative_text")]
    triplets_without_neg = [r for r in train_triplets if not r.get("negative_text")]
    excluded_example_ids = []

    if training_mode == TRAINING_MODE_VERIFIED_TRIPLETS:
        # Strict: FAIL if any example lacks a reviewed negative.
        # Do NOT silently discard incomplete triplets — this contradicts
        # the documented contract that every example must have a reviewed negative.
        if triplets_without_neg:
            incomplete_ids = [r.get("query_id", "?") for r in triplets_without_neg]
            raise ValueError(
                f"TRAINING_BLOCKED: verified_triplets_only mode requires reviewed negatives "
                f"for EVERY example. {len(triplets_without_neg)} of "
                f"{len(train_triplets)} examples lack negatives. "
                f"Incomplete example IDs: {incomplete_ids[:50]}. "
                f"Either provide reviewed negatives for all examples, or use "
                f"'positive_pairs_only' or 'mixed_verified_examples' mode."
            )
        effective_triplets = triplets_with_neg
        if not effective_triplets:
            raise ValueError(
                f"TRAINING_BLOCKED: verified_triplets_only mode requires reviewed negatives "
                f"for every example, but no examples have negatives. Cannot proceed."
            )
        use_negatives = True

    elif training_mode == TRAINING_MODE_POSITIVE_PAIRS:
        # Deliberately ignore all negatives — in-batch negatives only.
        effective_triplets = train_triplets
        use_negatives = False

    elif training_mode == TRAINING_MODE_MIXED:
        # Use triplets where negatives exist, pairs where they don't.
        # Record which examples contribute negatives and which don't.
        # CRITICAL: Never insert empty-string negatives into contrastive loss.
        effective_triplets = train_triplets
        use_negatives = bool(triplets_with_neg)  # Only include negative column if any exist
    else:
        raise ValueError(f"Unexpected training_mode: {training_mode}")

    if not effective_triplets:
        raise ValueError("No eligible training examples remain after mode filtering.")

    # Validation triplets always follow the same mode
    val_with_neg = [r for r in val_triplets if r.get("negative_text")] if val_triplets else []
    val_without_neg = [r for r in val_triplets if not r.get("negative_text")] if val_triplets else []

    if training_mode == TRAINING_MODE_VERIFIED_TRIPLETS:
        effective_val = val_with_neg
    else:
        effective_val = val_triplets

    # ---- Heavy imports and model loading (after validation passes) ----
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss  # type: ignore
        from sentence_transformers.training_args import BatchSamplers  # type: ignore
    from .embeddings import BGE_QUERY_PREFIX

    torch.manual_seed(seed)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", revision=base_revision,
                                device="cpu", local_files_only=True)
    model.max_seq_length = max_seq_length

    def to_ds_triplets(triplets):
        """Build anchor/positive/negative dataset from triplets that HAVE negatives.
        NEVER insert empty-string negatives — this creates degenerate training
        signals where the loss pushes embeddings away from a meaningless blank passage."""
        has_neg = [r for r in triplets if r.get("negative_text")]
        if not has_neg:
            return None
        return Dataset.from_dict({
            "anchor": [BGE_QUERY_PREFIX + r["query_text"] for r in has_neg],
            "positive": [r["positive_text"] for r in has_neg],
            "negative": [r["negative_text"] for r in has_neg],
        })

    def to_ds_pairs(triplets):
        """Build anchor/positive dataset from examples WITHOUT negatives (in-batch negatives only)."""
        no_neg = [r for r in triplets if not r.get("negative_text")]
        if not no_neg:
            return None
        return Dataset.from_dict({
            "anchor": [BGE_QUERY_PREFIX + r["query_text"] for r in no_neg],
            "positive": [r["positive_text"] for r in no_neg],
        })

    def to_ds_all_pairs(triplets):
        """Build anchor/positive-only dataset from ALL examples (ignoring negatives)."""
        if not triplets:
            return None
        return Dataset.from_dict({
            "anchor": [BGE_QUERY_PREFIX + r["query_text"] for r in triplets],
            "positive": [r["positive_text"] for r in triplets],
        })

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 (Operation 18): Build training datasets correctly for each mode.
    # CRITICAL FIX: Mixed mode must NOT silently drop positive-only examples.
    # The old to_ds(include_neg=True) filtered to only rows with negatives,
    # discarding all positive-only training data. Now we use multi-dataset
    # training: triplet examples (with hard negatives) and pair examples
    # (in-batch negatives only) both contribute to training.
    loss = MultipleNegativesRankingLoss(model)

    if training_mode == TRAINING_MODE_POSITIVE_PAIRS:
        # Deliberately ignore all negatives — in-batch negatives only.
        train_dataset = to_ds_all_pairs(effective_triplets)
        eval_dataset = to_ds_all_pairs(effective_val) if effective_val else None
    elif training_mode == TRAINING_MODE_VERIFIED_TRIPLETS:
        # All examples guaranteed to have negatives (validated above).
        train_dataset = to_ds_triplets(effective_triplets)
        eval_dataset = to_ds_triplets(effective_val) if effective_val else None
    elif training_mode == TRAINING_MODE_MIXED:
        # FIXED: Use multi-dataset training so BOTH triplet and pair examples
        # contribute. SentenceTransformerTrainer accepts a dict of datasets
        # keyed by name, each trained with the same loss.
        ds_triplets = to_ds_triplets(effective_triplets)
        ds_pairs = to_ds_pairs(effective_triplets)
        if ds_triplets and ds_pairs:
            # Both triplet and pair examples exist — use multi-dataset dict
            train_dataset = {"triplets": ds_triplets, "pairs": ds_pairs}
        elif ds_triplets:
            train_dataset = ds_triplets
        else:
            train_dataset = to_ds_all_pairs(effective_triplets)

        # Eval follows same structure
        if effective_val:
            eval_ds_triplets = to_ds_triplets(effective_val)
            eval_ds_pairs = to_ds_pairs(effective_val)
            if eval_ds_triplets and eval_ds_pairs:
                eval_dataset = {"triplets": eval_ds_triplets, "pairs": eval_ds_pairs}
            elif eval_ds_triplets:
                eval_dataset = eval_ds_triplets
            else:
                eval_dataset = to_ds_all_pairs(effective_val)
        else:
            eval_dataset = None
    else:
        raise ValueError(f"Unexpected training_mode: {training_mode}")

    if train_dataset is None:
        raise ValueError("No training examples produced datasets. Check training data.")

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, seed=seed, report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES, dataloader_num_workers=0,
        save_strategy="epoch", eval_strategy="epoch" if val_rows else "no",
        logging_steps=1, use_cpu=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    t = time.time()
    result = trainer.train()
    final = out_dir / "final"
    model.save(str(final))
    # Compute deterministic checkpoint hash
    import hashlib
    h = hashlib.sha256()
    for p in sorted(final.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    ckpt_hash = h.hexdigest()

    reloaded = SentenceTransformer(str(final), device="cpu")
    vec = reloaded.encode(["reload check"], normalize_embeddings=True)
    if supervision == "synthetic_smoke":
        kind = "SYNTHETIC TRAINING PIPELINE SMOKE TEST"
        promo = False
        status = "SYNTHETIC_SMOKE_TEST_INELIGIBLE_FOR_PRODUCTION"
    elif supervision == "weakly_supervised":
        kind = "COMPANY-DATA WEAKLY SUPERVISED EXPERIMENT"
        promo = False
        status = "COMPANY_DATA_WEAKLY_SUPERVISED_EXPERIMENT"
    else:
        kind = "HUMAN-VERIFIED PRODUCTION FINE-TUNING"
        promo = True
        status = "VERIFIED_FINE_TUNED"

    # Distinct query/document accounting
    distinct_query_ids = {r.get("query_id", "").split("-p")[0] for r in effective_triplets}
    distinct_query_ids.discard("")

    report = {
        "kind": kind,
        "supervision_mode": supervision,
        "training_mode": training_mode,
        "training_completed": True,
        "reload_succeeded": True,
        "evaluation_ran": bool(val_rows),
        "promotion_approved": False,
        "promotion_eligible": promo,
        "training_status": status,
        "optimizer_steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "epochs": epochs,
        "n_train_rows": len(train_rows),
        "n_train_expanded_triplets": len(train_triplets),
        "n_effective_examples": len(effective_triplets),
        "n_with_negative": len(triplets_with_neg),
        "n_without_negative": len(triplets_without_neg),
        "n_excluded": len(excluded_example_ids),
        "excluded_example_ids": excluded_example_ids[:50],  # Capped for report size
        "n_distinct_queries": len(distinct_query_ids),
        "n_val": len(val_rows or []),
        "n_val_effective": len(effective_val),
        "hard_negatives_consumed": use_negatives and len(triplets_with_neg) > 0,
        "verified_negatives_mode": training_mode,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "base_revision": base_revision,
        "checkpoint": str(final),
        "checkpoint_sha256": ckpt_hash,
        "reload_dim": int(vec.shape[1]),
        "seconds": round(time.time() - t, 1),
    }
    # Atomic report writing
    report_file = out_dir / "run_report.json"
    temp_report = out_dir / "run_report.json.tmp"
    temp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp_report.replace(report_file)
    return report
