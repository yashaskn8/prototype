"""Fresh mentor-only exploratory BGE reranker fine-tuning on provisional Servy V4 labels.

This is NOT the strict production/approved-data training path. Only use data
in a separately authorized research environment. Source and label proposals
are *weakly supervised*, and the dev set is NOT an independent holdout.
No invented approvals, metrics, gold injection or production promotion.

Run on an explicitly company-permitted Colab GPU with durable approved storage.
The original strict servy_v4_colab_train.py admission gate is UNCHANGED.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from servy_v4_private_reconciliation import reconcile, read_jsonl, refs, normalize
from servy_checkpoint_audit import normalize_inputs, validate_candidates, evaluate_one
from servy_v4_prepare_research_artifacts import BGE_MODEL, BGE_REVISION, sha256
from servy_checkpoint_backup import backup_checkpoint, collect


class ExplorationBlocked(ValueError):
    pass


def demand(ok, why):
    if not ok:
        raise ExplorationBlocked(why)


def audited_inputs(split_dir, corpus_path, family_path, candidates_path):
    """Model-free, fail-closed data inspection; no approval invented."""
    split = Path(split_dir)
    files = {
        "train": split / "train_unapproved_candidate.jsonl",
        "dev": split / "dev_unapproved_candidate.jsonl",
        "quarantine": split / "quarantined_negatives.jsonl",
        "report": split / "resplit_report.json",
        "corpus": Path(corpus_path),
        "families": Path(family_path),
        "candidates": Path(candidates_path),
    }
    for name, path in files.items():
        demand(path.is_file(), "MISSING_" + name.upper() + ":" + str(path))
    report = json.loads(files["report"].read_text(encoding="utf-8"))
    demand(report.get("status") == "PROVISIONAL_RESPLIT_ONLY", "NOT_A_VERIFIED_DRAFT_RESPLIT")
    for key, filename in (
        ("train_unapproved_candidate.jsonl", "train"),
        ("dev_unapproved_candidate.jsonl", "dev"),
        ("quarantined_negatives.jsonl", "quarantine"),
    ):
        demand(report.get("output_sha256", {}).get(key) == sha256(files[filename]),
               "SPLIT_FILE_HASH_MISMATCH:" + key)
    demand(report.get("input_sha256", {}).get("corpus") == sha256(files["corpus"]),
           "CORPUS_HASH_MISMATCH")
    demand(report.get("input_sha256", {}).get("recommendations") == sha256(files["families"]),
           "FAMILY_RECOMMENDATION_HASH_MISMATCH")
    data = {name: read_jsonl(files[name]) for name in
            ("train", "dev", "quarantine", "corpus", "families", "candidates")}
    demand(sum(map(len, (data["train"], data["dev"]))) == report.get("total_queries"),
           "QUERY_COUNT_CHANGED")
    actual, _, _, _ = reconcile(data["train"], data["dev"],
                                data["corpus"], quarantine=data["quarantine"])
    demand(not actual["issue_counts"] and
           not any(actual["train_dev_overlap"].values()) and
           actual["quarantine_ledger_matches_split_proposals"],
           "CURRENT_DATA_INTEGRITY_OR_SPLIT_CHECK_FAILED:" +
           str(actual["issue_counts"]))

    by_id = {}
    families = {}
    for chunk in data["corpus"]:
        cid = str(chunk["chunk_id"])
        demand(cid not in by_id, "DUPLICATE_CHUNK")
        by_id[cid] = chunk
    for family in data["families"]:
        key = (str(family["tenant_id"]), str(family["document_id"]))
        demand(key not in families and family.get("evidence_status") == "REGISTRY_METADATA_MATCH",
               "INVALID_PROVISIONAL_FAMILY:" + repr(key))
        families[key] = str(family["proposed_canonical_source_family"])
        demand(bool(families[key]), "EMPTY_FAMILY")
    require_docs = {(str(c["tenant_id"]), str(c["document_id"]))
                    for c in data["corpus"]}
    demand(set(families) == require_docs, "MISSING_OR_EXTRA_PROPOSED_SOURCE_FAMILY")
    used = {"train": set(), "dev": set()}
    for split_name in ("train", "dev"):
        for row in data[split_name]:
            for role in ("positive", "negative"):
                ids = refs(row, role + "_chunk_ids") or refs(row, role + "_chunk_id")
                for cid in ids:
                    chunk = by_id.get(cid)
                    demand(chunk is not None and
                           str(chunk["tenant_id"]) == str(row["tenant_id"]),
                           "UNKNOWN_OR_CROSS_TENANT_SOURCE:" + cid)
                    key = (str(chunk["tenant_id"]), str(chunk["document_id"]))
                    used[split_name].add((key[0], families[key]))
    demand(not (used["train"] & used["dev"]), "PROPOSED_FAMILY_SPLIT_LEAKAGE")
    cases, chunks = normalize_inputs(data["dev"], data["corpus"])
    frozen = validate_candidates(cases, chunks, data["candidates"], top_k=10)
    train_pairs = provisional_pairs(data["train"], by_id)
    dev_pairs = provisional_pairs(data["dev"], by_id)
    return {
        "train_pairs": train_pairs, "dev_pairs": dev_pairs,
        "cases": cases, "chunks": chunks, "frozen": frozen,
        "hashes": {name: sha256(path) for name, path in files.items()},
        "train_queries": len(data["train"]), "dev_queries": len(data["dev"]),
        "quarantined_negative_references": len(data["quarantine"]),
        "source_families": len(set(families.values())),
        "source_isolation_verified_against_proposed_families": True,
        "candidate_gold_present": sum(bool(set(c["gold_chunk_ids"]) &
                                           set(frozen[qid])) for qid,c in cases.items()),
    }


def provisional_pairs(rows, by_id):
    """Materialize exact provisional labels, without setting review_status."""
    pairs = []
    for row in rows:
        q = str(row.get("query_text") or row.get("query") or "").strip()
        tenant = str(row.get("tenant_id") or "")
        demand(q and tenant, "QUERY_OR_TENANT_MISSING")
        for role, value in (("positive", 1.0), ("negative", 0.0)):
            ids = refs(row, role + "_chunk_ids") or refs(row, role + "_chunk_id")
            if role == "positive":
                demand(bool(ids), "MISSING_POSITIVE")
            for cid in ids:
                chunk = by_id.get(cid)
                demand(chunk is not None and str(chunk["tenant_id"]) == tenant,
                       "UNRESOLVED_TRAINING_SOURCE")
                passage = str(chunk.get("passage") or "").strip()
                demand(passage, "EMPTY_TRAINING_PASSAGE")
                pairs.append({"text1": q, "text2": passage, "label": value})
    demand({p["label"] for p in pairs} == {0.0, 1.0}, "BCE_REQUIRES_BOTH_CLASSES")
    demand(len({(p["text1"], p["text2"], p["label"]) for p in pairs}) == len(pairs),
           "DUPLICATE_PAIR_IN_DRAFT_LABELS")
    return pairs


def check_storage(outdir):
    out = Path(outdir).resolve()
    normalized = out.as_posix()
    demand(out.is_absolute(), "STORAGE_MUST_BE_ABSOLUTE")
    demand(out.parent.is_dir(), "DURABLE_STORAGE_PARENT_MISSING")
    demand(not out.exists(), "REFUSING_OVERWRITE_PREVIOUS_EXPERIMENT")
    demand(not (normalized == "/content" or
                (normalized.startswith("/content/") and
                 not normalized.startswith("/content/drive/"))),
           "EPHEMERAL_COLAB_CHECKPOINT_DESTINATION")
    if normalized.startswith("/content/drive/"):
        demand(os.path.ismount("/content/drive"), "GOOGLE_DRIVE_NOT_MOUNTED")


def run(args):
    demand(args.i_confirm_authorized_environment,
           "ACTUAL_PERMISSION_FOR_THIS_COLAB_ENVIRONMENT_REQUIRED")
    demand(args.i_accept_provisional_labels,
           "EXPLORATORY_WEAK_LABELS_MUST_BE_ACKNOWLEDGED")
    demand(args.i_accept_dev_not_independent,
           "DEV_METRICS_ARE_NOT_AN_INDEPENDENT_HOLDOUT")
    demand(args.epochs in (1, 2, 3) and 1 <= args.batch_size <= 4 and
           math.isfinite(args.learning_rate) and 0 < args.learning_rate <= 2e-5,
           "INVALID_EXPERIMENT_BOUNDS")
    check_storage(args.out_dir)
    data = audited_inputs(args.split_dir, args.corpus,
                          args.families, args.candidates)
    demand(version("sentence-transformers") == "5.1.0",
           "REQUIRE_SENTENCE_TRANSFORMERS_5_1_0")

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import numpy as np
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    demand(torch.cuda.is_available(), "CUDA_REQUIRED_L4_OR_T4")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    out = Path(args.out_dir).resolve()
    out.mkdir(mode=0o700)
    status = {
        "experiment": "SERVY_V4_EXPLORATORY_MENTOR_DEMO",
        "status": "IN_PROGRESS",
        "data_status": "MACHINE_REVIEWED_PROVISIONAL_LABELS_NOT_EXPERT_APPROVED",
        "trained_from_random_initialization": False,
        "started_from_pretrained_revision": BGE_REVISION,
        "model": BGE_MODEL,
        "dev_not_independent_holdout": True,
        "train_queries": data["train_queries"],
        "dev_queries": data["dev_queries"],
        "pair_counts": {
            "train": len(data["train_pairs"]), "dev": len(data["dev_pairs"])
        },
        "candidate_gold_present": data["candidate_gold_present"],
        "quarantined_negative_references": data["quarantined_negative_references"],
        "source_isolation_on_proposed_families": True,
        "input_sha256": data["hashes"],
        "production_promotion_allowed": False,
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="servy-mentor-") as wd:
        work = Path(wd)
        model = CrossEncoder(
            BGE_MODEL, revision=BGE_REVISION, num_labels=1,
            max_length=256, device="cuda",
        )
        def scorer(checkpoint):
            def score(pairs):
                values = checkpoint.predict(
                    pairs, activation_fn=torch.nn.Identity(),
                    show_progress_bar=False,
                )
                return np.asarray(values).ravel().tolist()
            return score
        baseline = evaluate_one(
            data["cases"], data["chunks"], data["frozen"],
            scorer(model), pool_size=5)
        (out / "before_metrics.json").write_text(
            json.dumps(baseline["metrics"], indent=2), encoding="utf-8")
        (out / "before_per_query.json").write_text(
            json.dumps(baseline["queries"], indent=2), encoding="utf-8")

        class DurableCheckpoint(TrainerCallback):
            def on_save(self, train_args, state, control, **kwargs):
                checkpoint = Path(train_args.output_dir) / f"checkpoint-{state.global_step}"
                demand(checkpoint.is_dir(), "MISSING_TRAIN_CHECKPOINT")
                backup_checkpoint(checkpoint, out / checkpoint.name)
                return control

        use_bf16 = bool(torch.cuda.is_bf16_supported())
        training_args = CrossEncoderTrainingArguments(
            output_dir=str(work / "epochs"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate, warmup_ratio=0.05,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=3,
            logging_steps=1, seed=42, data_seed=42,
            bf16=use_bf16, fp16=not use_bf16,
            dataloader_num_workers=0, report_to="none", push_to_hub=False,
        )
        trainer = CrossEncoderTrainer(
            model=model, args=training_args,
            train_dataset=Dataset.from_list(data["train_pairs"]),
            eval_dataset=Dataset.from_list(data["dev_pairs"]),
            loss=BinaryCrossEntropyLoss(model),
            callbacks=[DurableCheckpoint()],
        )
        trainer.train()
        demand(trainer.state.best_model_checkpoint is not None and
               trainer.state.best_metric is not None and
               math.isfinite(float(trainer.state.best_metric)),
               "NO_FINITE_SELECTED_CHECKPOINT")
        local_best = work / "selected_model"
        trainer.model.save_pretrained(str(local_best))
        backup_checkpoint(local_best, out / "best_model")
        reloaded = CrossEncoder(str(out / "best_model"),
                                max_length=256, device="cuda")
        after = evaluate_one(data["cases"], data["chunks"],
                             data["frozen"], scorer(reloaded), pool_size=5)
        (out / "after_per_query.json").write_text(
            json.dumps(after["queries"], indent=2), encoding="utf-8")
        comparison = {
            "dataset_status": "provisional, weakly supervised; no independent holdout",
            "candidate_method": args.candidate_method,
            "before": baseline["metrics"],
            "after": after["metrics"],
            "best_dev_loss": float(trainer.state.best_metric),
            "number_of_dev_queries": data["dev_queries"],
            "improvement_not_guaranteed": True,
        }
        (out / "mentor_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8")
        status.update({
            "status": "EXPLORATORY_EXPERIMENT_COMPLETED",
            "selected_checkpoint": Path(trainer.state.best_model_checkpoint).name,
            "best_dev_loss": float(trainer.state.best_metric),
            "selected_model_saved": True,
        })
        (out / "experiment_manifest.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"status": status["status"],
                      "output": str(out), "comparison": comparison}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("split-dir", "corpus", "families", "candidates", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--candidate-method", choices=("BM25", "Qwen"),
                   default="BM25")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=7e-7)
    p.add_argument("--i-confirm-authorized-environment", action="store_true")
    p.add_argument("--i-accept-provisional-labels", action="store_true")
    p.add_argument("--i-accept-dev-not-independent", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
