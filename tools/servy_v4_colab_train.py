"""Opt-in, research-only Colab BGE reranker experiment.

No GPU imports occur until strict offline admission, storage, and fixed-candidate
checks pass. Never use private corporate data on an unapproved personal account.
This script is NOT a production promotion workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

from servy_training_preflight import inspect_inputs, sha256_file, _load_object
from servy_finetune_preflight import read_rows, get_positive_text
from servy_checkpoint_audit import (
    load_jsonl, normalize_inputs, validate_candidates, evaluate_one,
)
from servy_checkpoint_backup import backup_checkpoint


def raw_negative_texts(row):
    """Preserve original technical capitalization, error codes and units."""
    values = []
    for key in ("candidate_negative", "negative_text", "negative"):
        if row.get(key):
            values.append(str(row[key]).strip())
    extra = row.get("negative_texts") or []
    if not isinstance(extra, list):
        raise ValueError("negative_texts must be a list")
    values.extend(str(value).strip() for value in extra if str(value).strip())
    return list(dict.fromkeys(values))


def make_bce_pairs(rows):
    """Preserve every reviewed positive; require truly reviewed negative labels."""
    examples = []
    for row in rows:
        q = str(row.get("query_text") or row.get("query") or "").strip()
        positive = str(get_positive_text(row)).strip()
        if not q or not positive or row.get("review_status") != "approved":
            raise ValueError("BCE source lacks approved query/positive")
        examples.append({"text1": q, "text2": positive, "label": 1.0})
        negatives = raw_negative_texts(row)
        if (row.get("negative_chunk_id") or row.get("negative_chunk_ids")
                or row.get("candidate_negative_chunk_id")) and not negatives:
            raise ValueError("Negative source IDs have no exported passage text")
        if negatives and row.get("negative_review_status") != "approved":
            raise ValueError("BCE negatives have not been approved")
        for negative in negatives:
            if " ".join(negative.casefold().split()) == " ".join(positive.casefold().split()):
                raise ValueError("A negative repeats the positive passage")
            examples.append({"text1": q, "text2": negative, "label": 0.0})
    if not examples or {p["label"] for p in examples} != {0.0, 1.0}:
        raise ValueError("BCE requires both approved positive and negative examples")
    if len({(p["text1"], p["text2"], p["label"]) for p in examples}) != len(examples):
        raise ValueError("Duplicate labeled pair: deduplicate and reassess before training")
    return examples


def validate_model_tokenizer_binding(config):
    """Require the exact tokenizer revision actually consumed by CrossEncoder."""
    if config.get("base_model_id") != "BAAI/bge-reranker-v2-m3":
        raise ValueError("Only the explicitly reviewed BGE reranker can be trained")
    if config.get("tokenizer_id") != config.get("base_model_id"):
        raise ValueError("Configured tokenizer ID differs from the CrossEncoder model")
    if config.get("tokenizer_revision") != config.get("base_revision"):
        raise ValueError("Configured tokenizer revision differs from actual model revision")


def candidate_gate(cases_path, corpus_path, candidates_path, approved, top_k=10):
    """Forbid gold injection or changing candidates after research comparison."""
    approved_hash = approved.get("authorized_input_sha256", {}).get("candidates")
    if approved_hash != sha256_file(candidates_path):
        raise ValueError("Frozen candidates differ from authorized input digest")
    cases, chunks = normalize_inputs(load_jsonl(cases_path), load_jsonl(corpus_path))
    candidates = validate_candidates(cases, chunks, load_jsonl(candidates_path), top_k)
    return cases, chunks, candidates


def check_backup_destination(path):
    path = Path(path)
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("Approved backup destination must be absolute with existing parent")
    if str(path).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Google Drive is not mounted; no durable backup")
    if str(path).startswith("/content/") and not str(path).startswith("/content/drive/"):
        raise RuntimeError("Refusing ephemeral-only checkpoint backup")
    if path.exists():
        raise FileExistsError("Experiment backup already exists: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("train", "dev", "corpus", "model_config", "authorization",
                "artifact_dir", "candidates", "run_id"):
        parser.add_argument("--" + opt.replace("_", "-"), required=True)
    parser.add_argument("--stage2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=7e-7)
    parser.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", a.run_id):
        parser.error("Unique run ID must contain 6-64 safe characters")
    if not (1 <= a.epochs <= 3 and 1 <= a.batch_size <= 8 and
            math.isfinite(a.learning_rate) and 0 < a.learning_rate <= 2e-5):
        parser.error("Research bounds: epochs 1-3, batch 1-8, lr >0 and <=2e-5")
    gate = inspect_inputs(a.train, a.dev, a.corpus, a.model_config, a.authorization,
                          a.artifact_dir, stage2=a.stage2)
    print("TRAINING ADMISSION:", gate["status"], flush=True)
    if gate["status"] != "READY_FOR_RESEARCH_EXPERIMENT":
        raise SystemExit("BLOCKED: inspect preflight locally. No GPU/model initialized.")
    approval = _load_object(a.authorization)
    config = _load_object(a.model_config)
    if config["loss_name"] != "BinaryCrossEntropyLoss":
        raise SystemExit("BLOCKED: this controlled experiment only supports verified BCE")
    try:
        validate_model_tokenizer_binding(config)
    except ValueError as exc:
        raise SystemExit(f"BLOCKED_MODEL_COMPATIBILITY: {exc}") from exc
    if a.artifact_dir != str(Path(a.artifact_dir).parent / a.run_id):
        raise SystemExit("BLOCKED: approved artifact directory must end with exact run ID")
    check_backup_destination(a.artifact_dir)

    cases, chunks, candidates = candidate_gate(
        a.dev, a.corpus, a.candidates, approval,
        top_k=int(config["candidate_pool"]),
    )
    pairs_train = make_bce_pairs(read_rows(a.train))
    pairs_dev = make_bce_pairs(read_rows(a.dev))
    if a.stage2:
        pairs_train.extend(make_bce_pairs(read_rows(a.stage2)))
        if len({(p["text1"], p["text2"], p["label"]) for p in pairs_train}) != len(pairs_train):
            raise SystemExit("BLOCKED: Stage 2 repeats a training example")
    if version("sentence-transformers") != config["sentence_transformers_version"]:
        raise SystemExit("BLOCKED: installed Sentence Transformers version does not match pin")

    # Only now import heavyweight ML libraries.
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    if not torch.cuda.is_available():
        raise SystemExit("BLOCKED: CUDA unavailable; no CPU fallback for this Colab run")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bf16 = bool(torch.cuda.is_bf16_supported())
    print("GPU:", torch.cuda.get_device_name(0), "BF16:", bf16, flush=True)

    target = Path(a.artifact_dir)
    target.mkdir(mode=0o700)  # After all model-free checks; immutable run destination.
    with tempfile.TemporaryDirectory(prefix="servy-v4-") as temp:
        work = Path(temp)
        model = CrossEncoder(
            config["base_model_id"], revision=config["base_revision"],
            num_labels=1, max_length=int(config["max_seq_length"]), device="cuda",
        )
        # Identity returns raw single-label logits; sigmoid would still preserve
        # ordering but obscure checkpoint-to-checkpoint score comparisons.
        def predict(pairs):
            return model.predict(pairs, activation_fn=torch.nn.Identity(),
                                 show_progress_bar=False).tolist()
        baseline = evaluate_one(
            cases, chunks, candidates, predict,
            pool_size=min(5, int(config["candidate_pool"]))
        )
        (target / "baseline.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8")

        class BackupEachCheckpoint(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if not checkpoint.is_dir():
                    raise RuntimeError("Expected epoch checkpoint is missing")
                backup_checkpoint(checkpoint, target / checkpoint.name)
                return control

        training_args = CrossEncoderTrainingArguments(
            output_dir=str(work / "checkpoints"),
            num_train_epochs=a.epochs,
            per_device_train_batch_size=a.batch_size,
            per_device_eval_batch_size=a.batch_size,
            learning_rate=a.learning_rate, warmup_ratio=0.05,
            seed=a.seed, data_seed=a.seed, report_to="none",
            save_strategy="epoch", eval_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=3,
            logging_steps=1, bf16=bf16, fp16=False,
            dataloader_num_workers=0, push_to_hub=False,
        )
        trainer = CrossEncoderTrainer(
            model=model, args=training_args,
            train_dataset=Dataset.from_list(pairs_train),
            eval_dataset=Dataset.from_list(pairs_dev),
            loss=BinaryCrossEntropyLoss(model),
            callbacks=[BackupEachCheckpoint()],
        )
        trainer.train()
        if not trainer.state.best_model_checkpoint or not math.isfinite(
            float(trainer.state.best_metric)
        ):
            raise RuntimeError("No finite, selected best validation checkpoint")
        best = work / "best_reload"
        trainer.model.save_pretrained(str(best))
        backup_checkpoint(best, target / "best_model")
        reloaded = CrossEncoder(str(best), device="cuda",
                                max_length=int(config["max_seq_length"]))
        def predict_reload(pairs):
            return reloaded.predict(pairs, activation_fn=torch.nn.Identity(),
                                    show_progress_bar=False).tolist()
        probe = [["equipment question", "approved test passage"]]
        raw = predict(probe)[0]
        loaded = predict_reload(probe)[0]
        if not math.isfinite(raw) or not math.isfinite(loaded) or abs(raw-loaded) > 1e-4:
            raise RuntimeError("Selected model reloaded with changed logits")
        after = evaluate_one(
            cases, chunks, candidates, predict_reload,
            pool_size=min(5, int(config["candidate_pool"]))
        )
        (target / "comparison.json").write_text(
            json.dumps({"before": baseline["metrics"], "after": after["metrics"],
                        "dev_reused_not_independent_holdout": True},
                       indent=2), encoding="utf-8")
        (target / "per_query_selected.json").write_text(
            json.dumps(after["queries"], indent=2), encoding="utf-8")
        manifest = {
            "run_id": a.run_id, "research_only": True, "promotion_approved": False,
            "input_sha256": gate["input_sha256"],
            "frozen_candidates_sha256": sha256_file(a.candidates),
            "model_config_sha256": sha256_file(a.model_config),
            "best_validation_metric": trainer.state.best_metric,
            "best_checkpoint_name": Path(trainer.state.best_model_checkpoint).name,
            "save_and_reload_verified": True,
            "checkpoint_directory_hashes": {
                p.name: {
                    name: digest for name, digest in
                    backup_checkpoint_manifest(p).items()
                } for p in target.iterdir() if p.is_dir()
            },
        }
        (target / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    print("RESEARCH RUN FINISHED; VERIFY EVERY BACKUP BEFORE ENDING RUNTIME:", target)


def backup_checkpoint_manifest(path):
    from servy_checkpoint_backup import collect, sha256
    return {name: sha256(item) for name, item in collect(path)}


if __name__ == "__main__":
    main()
