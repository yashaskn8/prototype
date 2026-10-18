"""Synthetic-only Colab GPU compatibility test for Servy V4 CrossEncoder.

Uses only harmless generated strings and a tiny PUBLIC test model. Never
loads Servy datasets, authorizations, databases, production checkpoints or
personal cloud storage. It checks runtime APIs, GPU train/eval, checkpoint
selection and save/reload without ever claiming domain model readiness.

Example in an explicitly selected Colab GPU runtime:
  python tools/servy_v4_colab_smoke.py --expected-version 5.1.0
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from importlib.metadata import version
from pathlib import Path


PUBLIC_TINY_MODEL = "hf-internal-testing/tiny-random-bert"


def synthetic_pairs():
    """Two explicit BCE classes, no proprietary passage information."""
    train = [
        {"text1": "What is the color of the sky?", "text2": "The sky is blue.", "label": 1.0},
        {"text1": "What is the color of the sky?", "text2": "Water freezes at zero.", "label": 0.0},
        {"text1": "What is an apple?", "text2": "An apple is a fruit.", "label": 1.0},
        {"text1": "What is an apple?", "text2": "The keyboard has letters.", "label": 0.0},
        {"text1": "How many sides has a triangle?", "text2": "A triangle has three sides.", "label": 1.0},
        {"text1": "How many sides has a triangle?", "text2": "Winter can be cold.", "label": 0.0},
    ]
    dev = [
        {"text1": "What do birds use to fly?", "text2": "Birds use wings.", "label": 1.0},
        {"text1": "What do birds use to fly?", "text2": "A chair is furniture.", "label": 0.0},
    ]
    return train, dev


def validate_pairs(train, dev):
    for split in (train, dev):
        if not split or {row.get("label") for row in split} != {0.0, 1.0}:
            raise ValueError("Synthetic BCE train/dev require both relevance classes")
        for row in split:
            if set(row) != {"text1", "text2", "label"}:
                raise ValueError("Wrong CrossEncoder dataset columns")
            if not isinstance(row["text1"], str) or not isinstance(row["text2"], str):
                raise ValueError("Invalid synthetic sentence pair")


def smoke(expected_version, require_gpu=True):
    train, dev = synthetic_pairs()
    validate_pairs(train, dev)
    actual_version = version("sentence-transformers")
    if expected_version and actual_version != expected_version:
        raise RuntimeError("Installed Sentence Transformers version does not match requested pin")
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Colab compatibility test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="servy-v4-public-smoke-") as tmp:
        model = CrossEncoder(PUBLIC_TINY_MODEL, num_labels=1, max_length=64, device=device)
        args = CrossEncoderTrainingArguments(
            output_dir=str(Path(tmp) / "checkpoints"),
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-5,
            eval_strategy="steps", eval_steps=1,
            save_strategy="steps", save_steps=1,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=2,
            logging_steps=1, report_to="none", push_to_hub=False,
            dataloader_num_workers=0, bf16=False, fp16=False,
        )
        trainer = CrossEncoderTrainer(
            model=model, args=args,
            train_dataset=Dataset.from_list(train),
            eval_dataset=Dataset.from_list(dev),
            loss=BinaryCrossEntropyLoss(model),
        )
        trainer.train()
        if not trainer.state.best_model_checkpoint:
            raise RuntimeError("No saved best validation checkpoint")
        if trainer.state.best_metric is None or not math.isfinite(float(trainer.state.best_metric)):
            raise RuntimeError("No finite selected development loss")
        best = Path(tmp) / "selected_model"
        trainer.model.save_pretrained(str(best))
        loaded = CrossEncoder(str(best), max_length=64, device=device)
        probe = [["What is the color of the sky?", "The sky is blue."]]
        score = float(loaded.predict(probe, activation_fn=torch.nn.Identity(),
                                     show_progress_bar=False)[0])
        if not math.isfinite(score):
            raise RuntimeError("Reloaded model produced non-finite output")
        return {
            "status": "SYNTHETIC_GPU_SMOKE_PASSED" if device == "cuda" else "SYNTHETIC_CPU_SMOKE_PASSED",
            "sentence_transformers_version": actual_version,
            "torch_version": torch.__version__,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "tiny_public_model": PUBLIC_TINY_MODEL,
            "training_steps": int(trainer.state.global_step),
            "best_eval_loss": float(trainer.state.best_metric),
            "checkpoint_reload_finite": True,
            "private_data_processed": False,
            "real_servy_training_authorized": False,
            "real_servy_training_performed": False,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-version", required=True,
                   help="Exact version pinned in the separately reviewed private V4 model config")
    args = p.parse_args()
    print(json.dumps(smoke(args.expected_version), indent=2))


if __name__ == "__main__":
    main()
