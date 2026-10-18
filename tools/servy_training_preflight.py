"""Strict, model-free Servy V4 research admission gate.

Run BEFORE importing torch, loading CUDA or downloading any model. This tool
only checks locally supplied, company-authorized files. No production writes.
A READY result is *research* admission, never proof of human truth or safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from servy_finetune_preflight import audit as structural_audit, read_rows


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object expected")
    return value


def inspect_inputs(train, dev, corpus, model_config, authorization,
                   artifact_dir, stage2=None, holdout=None):
    """Machine-verifiable structural admission, not independent label adjudication."""
    paths = {"train": train, "dev": dev, "corpus": corpus}
    if stage2:
        paths["stage2"] = stage2
    if holdout:
        paths["holdout"] = holdout
    issues = []
    def issue(status, code, role):
        issues.append({"status": status, "code": code, "role": role})

    missing = [name for name, path in paths.items() if not Path(path).is_file()]
    for name in missing:
        issue("BLOCKED_MISSING_ARTIFACT", "INPUT_FILE_MISSING", name)
    if not Path(model_config).is_file():
        issue("BLOCKED_MISSING_ARTIFACT", "MODEL_CONFIG_MISSING", "model")
    if not Path(authorization).is_file():
        issue("BLOCKED_MISSING_ARTIFACT", "AUTHORIZATION_MANIFEST_MISSING", "approval")

    hashes = {key: sha256_file(path) for key, path in paths.items() if key not in missing}
    structure = None
    if not missing:
        try:
            structure = structural_audit(
                train, dev, corpus, stage2_path=stage2,
                holdout_path=holdout, artifact_dir=artifact_dir,
            )
            for finding in structure["findings"]:
                if finding["severity"] != "BLOCKER":
                    continue
                code = finding["code"]
                classification = ("BLOCKED_SPLIT_LEAKAGE"
                                  if code.startswith(("SPLIT_", "STAGE2_"))
                                  else "BLOCKED_DATA_INTEGRITY")
                issue(classification, code, finding["ref"])
            # A set-level collision warning is unsafe for in-batch contrastive
            # training unless a collision-free sampler is demonstrated.
            for code in ("MULTI_POSITIVE_SAME_QUESTION_IN_BATCH",
                         "SHARED_POSITIVE_ACROSS_QUESTIONS",
                         "REPEATED_QUERY_POSITIVE_PAIR"):
                if code in structure["issue_counts"]:
                    issue("BLOCKED_DATA_INTEGRITY", "UNSAFE_IN_BATCH_COLLISION", code)
        except (ValueError, OSError, KeyError, TypeError) as error:
            issue("BLOCKED_DATA_INTEGRITY", type(error).__name__, "structural_validation")

    config = {}
    if Path(model_config).is_file():
        try:
            config = _load_object(model_config)
            required = ("base_model_id", "base_revision",
                        "tokenizer_id", "tokenizer_revision",
                        "sentence_transformers_version", "loss_name",
                        "max_seq_length", "candidate_pool",
                        "query_format", "passage_format")
            if any(not config.get(key) for key in required):
                issue("BLOCKED_MODEL_COMPATIBILITY", "UNPINNED_MODEL_CONFIGURATION", "model")
            if not isinstance(config.get("max_seq_length"), int) or config.get("max_seq_length", 0) <= 0:
                issue("BLOCKED_MODEL_COMPATIBILITY", "INVALID_MAX_SEQ_LENGTH", "model")
            if not isinstance(config.get("candidate_pool"), int) or config.get("candidate_pool", 0) < 5:
                issue("BLOCKED_MODEL_COMPATIBILITY", "INVALID_CANDIDATE_POOL", "model")
            if config.get("loss_name") not in (
                "BinaryCrossEntropyLoss", "CachedMultipleNegativesRankingLoss"
            ):
                issue("BLOCKED_MODEL_COMPATIBILITY", "UNREVIEWED_LOSS", "model")
        except (ValueError, OSError, TypeError) as error:
            issue("BLOCKED_MODEL_COMPATIBILITY", type(error).__name__, "model")

    approved = {}
    if Path(authorization).is_file():
        try:
            approved = _load_object(authorization)
            authorized_hashes = approved.get("authorized_input_sha256")
            if not isinstance(authorized_hashes, dict):
                issue("BLOCKED_LABEL_VERIFICATION", "MISSING_AUTHORIZED_INPUT_HASHES", "approval")
            else:
                for name, digest in hashes.items():
                    if authorized_hashes.get(name) != digest:
                        issue("BLOCKED_LABEL_VERIFICATION", "INPUT_NOT_IN_APPROVED_EXPORT", name)
            if approved.get("approved_persistent_destination") != str(artifact_dir):
                issue("BLOCKED_MISSING_ARTIFACT", "CHECKPOINT_DESTINATION_NOT_APPROVED", "approval")
            if not approved.get("approval_reference"):
                issue("BLOCKED_LABEL_VERIFICATION", "NO_APPROVAL_REFERENCE", "approval")
            reviewers = approved.get("authorized_reviewer_ids", [])
            if not isinstance(reviewers, list) or not reviewers or not all(
                isinstance(x, str) and x.strip() for x in reviewers
            ):
                issue("BLOCKED_LABEL_VERIFICATION", "NO_REVIEWER_ALLOWLIST", "approval")
                reviewers = []
            if not missing:
                for role in ("train", "dev", "stage2"):
                    if role not in paths:
                        continue
                    for index, row in enumerate(read_rows(paths[role]), 1):
                        if row.get("review_status") != "approved" or row.get("reviewer_id") not in reviewers:
                            issue("BLOCKED_LABEL_VERIFICATION", "POSITIVE_REVIEW_UNATTESTED", f"{role}:{index}")
                        has_negative = bool(
                            row.get("negative_chunk_ids") or row.get("negative_texts")
                            or row.get("candidate_negative") or row.get("negative")
                            or row.get("negative_text")
                        )
                        if has_negative and (
                            row.get("negative_review_status") != "approved"
                            or row.get("negative_reviewer_id") not in reviewers
                        ):
                            issue("BLOCKED_LABEL_VERIFICATION", "NEGATIVE_REVIEW_UNATTESTED", f"{role}:{index}")
                for index, row in enumerate(read_rows(corpus), 1):
                    if not row.get("source_group") or not row.get("source_version"):
                        issue("BLOCKED_DATA_INTEGRITY", "MISSING_SOURCE_FAMILY_OR_VERSION", f"corpus:{index}")
        except (ValueError, OSError, TypeError) as error:
            issue("BLOCKED_LABEL_VERIFICATION", type(error).__name__, "approval")

    target = Path(artifact_dir)
    normalized = str(artifact_dir).replace("\\", "/")
    if not target.is_absolute() and not re.match(r"^[A-Za-z]:/", normalized):
        issue("BLOCKED_MISSING_ARTIFACT", "DESTINATION_NOT_ABSOLUTE", "artifacts")
    if normalized == "/content" or (normalized.startswith("/content/") and
                                     not normalized.startswith("/content/drive/")):
        issue("BLOCKED_MISSING_ARTIFACT", "DESTINATION_EPHEMERAL", "artifacts")
    if target.exists():
        issue("BLOCKED_MISSING_ARTIFACT", "DESTINATION_EXISTS_NO_OVERWRITE", "artifacts")

    priority = ("BLOCKED_MISSING_ARTIFACT", "BLOCKED_SPLIT_LEAKAGE",
                "BLOCKED_DATA_INTEGRITY", "BLOCKED_LABEL_VERIFICATION",
                "BLOCKED_MODEL_COMPATIBILITY")
    status = next((kind for kind in priority if any(x["status"] == kind for x in issues)),
                  "READY_FOR_RESEARCH_EXPERIMENT")
    return {
        "schema_version": "servy-v4-strict-1",
        "status": status,
        "purpose": "Research admission only; not production or accuracy certification",
        "training_performed": False,
        "input_sha256": hashes,
        "model_configuration_sha256": sha256_file(model_config) if Path(model_config).is_file() else None,
        "authorization_manifest_sha256": sha256_file(authorization) if Path(authorization).is_file() else None,
        "structural_issue_counts": structure["issue_counts"] if structure else {},
        "issues": issues[:500],
        "issues_truncated": len(issues) > 500,
        "limitations": [
            "Reviewer credentials and expertise are not cryptographically verified",
            "Semantic near-duplicates and incorrect expert judgments require manual review",
            "Model/loss API compatibility, tokenizer lengths and CUDA must be checked at runtime",
            "Storage manifest is not proof of corporate permission or live mount",
            "Independent answer correctness and source authorization remain unverified",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "dev", "corpus", "model_config", "authorization",
                 "artifact_dir", "report"):
        p.add_argument("--" + flag.replace("_", "-"), required=True)
    p.add_argument("--stage2")
    p.add_argument("--holdout")
    a = p.parse_args()
    try:
        report = inspect_inputs(a.train, a.dev, a.corpus, a.model_config,
                                a.authorization, a.artifact_dir, a.stage2, a.holdout)
    except (ValueError, OSError, TypeError, KeyError) as error:
        report = {"schema_version": "servy-v4-strict-1",
                  "status": "BLOCKED_DATA_INTEGRITY", "training_performed": False,
                  "error_type": type(error).__name__}
    output = Path(a.report)
    if output.exists():
        p.error("Report path exists: refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "issue_count": len(report.get("issues", [])),
                      "report": str(output)}))
    raise SystemExit(0 if report["status"] == "READY_FOR_RESEARCH_EXPERIMENT" else 2)


if __name__ == "__main__":
    main()
