"""Private, read-only Servy V5 artifact audit for the four known Drive files.

Run from repository root:
    python tools/servy_v5_artifact_audit.py --experiment-root "/content/drive/MyDrive/servy_experiments/servy_v4_bge_20261018T193940Z"

This tool never downloads checkpoints, edits labels, uploads private text or
treats the reused 42-question development set as an independent holdout.
A structural pass is NOT permission to begin production fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class PrivateArtifactError(ValueError):
    """A blocking provenance, data-shape or selection-integrity failure."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise PrivateArtifactError("MISSING_ARTIFACT:" + path.name)
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrivateArtifactError(
                    f"INVALID_JSONL:{path.name}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PrivateArtifactError(f"NON_OBJECT_ROW:{path.name}:{line_number}")
            rows.append(row)
    if not rows:
        raise PrivateArtifactError("EMPTY_ARTIFACT:" + path.name)
    return rows


def _identity(row: dict) -> tuple[str, str]:
    tenant = str(row.get("tenant_id") or "").strip()
    query = str(row.get("query_id") or "").strip()
    if not tenant or not query:
        raise PrivateArtifactError("MISSING_TENANT_OR_QUERY_ID")
    return tenant, query


def _refs(row: dict, role: str) -> list[str]:
    values = row.get(role + "_chunk_ids")
    if values is None:
        values = row.get(role + "_chunk_id")
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [str(x).strip() for x in values if x is not None and str(x).strip()]


def validate_selected_subset(original: list[dict], selected: list[dict]) -> dict:
    """Require exact original rows; never accept AI-rewritten labels or passages."""
    if len(selected) > len(original) or not selected:
        raise PrivateArtifactError("INVALID_SUBSET_SIZE")
    by_identity = {}
    for row in original:
        key = _identity(row)
        if key in by_identity:
            raise PrivateArtifactError("DUPLICATE_ORIGINAL_QUERY_ID")
        by_identity[key] = row

    selected_keys = set()
    positive_pairs = negative_pairs = 0
    for row in selected:
        key = _identity(row)
        if key in selected_keys:
            raise PrivateArtifactError("DUPLICATE_SELECTED_QUERY_ID")
        selected_keys.add(key)
        source = by_identity.get(key)
        if source is None or source != row:
            raise PrivateArtifactError("SELECTED_ROW_MODIFIED_OR_NOT_IN_ORIGINAL")
        positive = _refs(row, "positive")
        negative = _refs(row, "negative")
        quarantine = _refs(row, "quarantined_negative")
        if not positive or not negative:
            raise PrivateArtifactError("EACH_SELECTED_QUERY_NEEDS_POSITIVE_AND_ACTIVE_NEGATIVE")
        if len(set(positive)) != len(positive) or len(set(negative)) != len(negative):
            raise PrivateArtifactError("DUPLICATE_LABEL_REFERENCE")
        if set(positive) & set(negative):
            raise PrivateArtifactError("CONTRADICTORY_POSITIVE_NEGATIVE_LABELS")
        if set(negative) & set(quarantine):
            raise PrivateArtifactError("QUARANTINED_NEGATIVE_STILL_ACTIVE")
        positive_pairs += len(positive)
        negative_pairs += len(negative)
    return {
        "original_queries": len(original),
        "selected_queries": len(selected),
        "excluded_queries": len(original) - len(selected),
        "positive_pairs": positive_pairs,
        "active_negative_pairs": negative_pairs,
        "exact_original_row_match": True,
        "expert_labels_approved": False,
    }


def validate_candidate_files(qwen: list[dict], hybrid: list[dict], dev=None) -> dict:
    """Check candidate format and query alignment, never insert missing gold IDs."""
    def parse(rows, label):
        result = {}
        for row in rows:
            qid = str(row.get("query_id") or "").strip()
            if not qid or qid in result:
                raise PrivateArtifactError("MISSING_OR_DUPLICATE_CANDIDATE_QUERY:" + label)
            ids = row.get("chunk_ids")
            if ids is None:
                ids = row.get("candidate_chunk_ids")
            if not isinstance(ids, list) or not 1 <= len(ids) <= 10:
                raise PrivateArtifactError("INVALID_TOP10_CANDIDATE_COUNT:" + label)
            normalized = [str(x).strip() for x in ids]
            if any(not x or x.lower() == "none" for x in normalized):
                raise PrivateArtifactError("EMPTY_CANDIDATE_ID:" + label)
            if len(normalized) != len(set(normalized)):
                raise PrivateArtifactError("DUPLICATE_CANDIDATE_CHUNK_ID:" + label)
            result[qid] = normalized
        return result

    a, b = parse(qwen, "qwen3"), parse(hybrid, "hybrid_rrf60")
    if set(a) != set(b):
        raise PrivateArtifactError("CANDIDATE_QUERY_SETS_DO_NOT_MATCH")
    if dev is not None:
        dev_ids = [str(row.get("query_id") or "").strip() for row in dev]
        if len(set(dev_ids)) != len(dev_ids) or set(dev_ids) != set(a):
            raise PrivateArtifactError("DEV_AND_CANDIDATE_QUERY_IDS_DO_NOT_MATCH")
    return {
        "dev_query_count": len(a),
        "all_query_ids_match": True,
        "identical_top1": sum(a[q][0] == b[q][0] for q in a),
        "identical_top10_order": sum(a[q] == b[q] for q in a),
        "different_ranked_candidates": sum(a[q] != b[q] for q in a),
        "gold_coverage_verified": False,
        "tenant_authorization_verified": False,
    }


def _single_path(paths, name: str) -> Path:
    paths = sorted(paths)
    if not paths:
        raise PrivateArtifactError("MISSING_ARTIFACT:" + name)
    if len(paths) != 1:
        raise PrivateArtifactError("AMBIGUOUS_ARTIFACT:" + name)
    return paths[0]


def discover(root: Path) -> dict[str, Path]:
    """Use exact observed V5 folder names; refuse ambiguous candidate snapshots."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise PrivateArtifactError("EXPERIMENT_ROOT_NOT_FOUND")
    split = root / "v5_gemini_semantic_audit_v1" / "input_snapshot" / "split"
    return {
        "original_train": split / "train_unapproved_candidate.jsonl",
        "selected_train": _single_path(
            root.glob("v5_training_candidates_*/train_keep_ai_provisional.jsonl"),
            "train_keep_ai_provisional.jsonl",
        ),
        "qwen3_candidates": root / "qwen3_candidates_top10.jsonl",
        "hybrid_candidates": root / "hybrid_rrf60_candidates_top10.jsonl",
        "dev": split / "dev_unapproved_candidate.jsonl",
        "resplit_report": split / "resplit_report.json",
        "quarantine": split / "quarantined_negatives.jsonl",
    }


def audit_experiment(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    files = discover(root)
    required = ("original_train", "selected_train", "qwen3_candidates", "hybrid_candidates")
    for name in required:
        if not files[name].is_file():
            raise PrivateArtifactError("MISSING_ARTIFACT:" + name)
    train = validate_selected_subset(
        read_jsonl(files["original_train"]),
        read_jsonl(files["selected_train"]),
    )
    dev = read_jsonl(files["dev"]) if files["dev"].is_file() else None
    candidates = validate_candidate_files(
        read_jsonl(files["qwen3_candidates"]),
        read_jsonl(files["hybrid_candidates"]),
        dev,
    )
    # If a signed-by-content resplit report is available, check its frozen
    # original training digest. Never replace the report with self-signed hashes.
    split_hash_verified = False
    if files["resplit_report"].is_file():
        try:
            report = json.loads(files["resplit_report"].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PrivateArtifactError("INVALID_RESPLIT_REPORT") from exc
        expected = report.get("output_sha256", {}).get(
            "train_unapproved_candidate.jsonl"
        )
        if not expected or expected != file_sha256(files["original_train"]):
            raise PrivateArtifactError("RESPLIT_ORIGINAL_TRAIN_SHA256_MISMATCH")
        split_hash_verified = True

    # Report only file names, counts and hashes; never print source questions.
    return {
        "status": "STRUCTURAL_AUDIT_PASSED_RESEARCH_ONLY",
        "training_authorized": False,
        "fine_tuning_admitted": False,
        "production_promotion_allowed": False,
        "independent_holdout": False,
        "source_family_split_verified": False,
        "split_original_train_hash_verified": split_hash_verified,
        "training": train,
        "candidate_comparison": candidates,
        "files": {
            name: {
                "relative_path": str(path.relative_to(root)),
                "present": path.is_file(),
                "sha256": file_sha256(path) if path.is_file() else None,
            } for name, path in files.items()
        },
        "next_required": [
            "Verify full corpus and source-family recommendation exports",
            "Run strict servy_v5_training_preflight.py against signed split and candidate files",
            "Independently review selected positive and hard-negative labels",
            "Evaluate on an untouched independent holdout before any promotion",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument(
        "--out", type=Path,
        help="Optional private JSON report path; refuses overwrite",
    )
    args = p.parse_args()
    try:
        result = audit_experiment(args.experiment_root)
        output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.out:
            dest = args.out.expanduser().resolve()
            if not dest.parent.is_dir():
                raise PrivateArtifactError("REPORT_PARENT_MISSING")
            with dest.open("x", encoding="utf-8") as stream:
                stream.write(output)
        print(output, end="")
    except (OSError, PrivateArtifactError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
