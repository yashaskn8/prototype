"""Export an integrity-checked PRIVATE ZIP for the separate Servy V4 mentor demo.

Never commit or upload this archive to public GitHub. The resulting archive
contains private passages. Transfer only to an authorized GPU environment.
The data remains provisional and is NOT considered expert-approved.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from servy_v4_prepare_research_artifacts import freeze, sha256
from servy_v4_mentor_experiment import audited_inputs


NAMES = {
    "split/train_unapproved_candidate.jsonl": "train_unapproved_candidate.jsonl",
    "split/dev_unapproved_candidate.jsonl": "dev_unapproved_candidate.jsonl",
    "split/quarantined_negatives.jsonl": "quarantined_negatives.jsonl",
    "split/resplit_report.json": "resplit_report.json",
}


def pack(split_dir, corpus, families, destination, *,
         method="bm25", qwen_candidates=None):
    split = Path(split_dir).resolve()
    corpus = Path(corpus).resolve()
    families = Path(families).resolve()
    target = Path(destination).resolve()
    if target.suffix.lower() != ".zip" or "private_training_data" not in {
        part.casefold() for part in target.parts}:
        raise ValueError("Output must be a private_training_data .zip archive")
    if target.exists():
        raise FileExistsError("Archive already exists; no overwrite")
    if not target.parent.is_dir():
        raise FileNotFoundError("Destination directory not present")
    with tempfile.TemporaryDirectory(prefix="servy-mentor-stage-",
                                     dir=str(target.parent)) as temporary:
        prepared = Path(temporary) / "frozen"
        freeze(split / NAMES["split/dev_unapproved_candidate.jsonl"],
               corpus, prepared, method=method,
               retrieval_candidates=qwen_candidates)
        candidates = prepared / "frozen_candidates.jsonl"
        audit = audited_inputs(split, corpus, families, candidates)
        inputs = {
            **{name: split / file for name, file in NAMES.items()},
            "corpus.jsonl": corpus,
            "families.jsonl": families,
            "candidates.jsonl": candidates,
        }
        metadata = {
            "schema": "servy-v4-mentor-input-1",
            "research_only": True,
            "expert_approval_claimed": False,
            "corporate_approval_embedded": False,
            "candidate_method": "BM25" if method == "bm25" else "Qwen",
            "train_queries": audit["train_queries"],
            "dev_queries": audit["dev_queries"],
            "positive_and_negative_labels": "PROVISIONAL_WEAK_SUPERVISION",
            "source_isolation_verified_against_machine_proposals": True,
            "file_sha256": {name: sha256(path) for name, path in inputs.items()},
        }
        # ZIP is created at an exclusively opened path, never replacing data.
        with target.open("xb") as raw:
            with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive:
                for name, path in inputs.items():
                    archive.write(path, arcname=name)
                archive.writestr("manifest.json", json.dumps(metadata, indent=2))
        with zipfile.ZipFile(target, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise IOError("ZIP integrity error for " + bad)
        metadata["archive_path"] = str(target)
        metadata["archive_sha256"] = sha256(target)
        return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--families", required=True)
    p.add_argument("--out-zip", required=True)
    p.add_argument("--method", choices=("bm25", "qwen-export"), default="bm25")
    p.add_argument("--qwen-candidates")
    a = p.parse_args()
    print(json.dumps(pack(a.split_dir, a.corpus, a.families, a.out_zip,
                          method=a.method, qwen_candidates=a.qwen_candidates),
                     indent=2))


if __name__ == "__main__":
    main()
