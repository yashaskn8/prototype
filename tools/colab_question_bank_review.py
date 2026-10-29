"""Review a question-only Servy benchmark JSON in Colab or locally.

Does not access company databases, send network requests, run inference, or
treat hypothetical questions as verified ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def load_bank(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError("Expected a JSON object containing a 'questions' list")
    metadata = payload.get("metadata", {})
    questions = payload["questions"]
    ids = set()
    for n, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {n} is not an object")
        for key in ("id", "product", "module", "question", "question_sha256", "asset_id", "coverage_status"):
            if key not in item:
                raise ValueError(f"Question {n} has no {key}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate question ID: {item['id']}")
        ids.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Blank question: {item['id']}")
        actual = sha256(item["question"].encode("utf-8")).hexdigest()
        if item["question_sha256"] != actual:
            raise ValueError(f"Full-text checksum mismatch: {item['id']}")
    return metadata, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Locally uploaded question-only JSON")
    parser.add_argument("--asset-id", type=int, help="Filter to one asset (100 questions)")
    parser.add_argument("--module", choices=["installation_and_commissioning", "preventive_maintenance", "troubleshooting"])
    parser.add_argument("--export", type=Path, help="Export the filtered question-only JSON; excludes actual answers")
    parser.add_argument("--show", type=int, default=0, help="Show the first N full questions")
    args = parser.parse_args()

    metadata, rows = load_bank(args.input)
    totals = Counter((r["asset_id"], r["product"]) for r in rows)
    print(f"Validated {len(rows)} unique questions with complete SHA-256 checksums")
    for (asset_id, product), count in sorted(totals.items()):
        print(f"  {asset_id}: {product} — {count}")
    print("WARNING: Questions are unreviewed probes, not certified answerable cases or model accuracy results.")
    print("WARNING: Do not upload confidential manuals, answers, or company data to personal Colab.")
    print("WARNING: Do not run on an asset without explicit authorization and approved source handling.")

    selected = [r for r in rows if (args.asset_id is None or r["asset_id"] == args.asset_id)
                and (args.module is None or r["module"] == args.module)]
    print(f"Selected {len(selected)} questions")
    for item in selected[:max(0, args.show)]:
        print(f"\n{item['id']} [{item['module']}, {item['coverage_status']}]\n{item['question']}")
    if args.export:
        args.export.write_text(json.dumps({"metadata": metadata, "questions": selected},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Exported {len(selected)} questions to {args.export}")


if __name__ == "__main__":
    main()
