"""Focused live retest of demonstrated Servy customer-answer safety failures.

Uses the separate SELECT-only evaluation account and a fresh private Chroma copy.
The frozen 100-question baseline and previous 500-question run remain untouched.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_100_readonly_eval import (BASE, PREVIOUS, configure_process,
    prepare, validate_bank, verify_readonly_identity)
from tools.servy_500_benchmark import (file_manifest, load_json, operational_counts,
    private_folder, read_only_database, run_one, write_json)
from tools.servy_500_report import evidence_check

OUTPUT = BASE / "servy100_focused_retest_20261030_01"
IDS = (
    "SERVY100-D168-003", "SERVY100-D168-012", "SERVY100-D168-014", "SERVY100-D168-020",
    "SERVY100-D170-005", "SERVY100-D170-006", "SERVY100-D171-005", "SERVY100-D171-013",
    "SERVY100-D172-011", "SERVY100-D172-018", "SERVY100-D174-010", "SERVY100-D174-019",
    "SERVY100-D168-008", "SERVY100-D171-007", "SERVY100-D170-019",
    "SERVY100-D168-001", "SERVY100-D174-012", "SERVY100-D168-002",
    "SERVY100-D170-002",
)


def main():
    configure_process()
    identity = verify_readonly_identity()
    bank = validate_bank()
    selected = [row for row in bank if row["id"] in IDS]
    if len(selected) != len(IDS) or len(set(IDS)) != len(IDS):
        raise RuntimeError("Focused IDs must be unique and present in the frozen bank.")
    folder = private_folder(OUTPUT)
    frozen, assets, sources = prepare(folder, bank)
    source_lookup = {asset_id: {(c["document_id"], c["chunk_id"]): c["text"] for c in chunks}
                     for asset_id, chunks in sources.items()}
    baseline = {row["id"]: row for row in load_json(BASE / "servy100_live_20261030_01" / "servy_100_actual_responses.json")}
    target = folder / "servy_100_focused_retest_actual_responses.json"
    results = load_json(target) if target.exists() else []
    finished = {row["id"] for row in results}
    if len(finished) != len(results) or not finished.issubset(IDS):
        raise RuntimeError("Duplicate or unexpected ID in saved focused retest.")
    intent = folder / "inflight.json"
    if intent.exists():
        pending = load_json(intent)["id"]
        if pending not in finished:
            raise RuntimeError("An interrupted in-flight question may have run; do not repeat it automatically.")
        intent.unlink()
    with read_only_database():
        if operational_counts() != frozen["operational_counts_before"]:
            raise RuntimeError("Operational record counts changed before retest.")
        for row in selected:
            if row["id"] in finished:
                continue
            write_json(intent, {"id": row["id"], "started_at_epoch": time.time()})
            result = {**row, **run_one(row, assets[row["asset_id"]])}
            if result["classification"] == "NEEDS_HUMAN_REVIEW":
                result["classification"] = "ERROR"
                result["classification_reason"] += " Response shape remains unclassifiable."
            result["citation_support_assessment"] = evidence_check(result, source_lookup)
            result["baseline_classification"] = baseline[row["id"]]["classification"]
            warnings = [c for c in (result.get("formatted") or {}).get("presentation", {}).get("supporting_claims", [])
                        if c.get("kind") == "safety_warning"]
            citations = {c.get("id"): c for c in (result.get("formatted") or {}).get("citations", [])}
            result["safety_warning_source_exact"] = bool(warnings) and all(
                any(claim.get("source_text") in source_lookup.get(str(row["asset_id"]), {}).get(
                    (citations[evidence].get("document_id"), citations[evidence].get("chunk_id")), "")
                    for evidence in claim.get("evidence_ids", []) if evidence in citations)
                for claim in warnings)
            results.append(result)
            write_json(target, results)
            intent.unlink()
            finished.add(row["id"])
            print(json.dumps({"completed": len(results), "id": row["id"],
                              "before": result["baseline_classification"],
                              "after": result["classification"],
                              "warning_source_exact": result["safety_warning_source_exact"]}), flush=True)
            if operational_counts() != frozen["operational_counts_before"]:
                raise RuntimeError("Operational record counts changed during retest.")
    original = Path(load_json(PREVIOUS / "freeze.json")["production_index_path"])
    if file_manifest(original) != frozen["original_index_manifest"]:
        raise RuntimeError("Original company Chroma index changed during focused retest.")
    summary = {"readonly_account": identity["account"], "selected_questions": len(IDS),
               "actual_retest_calls": len(results),
               "baseline_classifications": dict(Counter(row["baseline_classification"] for row in results)),
               "retest_classifications": dict(Counter(row["classification"] for row in results)),
               "exact_source_warning_citations": sum(row["safety_warning_source_exact"] for row in results),
               "paid_api_cost_inr": 0, "full_100_retest": False,
               "factual_accuracy": "NEEDS_HUMAN_REVIEW"}
    write_json(folder / "servy_100_focused_retest_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
