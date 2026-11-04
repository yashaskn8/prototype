"""Combine actual local Servy 500 responses into the three requested private reports.

Exact source quotations are mechanical evidence checks, not technical ground truth.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_500_benchmark import load_json, private_folder, write_json


def evidence_check(record, source_lookup):
    label = record["classification"]
    if label in ("NOT_RUN_UNAUTHORIZED", "NOT_RUN_COVERAGE_GAP"):
        return {"status": "NOT_APPLICABLE", "technical_correctness": "NOT_VERIFIED"}
    if label == "ERROR":
        return {"status": "NOT_VERIFIED", "technical_correctness": "NOT_VERIFIED"}
    raw = record.get("raw") or {}
    formatted = record.get("formatted") or {}
    source = source_lookup.get(str(record["asset_id"]), {})
    raw_refs = raw.get("references") or []
    citations = formatted.get("citations") or []
    sections = (formatted.get("presentation") or {}).get("sections") or []
    claims = [claim for section in sections for claim in section.get("claims", [])
              if claim.get("kind") not in ("uncertainty", "next_step")]
    cited = {c.get("id"): c for c in citations}
    raw_in_scope = all((r.get("document_id"), r.get("chunk_id")) in source for r in raw_refs)
    citations_in_scope = all((c.get("document_id"), c.get("chunk_id")) in source for c in citations)
    claims_exact = all(claim.get("source_text") and any(
        claim["source_text"] in source.get((cited[e].get("document_id"), cited[e].get("chunk_id")), "")
        for e in claim.get("evidence_ids", []) if e in cited) for claim in claims)
    if label in ("REFUSED", "CLARIFICATION"):
        return {"status": "NO_PUBLISHED_TECHNICAL_CLAIMS", "raw_refs_in_scope": raw_in_scope,
                "eligible_chunk_count": len(source),
                "refusal_due_no_company_documents": label == "REFUSED" and len(source) == 0,
                "technical_correctness": "NOT_VERIFIED"}
    if label in ("ANSWERED", "PARTIAL"):
        mechanical = bool(claims and citations and raw_in_scope and citations_in_scope and claims_exact)
        points = {str(n): {"present": bool(section := next((s for s in sections if s.get("number") == n), None)),
                           "claim_count": len((section or {}).get("claims", [])),
                           "semantic_quality": "NOT_VERIFIED"}
                  for n in range(1, 5)}
        return {"status": "SOURCE_QUOTES_EXACT_SEMANTICS_NOT_VERIFIED" if mechanical else "CITATION_OR_SOURCE_CHECK_FAILED",
                "raw_refs_in_scope": raw_in_scope, "citations_in_scope": citations_in_scope,
                "claim_source_quotes_exact": claims_exact, "technical_claims_checked": len(claims),
                "citation_count": len(citations), "four_point_structure_present": all(p["present"] for p in points.values()),
                "four_points": points, "technical_correctness": "NOT_VERIFIED"}
    return {"status": "NEEDS_HUMAN_REVIEW", "technical_correctness": "NOT_VERIFIED"}


def create_reports(folder):
    frozen = load_json(folder / "freeze.json")
    bank = load_json(folder / "servy_500_corrected_questions_v1.json")
    questions = bank["questions"]
    split = load_json(folder / "split.json")
    source_rows = load_json(folder / "authorized_source_snapshot.json")
    sources = {key: {(c["document_id"], c["chunk_id"]): c["text"] for c in rows}
               for key, rows in source_rows.items()}
    dev = load_json(folder / "servy_500_baseline_actual_test_results.json")
    holdout = load_json(folder / "servy_500_holdout_actual_test_results.json")
    if len(dev) != len(split["development"]) or len(holdout) != len(split["holdout"]):
        raise RuntimeError("Both frozen splits must finish before making a complete report.")
    actual = dev + holdout
    by_id = {r["id"]: r for r in actual}
    if len(by_id) != 500 or set(by_id) != {q["id"] for q in questions}:
        raise RuntimeError("Actual results must contain each original question exactly once.")
    records = []
    by_asset = defaultdict(Counter)
    by_intent = defaultdict(Counter)
    by_module = defaultdict(Counter)
    for q in questions:
        row = {**q, **by_id[q["id"]]}
        row["split"] = "development" if q["id"] in split["development"] else "holdout"
        row["evidence_assessment"] = evidence_check(row, sources)
        records.append(row)
        label = row["classification"]
        by_asset[str(q["asset_id"])][label] += 1
        by_intent[q["intent"]][label] += 1
        by_module[q["module"]][label] += 1
    counts = Counter(r["classification"] for r in records)
    executed = 500 - counts["NOT_RUN_UNAUTHORIZED"] - counts["NOT_RUN_COVERAGE_GAP"]
    documented = [r for r in records if r["classification"] != "NOT_RUN_UNAUTHORIZED"
                  and bool(source_rows.get(str(r["asset_id"]), []))]
    undocumented = [r for r in records if r["classification"] != "NOT_RUN_UNAUTHORIZED"
                    and not source_rows.get(str(r["asset_id"]), [])]
    evidence_counts = Counter(r["evidence_assessment"]["status"] for r in records)
    summary = {
        "total_questions": 500, "executed_questions": executed,
        "not_run_unauthorized": counts["NOT_RUN_UNAUTHORIZED"],
        "not_run_coverage_gap": counts["NOT_RUN_COVERAGE_GAP"],
        "classification_counts": dict(counts), "rate_denominator": "executed_questions_only",
        "answer_rate": counts["ANSWERED"] / executed,
        "refusal_rate": counts["REFUSED"] / executed,
        "outcome_ids": {label: [r["id"] for r in records if r["classification"] == label]
                        for label in ("ANSWERED", "REFUSED", "PARTIAL", "CLARIFICATION", "ERROR", "NOT_RUN_UNAUTHORIZED", "NOT_RUN_COVERAGE_GAP", "NEEDS_HUMAN_REVIEW")},
        "by_asset": {key: dict(value) for key, value in by_asset.items()},
        "by_intent": {key: dict(value) for key, value in by_intent.items()},
        "by_module": {key: dict(value) for key, value in by_module.items()},
        "evidence_check_counts": dict(evidence_counts),
        "answered_four_point_structure_count": sum(
            bool(r["evidence_assessment"].get("four_point_structure_present")) for r in records),
        "refusals_on_assets_with_no_eligible_documents": sum(
            bool(r["evidence_assessment"].get("refusal_due_no_company_documents")) for r in records),
        "documented_asset_question_count": len(documented),
        "undocumented_asset_coverage_gap_count": len(undocumented),
        "documented_asset_outcomes": dict(Counter(r["classification"] for r in documented)),
        "undocumented_asset_outcomes": dict(Counter(r["classification"] for r in undocumented)),
        "answerable_question_accuracy_denominator": None,
        "answerable_question_accuracy": None,
        "semantic_answer_accuracy": None, "citation_semantic_precision": None,
        "confirmed_unsupported_answer_count": None,
        "confirmed_inappropriate_refusal_count": None,
        "expert_review_status": "PENDING_REVIEW",
        "serving": frozen["serving"],
        "limitations": ["Direct ask_rag plus unsaved customer formatter, not authenticated HTTP portal testing.",
                        "Exact in-scope source quote matching does not prove semantic relevance, completeness, or customer action approval.",
                        "Questions and source review candidates have no human-approved answer keys.",
                        "Holdout was recorded separately; its outcomes must not guide code changes."],
    }
    write_json(folder / "servy_500_actual_test_results.json", records)
    write_json(folder / "servy_500_test_summary.json", summary)
    lines = ["# Servy 500 actual local chatbot baseline", "",
             f"Executed: {executed}/500. Unauthorized: {counts['NOT_RUN_UNAUTHORIZED']}.", "",
             "| Outcome | Count |", "|---|---:|"]
    for label in ("ANSWERED", "REFUSED", "PARTIAL", "CLARIFICATION", "ERROR", "NEEDS_HUMAN_REVIEW", "NOT_RUN_UNAUTHORIZED", "NOT_RUN_COVERAGE_GAP"):
        lines.append(f"| {label} | {counts[label]} |")
    lines += ["", "## Evidence limits", "",
              f"Documented-asset questions: {len(documented)}. Undocumented-asset coverage-gap questions: {len(undocumented)}. Neither count is an expert-approved answerable denominator.",
              f"Exact in-scope source quotes for answered/partial records: {evidence_counts['SOURCE_QUOTES_EXACT_SEMANTICS_NOT_VERIFIED']}.",
              f"Refusals with no eligible company documents: {summary['refusals_on_assets_with_no_eligible_documents']}.",
              "Technical correctness, citation semantics, appropriate refusal for documented assets, and customer action safety require human review.",
              "", "## Outcomes by asset", "", "| Asset | Answered | Refused | Partial | Clarification | Error | Unauthorized |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for asset_id, labels in sorted(by_asset.items(), key=lambda x: int(x[0])):
        lines.append(f"| {asset_id} | {labels['ANSWERED']} | {labels['REFUSED']} | {labels['PARTIAL']} | "
                     f"{labels['CLARIFICATION']} | {labels['ERROR']} | {labels['NOT_RUN_UNAUTHORIZED']} |")
    lines += ["", "## Outcome IDs", ""]
    for label, ids in summary["outcome_ids"].items():
        lines += [f"### {label}", "", ", ".join(ids) if ids else "None.", ""]
    lines += ["## Confirmed source gaps", "",
              "Asset 64 has no verified customer binding. Its 25 questions were not sent.",
              "Assets 62, 65, 66, 67, 68, 69, and 70 have zero local company documents and zero upstream attachments; do not treat these as retrieval-ranking failures.",
              "Only assets 59, 60, and 61 have locally eligible company technical documents.",
              "", "## Reproduction", "",
              "Run tools/servy_500_benchmark.py dry-run, baseline, and holdout with the original JSON and a private LocalAppData run directory; then run tools/servy_500_report.py.",
              "Do not use this report as human-approved technical ground truth.", ""]
    (folder / "servy_500_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def create_final_summary(folder):
    """Summarize saved final responses without issuing another chatbot request."""
    before = load_json(folder / "servy_500_baseline_actual_test_results.json") + load_json(
        folder / "servy_500_holdout_actual_test_results.json")
    after = load_json(folder / "servy_500_final_actual_test_results.json") + load_json(
        folder / "servy_500_holdout-final_actual_test_results.json")
    old_ids, new_ids = {r["id"] for r in before}, {r["id"] for r in after}
    if len(before) != 500 or len(after) != 500 or len(old_ids) != 500 or old_ids != new_ids:
        raise RuntimeError("All 500 frozen IDs must be present exactly once in both runs.")
    sources = load_json(folder / "final_source_snapshot.json")
    source_map = {key: {(c["document_id"], c["chunk_id"]): c["text"] for c in chunks}
                  for key, chunks in sources.items()}
    checks = {r["id"]: evidence_check(r, source_map) for r in after}
    before_by_id = {r["id"]: r for r in before}
    undocumented = [r for r in after if r["classification"] != "NOT_RUN_UNAUTHORIZED"
                    and not sources.get(str(r["asset_id"]))]
    documented = [r for r in after if r["classification"] != "NOT_RUN_UNAUTHORIZED"
                  and sources.get(str(r["asset_id"]))]
    summary = {
        "total_questions": 500,
        "executed_questions": sum(not r["classification"].startswith("NOT_RUN") for r in after),
        "classification_before": dict(Counter(r["classification"] for r in before)),
        "classification_after": dict(Counter(r["classification"] for r in after)),
        "transitions": dict(Counter(f"{before_by_id[r['id']]['classification']} -> {r['classification']}" for r in after)),
        "documented_asset_question_count": len(documented),
        "documented_asset_outcomes": dict(Counter(r["classification"] for r in documented)),
        "undocumented_asset_coverage_gap_count": len(undocumented),
        "undocumented_asset_outcomes": dict(Counter(r["classification"] for r in undocumented)),
        "no_documentation_reason_count": sum(
            (r.get("raw") or {}).get("reason_code") == "NO_AUTHORIZED_ORIGINAL_DOCUMENTATION"
            for r in undocumented),
        "citation_check_counts": dict(Counter(c["status"] for c in checks.values())),
        "answered_four_point_structure_count": sum(bool(c.get("four_point_structure_present")) for c in checks.values()),
        "answerable_question_accuracy_denominator": None,
        "confirmed_factual_answer_accuracy": None,
        "confirmed_inappropriate_refusal_count": None,
        "expert_review_status": "PENDING_REVIEW",
        "limitations": ["Exact source quotation and four-point structure do not establish technical correctness or completeness.",
                        "The 300 documented-asset questions are not all necessarily answerable.",
                        "No-document refusals are coverage and abstention outcomes, not answerable accuracy failures."],
    }
    write_json(folder / "servy_500_final_combined_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--final", action="store_true", help="Summarize saved final responses without new queries.")
    args = parser.parse_args()
    folder = private_folder(args.data_dir)
    if args.final:
        summary = create_final_summary(folder)
        print({"executed": summary["executed_questions"], "counts": summary["classification_after"],
               "evidence_checks": summary["citation_check_counts"]})
    else:
        summary = create_reports(folder)
        print({"executed": summary["executed_questions"], "counts": summary["classification_counts"],
               "evidence_checks": summary["evidence_check_counts"]})


if __name__ == "__main__":
    main()
