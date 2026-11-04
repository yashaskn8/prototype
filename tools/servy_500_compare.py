"""Private before/after comparison of actual Servy development responses."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.servy_500_benchmark import load_json, private_folder, write_json
from tools.servy_500_report import evidence_check


def compare(folder, after_phase="retest", split="development"):
    before = load_json(folder / f"servy_500_{'baseline' if split == 'development' else 'holdout'}_actual_test_results.json")
    after = load_json(folder / f"servy_500_{after_phase}_actual_test_results.json")
    expected = 400 if split == "development" else 100
    if len(before) != expected or len(after) != expected:
        raise RuntimeError("Both frozen runs must complete before comparison.")
    by_id_before = {row["id"]: row for row in before}
    by_id_after = {row["id"]: row for row in after}
    if len(by_id_before) != expected or by_id_before.keys() != by_id_after.keys():
        raise RuntimeError("Question IDs differ or repeat between runs.")
    old_sources = load_json(folder / "authorized_source_snapshot.json")
    new_sources = load_json(folder / f"{'final' if after_phase.endswith('final') else after_phase}_source_snapshot.json")
    source_maps = [{key: {(c["document_id"], c["chunk_id"]): c["text"] for c in rows}
                    for key, rows in snapshot.items()} for snapshot in (old_sources, new_sources)]
    transitions = Counter()
    fixed_citations = []
    new_citation_failures = []
    changed_answers = []
    for id_, old in by_id_before.items():
        new = by_id_after[id_]
        transitions[f"{old['classification']} -> {new['classification']}"] += 1
        old_check = evidence_check(old, source_maps[0])["status"]
        new_check = evidence_check(new, source_maps[1])["status"]
        if old_check == "CITATION_OR_SOURCE_CHECK_FAILED" and new_check == "SOURCE_QUOTES_EXACT_SEMANTICS_NOT_VERIFIED":
            fixed_citations.append(id_)
        if old_check != "CITATION_OR_SOURCE_CHECK_FAILED" and new_check == "CITATION_OR_SOURCE_CHECK_FAILED":
            new_citation_failures.append(id_)
        if (old.get("formatted") or {}).get("answer") != (new.get("formatted") or {}).get("answer"):
            changed_answers.append(id_)
    summary = {"after_phase": after_phase, "split": split, "selected_rows": expected,
               "actual_live_calls_each_run": expected - (20 if split == "development" else 5),
               "classification_before": dict(Counter(r["classification"] for r in before)),
               "classification_after": dict(Counter(r["classification"] for r in after)),
               "transitions": dict(transitions),
               "mechanical_citation_flags_resolved_ids": fixed_citations,
               "new_mechanical_citation_flags_ids": new_citation_failures,
               "formatted_answer_changed_ids": changed_answers,
               "confirmed_answer_accuracy_before": None,
               "confirmed_answer_accuracy_after": None,
               "limitations": ["Response transitions and exact source quotations do not establish factual accuracy.",
                               "Human review remains required for technical correctness, completeness, and customer action safety."]}
    write_json(folder / f"servy_500_{after_phase}_{split}_comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--after-phase", choices=("retest", "retest2", "final", "holdout-final"), default="retest")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    args = parser.parse_args()
    summary = compare(private_folder(args.data_dir), args.after_phase, args.split)
    print({key: value for key, value in summary.items() if key not in ("formatted_answer_changed_ids",)})


if __name__ == "__main__":
    main()
