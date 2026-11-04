"""Offline, private before/after audit of the frozen Servy 100-question run.

This checks identity, source scope, exact quotations, and response structure.
It deliberately does not convert exact quotes or candidate labels into expert truth.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_500_benchmark import load_json, write_json


BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
OLD = BASE / "servy100_live_20261030_01"
NEW = BASE / "servy100_postfix_full_20261030_01"


def audit():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    import django
    django.setup()
    from core.services.customer_answer import _question_guard
    bank = load_json(BASE / "servy100_docgrounded_20261030_01" /
                     "servy_100_document_grounded_questions.json")["questions"]
    baseline = load_json(OLD / "servy_100_actual_responses.json")
    current = load_json(NEW / "servy_100_final_actual_responses.json")
    frozen = load_json(NEW / "run_freeze.json")
    if (len(bank) != len(baseline) or len(bank) != len(current) or len(bank) != 100
        or len({r["id"] for r in bank}) != 100):
        raise RuntimeError("Expected exactly 100 unique frozen questions and two complete runs.")
    before = {r["id"]: r for r in baseline}
    after = {r["id"]: r for r in current}
    if set(before) != set(after) or set(after) != {r["id"] for r in bank}:
        raise RuntimeError("The runs do not cover the same 100 question IDs.")
    sources = {asset_id: {(s["document_id"], s["chunk_id"]): s["text"] for s in chunks}
               for asset_id, chunks in frozen["sources"].items()}
    rows = []
    transitions = Counter()
    for question in bank:
        qid = question["id"]
        old, new = before[qid], after[qid]
        digest = hashlib.sha256(question["question"].encode()).hexdigest()
        if (digest != question["question_sha256"]
            or len(question["question"]) != question["question_char_count"]
            or old["question"] != question["question"]
            or new["question"] != question["question"]
            or old["question_sha256"] != digest or new["question_sha256"] != digest):
            raise RuntimeError(f"Question text or hash changed for {qid}.")
        if (old["asset_id"] != new["asset_id"] or old["customer_id"] != new["customer_id"]
            or new["asset_id"] != question["asset_id"] or new["customer_id"] != question["customer_id"]):
            raise RuntimeError(f"Customer or asset scope changed for {qid}.")
        response = new.get("formatted") or {}
        presentation = response.get("presentation") or {}
        sections = presentation.get("sections") or []
        citations = response.get("citations") or []
        cited = {c.get("id"): c for c in citations}
        allowed = sources[str(question["asset_id"])]
        refs = (new.get("raw") or {}).get("references") or []
        candidates = (new.get("raw") or {}).get("retrieved_candidates") or []
        source_ids = {(r["document_id"], r["chunk_id"]) for r in question["evidence_reference"]}
        candidate_ids = {(r.get("document_id"), r.get("chunk_id")) for r in candidates}
        ref_ids = {(r.get("document_id"), r.get("chunk_id")) for r in refs}
        claims = presentation.get("supporting_claims") or []
        if not claims:
            claims = [c for section in sections for c in section.get("claims", [])
                      if c.get("kind") not in ("uncertainty", "next_step")]
        quote_exact = bool(claims) and all(c.get("source_text") and any(
            c["source_text"] in allowed.get((cited[e].get("document_id"),
                                               cited[e].get("chunk_id")), "")
            for e in c.get("evidence_ids", []) if e in cited) for c in claims)
        citations_in_scope = all((c.get("document_id"), c.get("chunk_id")) in allowed
                                 for c in citations)
        refs_in_scope = all((r.get("document_id"), r.get("chunk_id")) in allowed for r in refs)
        sections_present = [bool(next((s for s in sections if s.get("number") == n), None))
                            for n in range(1, 5)]
        sections_nonempty = [bool(next((s.get("claims") for s in sections
                                        if s.get("number") == n), [])) for n in range(1, 5)]
        claim_evidence = []
        for claim in claims:
            for evidence_id in claim.get("evidence_ids", []):
                citation = cited.get(evidence_id)
                if not citation:
                    claim_evidence.append({"claim_id": claim.get("id"), "evidence_id": evidence_id,
                                           "status": "MISSING_CITATION"})
                    continue
                key = (citation.get("document_id"), citation.get("chunk_id"))
                source_text = allowed.get(key, "")
                quoted = claim.get("source_text") or ""
                claim_evidence.append({
                    "claim_id": claim.get("id"), "claim_kind": claim.get("kind"),
                    "evidence_id": evidence_id, "document_id": key[0], "chunk_id": key[1],
                    "source_sha256": hashlib.sha256(source_text.encode()).hexdigest() if source_text else None,
                    "quote_sha256": hashlib.sha256(quoted.encode()).hexdigest() if quoted else None,
                    "exact_quote_in_authorized_source": bool(source_text and quoted and quoted in source_text),
                    "semantic_support": "NEEDS_EXPERT_REVIEW",
                })
        changed = old["classification"] != new["classification"]
        transitions[f'{old["classification"]}->{new["classification"]}'] += 1
        refusal_stage = None
        if new["classification"] == "REFUSED":
            guard = _question_guard(question["question"])
            if (new.get("raw") or {}).get("engine") == "pre_rag_customer_guard":
                refusal_stage = "PRE_RAG_HAZARD_GUARD"
            elif guard:
                refusal_stage = "QUESTION_GUARD_" + guard.upper()
            elif not refs:
                refusal_stage = "RAW_ANSWER_HAS_NO_REFERENCES"
            else:
                refusal_stage = "ANSWER_CONSTRUCTION_OR_EVIDENCE_FILTER"
        rows.append({
            "id": qid, "asset_id": question["asset_id"], "customer_id": question["customer_id"],
            "question_sha256": digest, "baseline": old["classification"],
            "postfix": new["classification"], "outcome_changed": changed,
            "final_source_run": new["final_source_run"],
            "candidate_expected_behavior_unapproved": question["expected_behavior"],
            "candidate_evidence_in_retrieval_pool": bool(source_ids & candidate_ids),
            "candidate_evidence_in_answer_refs": bool(source_ids & ref_ids),
            "refusal_stage": refusal_stage,
            "raw_refs_in_authorized_scope": refs_in_scope,
            "citations_in_authorized_scope": citations_in_scope,
            "technical_claim_count": len(claims), "technical_claim_quotes_exact": quote_exact,
            "claim_evidence": claim_evidence,
            "citation_count": len(citations),
            "four_points_present": sections_present,
            "four_points_nonempty": sections_nonempty,
            "point_1_has_uncertainty_placeholder": any(
                c.get("kind") == "uncertainty" for s in sections if s.get("number") == 1
                for c in s.get("claims", [])),
            "point_2_has_customer_observation": any(
                c.get("kind") == "observation" for s in sections if s.get("number") == 2
                for c in s.get("claims", [])),
            "customer_authorized_action_count": len(presentation.get("customer_authorized_actions") or []),
            "answer_completeness": "PENDING_TECHNICAL_REVIEW",
            "answer_safety": "PENDING_TECHNICAL_REVIEW",
            "uncertainty_quality": "PENDING_TECHNICAL_REVIEW",
            "citation_semantics": "PENDING_TECHNICAL_REVIEW",
            "refusal_correctness": "PENDING_TECHNICAL_REVIEW",
        })
    counts = Counter(r["postfix"] for r in rows)
    answered = [r for r in rows if r["postfix"] == "ANSWERED"]
    changed = [r for r in rows if r["outcome_changed"]]
    refused = [r for r in rows if r["postfix"] == "REFUSED"]
    review_leads = [r["id"] for r in refused
                    if r["candidate_expected_behavior_unapproved"] == "ANSWER_WITH_CITATION"]
    summary = {
        "questions": len(rows), "baseline_counts": dict(Counter(r["baseline"] for r in rows)),
        "postfix_counts": dict(counts), "outcome_transitions": dict(transitions),
        "changed_outcomes": len(changed),
        "refusal_stage_counts": dict(Counter(r["refusal_stage"] for r in refused)),
        "full_live_results_reused": sum(r["final_source_run"] == "full_live_rerun_formatter_unchanged"
                                        for r in rows),
        "focused_live_replacements": [r["id"] for r in rows
                                    if r["final_source_run"] == "focused_live_retest"],
        "answered_with_exact_authorized_quotes": sum(r["technical_claim_quotes_exact"]
                                                    and r["citations_in_authorized_scope"] for r in answered),
        "answered_with_four_nonempty_sections": sum(all(r["four_points_nonempty"]) for r in answered),
        "answered_with_point_1_uncertainty_placeholder": sum(
            r["point_1_has_uncertainty_placeholder"] for r in answered),
        "answered_without_point_2_observation": sum(
            not r["point_2_has_customer_observation"] for r in answered),
        "claim_evidence_links": sum(len(r["claim_evidence"]) for r in rows),
        "claim_evidence_links_with_exact_quotes": sum(
            e.get("exact_quote_in_authorized_source", False)
            for r in rows for e in r["claim_evidence"]),
        "newly_answered_ids": [r["id"] for r in changed if r["postfix"] == "ANSWERED"],
        "refusal_review_leads_from_unapproved_labels": review_leads,
        "independently_verified_complete_answer_accuracy": None,
        "confirmed_correct_refusals": None, "confirmed_incorrect_refusals": None,
        "confirmed_unsupported_claims": None, "semantic_citation_precision": None,
        "semantic_review_status": "PENDING_TECHNICAL_REVIEW",
        "paid_api_cost_inr": 0,
    }
    write_json(NEW / "servy_100_postfix_per_question_audit.json", rows)
    write_json(NEW / "servy_100_postfix_comparison_summary.json", summary)
    lines = ["# Servy 100-question post-fix audit", "",
             "Exact full-text question hashes, tenant/customer/asset bindings and source scope were checked.",
             "Verbatim source quotation and four-section presence are mechanical checks, not accuracy proof.",
             "", "## Outcome counts", "",
             f"Baseline: {summary['baseline_counts']}", f"Post-fix: {summary['postfix_counts']}",
             f"Changed outcomes: {len(changed)}",
             f"Live provenance: {summary['full_live_results_reused']} unchanged full-run results plus "
             f"{len(summary['focused_live_replacements'])} focused live replacements.",
             "", "## Evidence checks and refusal triage", "",
             f"Exact authorized claim-to-source links: {summary['claim_evidence_links_with_exact_quotes']}/"
             f"{summary['claim_evidence_links']}.",
             f"Answered cases with four nonempty sections: {summary['answered_with_four_nonempty_sections']}/"
             f"{len(answered)}. Section presence does not prove complete or safe content.",
             f"Refusal stages: {summary['refusal_stage_counts']}",
             f"Unapproved-label answer leads requiring review: {', '.join(review_leads)}.",
             "", "## Question-by-question comparison", "",
             "| ID | Before | After | Changed | Exact quote | Four nonempty points |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['baseline']} | {row['postfix']} | "
                     f"{row['outcome_changed']} | {row['technical_claim_quotes_exact']} | "
                     f"{all(row['four_points_nonempty'])} |")
    lines += ["", "## Accuracy limits", "",
              "Complete-answer accuracy, refusal correctness, unsupported claims and semantic citation precision remain PENDING_TECHNICAL_REVIEW.",
              "Candidate expected behavior is not approved ground truth. No 98% accuracy claim is made.", ""]
    (NEW / "servy_100_postfix_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = audit()
    print(json.dumps({key: summary[key] for key in (
        "questions", "baseline_counts", "postfix_counts", "changed_outcomes",
        "answered_with_exact_authorized_quotes", "answered_with_four_nonempty_sections",
        "newly_answered_ids", "refusal_review_leads_from_unapproved_labels",
        "independently_verified_complete_answer_accuracy")}))
