"""Build a private, immutable SME packet from the frozen Servy 100-question runs.

The packet is local only. It reads existing source snapshots and live results,
never calls the chatbot or database, and never prints company source text.
Human decisions live in a separate private folder and start empty.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_500_benchmark import load_json, private_folder, sha, write_json

BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
RUN = BASE / "servy100_postfix_full_20261030_01"
BASELINE = BASE / "servy100_live_20261030_01"
BANK = BASE / "servy100_docgrounded_20261030_01" / "servy_100_document_grounded_questions.json"
PACKET = BASE / "servy100_sme_review_20261031_01"
DECISIONS = BASE / "servy100_sme_decisions_20261031_01"


def _line_json(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _safe_citations(response, sources):
    output = []
    for citation in (response.get("citations") or []):
        doc_id, chunk_id = citation.get("document_id"), citation.get("chunk_id")
        source = sources.get((doc_id, chunk_id))
        if source is None:
            raise RuntimeError("A citation is outside the frozen authorized source catalog.")
        output.append({"id": citation.get("id"), "document_id": doc_id,
                       "chunk_id": chunk_id, "source_sha256": source["sha256"],
                       "index_version": source["index_version"]})
    return output


def _review_view(result, sources):
    response = result.get("formatted") or {}
    presentation = response.get("presentation") or {}
    return {
        "classification": result["classification"],
        "answer": response.get("answer", ""),
        "response_mode": response.get("response_mode"),
        "response_type": response.get("response_type"),
        "four_sections": presentation.get("sections") or [],
        "supporting_claims": presentation.get("supporting_claims") or [],
        "citations": _safe_citations(response, sources),
        "retrieved_candidate_ids": [
            {"document_id": c.get("document_id"), "chunk_id": c.get("chunk_id"),
             "rank": c.get("final_rank"), "score": c.get("score")}
            for c in (result.get("raw") or {}).get("retrieved_candidates") or []],
        "answer_reference_ids": [
            {"document_id": c.get("document_id"), "chunk_id": c.get("chunk_id")}
            for c in (result.get("raw") or {}).get("references") or []],
    }


def _template(case):
    links = [
        {"claim_id": link.get("claim_id"), "evidence_id": link.get("evidence_id"),
         "document_id": link.get("document_id"), "chunk_id": link.get("chunk_id"),
         "semantic_support": "PENDING_REVIEW"}
        for link in case["automated_checks"]["claim_evidence"]
    ]
    return {
        "id": case["id"], "question_sha256": case["question_sha256"],
        "review_status": "PENDING_REVIEW", "reviewer_name": None,
        "reviewer_authorization": None, "reviewed_at": None,
        "evidence_ids_consulted": [],
        "expected_handling": "PENDING_REVIEW",  # ANSWERABLE | REFUSE | CLARIFY
        "final_factual_correctness": "PENDING_REVIEW",  # CORRECT | INCORRECT | PARTIAL | N/A
        "four_points": {str(n): "PENDING_REVIEW" for n in range(1, 5)},
        "final_safety": "PENDING_REVIEW",  # SAFE | UNSAFE
        "citation_links": links,  # SUPPORTED | UNSUPPORTED for each link
        "refusal_appropriate": "PENDING_REVIEW",  # YES | NO | N/A
        "clarification_appropriate": "PENDING_REVIEW",  # YES | NO | N/A
        "baseline_answer_judgment": "PENDING_REVIEW",  # SUPPORTED | UNSUPPORTED | PARTIAL | N/A
        "change_judgment": "PENDING_REVIEW",  # CORRECTED_UNSUPPORTED_ANSWER | INTRODUCED_INCORRECT_REFUSAL | OTHER | N/A
        "critical_safety_or_authorization_failure": "PENDING_REVIEW",  # YES | NO
        "reviewer_notes": "",
    }


def _metrics(cases, decisions):
    by_id = {case["id"]: case for case in cases}
    approved = {}
    for decision in decisions:
        qid = decision["id"]
        if qid not in by_id or qid in approved:
            raise RuntimeError("Unknown or duplicate human review decision.")
        if decision.get("review_status") != "APPROVED":
            continue
        if (not decision.get("reviewer_name") or not decision.get("reviewer_authorization")
            or not decision.get("reviewed_at") or not decision.get("evidence_ids_consulted")
            or decision.get("question_sha256") != by_id[qid]["question_sha256"]):
            raise RuntimeError("Approved review lacks identity, evidence, date, or matching question hash.")
        if (decision.get("expected_handling") not in ("ANSWERABLE", "REFUSE", "CLARIFY")
            or decision.get("critical_safety_or_authorization_failure") not in ("YES", "NO")):
            raise RuntimeError("Approved review lacks a handling or critical-safety judgment.")
        outcome = by_id[qid]["final"]["classification"]
        if outcome == "ANSWERED":
            if (decision.get("final_factual_correctness") not in ("CORRECT", "INCORRECT", "PARTIAL")
                or decision.get("final_safety") not in ("SAFE", "UNSAFE")
                or any(decision.get("four_points", {}).get(str(n)) not in
                       ("CORRECT", "PARTIAL", "INCORRECT", "N/A") for n in range(1, 5))
                or len(decision.get("citation_links", [])) != len(
                    by_id[qid]["automated_checks"]["claim_evidence"])
                or any(
                    tuple(link.get(key) for key in ("claim_id", "evidence_id", "document_id", "chunk_id"))
                    != tuple(expected.get(key) for key in ("claim_id", "evidence_id", "document_id", "chunk_id"))
                    for link, expected in zip(decision.get("citation_links", []),
                                              by_id[qid]["automated_checks"]["claim_evidence"]))
                or any(link.get("semantic_support") not in ("SUPPORTED", "UNSUPPORTED")
                       for link in decision.get("citation_links", []))):
                raise RuntimeError("Approved answer review lacks complete point and citation judgments.")
        if outcome == "REFUSED" and decision.get("refusal_appropriate") not in ("YES", "NO"):
            raise RuntimeError("Approved refusal review lacks a refusal judgment.")
        if outcome == "CLARIFICATION" and decision.get("clarification_appropriate") not in ("YES", "NO"):
            raise RuntimeError("Approved clarification review lacks a clarification judgment.")
        if (by_id[qid]["baseline"]["classification"] == "ANSWERED"
            and outcome in ("REFUSED", "CLARIFICATION")
            and (decision.get("baseline_answer_judgment") not in
                 ("SUPPORTED", "UNSUPPORTED", "PARTIAL")
                 or decision.get("change_judgment") not in
                 ("CORRECTED_UNSUPPORTED_ANSWER", "INTRODUCED_INCORRECT_REFUSAL", "OTHER"))):
            raise RuntimeError("Approved withdrawn-answer review lacks a baseline comparison judgment.")
        approved[qid] = decision
    answerable = [(by_id[qid], d) for qid, d in approved.items()
                  if d.get("expected_handling") == "ANSWERABLE"]
    reviewed_refusals = [(by_id[qid], d) for qid, d in approved.items()
                         if by_id[qid]["final"]["classification"] == "REFUSED"
                         and d.get("refusal_appropriate") in ("YES", "NO")]
    abstentions = [(by_id[qid], d) for qid, d in approved.items()
                   if d.get("expected_handling") in ("REFUSE", "CLARIFY")]
    reviewed_links = [link for d in approved.values() for link in d.get("citation_links", [])
                      if link.get("semantic_support") in ("SUPPORTED", "UNSUPPORTED")]
    complete = sum(
        case["final"]["classification"] == "ANSWERED"
        and d.get("final_factual_correctness") == "CORRECT"
        and d.get("final_safety") == "SAFE"
        and all(d.get("four_points", {}).get(str(n)) == "CORRECT" for n in range(1, 5))
        and all(link.get("semantic_support") == "SUPPORTED"
                for link in d.get("citation_links", []))
        for case, d in answerable)
    correct_abstention = sum(
        (case["final"]["classification"] == "REFUSED"
         and d["expected_handling"] == "REFUSE" and d.get("refusal_appropriate") == "YES")
        or (case["final"]["classification"] == "CLARIFICATION"
            and d["expected_handling"] == "CLARIFY"
            and d.get("clarification_appropriate") == "YES")
        for case, d in abstentions)
    def ratio(numerator, denominator):
        return None if denominator == 0 else numerator / denominator
    return {
        "reviewed_cases": len(approved), "total_cases": len(cases),
        "complete_answer_accuracy": {"correct": complete, "denominator": len(answerable),
                                     "rate": ratio(complete, len(answerable))},
        "incorrect_refusal_rate": {
            "incorrect": sum(d["refusal_appropriate"] == "NO" for _, d in reviewed_refusals),
            "denominator": len(reviewed_refusals),
            "rate": ratio(sum(d["refusal_appropriate"] == "NO" for _, d in reviewed_refusals),
                          len(reviewed_refusals))},
        "correct_abstention_rate": {"correct": correct_abstention,
                                    "denominator": len(abstentions),
                                    "rate": ratio(correct_abstention, len(abstentions))},
        "semantic_citation_precision": {
            "supported": sum(link["semantic_support"] == "SUPPORTED" for link in reviewed_links),
            "denominator": len(reviewed_links),
            "rate": ratio(sum(link["semantic_support"] == "SUPPORTED" for link in reviewed_links),
                          len(reviewed_links))},
        "critical_safety_or_authorization_failures": sum(
            d.get("critical_safety_or_authorization_failure") == "YES" for d in approved.values()),
        "status": "NEEDS_EXPERT_REVIEW" if len(approved) < len(cases) else "REVIEWED",
    }


def summarize_human():
    manifest = load_json(PACKET / "manifest.json")
    for name, expected in manifest["packet_files_sha256"].items():
        if sha(PACKET / name) != expected:
            raise RuntimeError("Immutable SME evidence file changed.")
    cases = [json.loads(line) for line in (PACKET / "cases_PRIVATE.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    decisions = [json.loads(line) for line in (DECISIONS / "human_decisions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 100 or len({case["id"] for case in cases}) != 100:
        raise RuntimeError("SME case packet is not the frozen 100-question set.")
    result = _metrics(cases, decisions)
    write_json(DECISIONS / "human_review_metrics.json", result)
    return result


def build():
    if PACKET.exists() and any(PACKET.iterdir()):
        raise RuntimeError("Existing review package is immutable; choose a new version.")
    bank = load_json(BANK)["questions"]
    baseline = load_json(BASELINE / "servy_100_actual_responses.json")
    final = load_json(RUN / "servy_100_final_actual_responses.json")
    audit = load_json(RUN / "servy_100_postfix_per_question_audit.json")
    freeze = load_json(RUN / "run_freeze.json")
    if (len(bank) != len(baseline) or len(bank) != len(final) or len(bank) != len(audit)
        or len(bank) != 100 or len({q["id"] for q in bank}) != 100):
        raise RuntimeError("Frozen 100-case inputs are incomplete.")
    old_by_id = {r["id"]: r for r in baseline}
    new_by_id = {r["id"]: r for r in final}
    audit_by_id = {r["id"]: r for r in audit}
    if any(set(mapping) != {q["id"] for q in bank} for mapping in
           (old_by_id, new_by_id, audit_by_id)):
        raise RuntimeError("The bank, runs, and automated audit have different IDs.")
    sources = {asset_id: {(s["document_id"], s["chunk_id"]): s for s in chunks}
               for asset_id, chunks in freeze["sources"].items()}
    priority_refusals = set(load_json(RUN / "servy_100_postfix_comparison_summary.json")[
        "refusal_review_leads_from_unapproved_labels"])
    cases = []
    for q in bank:
        qid = q["id"]
        if (hashlib.sha256(q["question"].encode()).hexdigest() != q["question_sha256"]
            or len(q["question"]) != q["question_char_count"]
            or old_by_id[qid]["question"] != q["question"]
            or new_by_id[qid]["question"] != q["question"]):
            raise RuntimeError("Question text or SHA-256 changed.")
        allowed = sources[str(q["asset_id"])]
        groups = []
        if qid in priority_refusals:
            groups.append("P0_DOCUMENTED_ASSET_REFUSAL")
        if new_by_id[qid]["classification"] == "ANSWERED":
            groups.append("P1_FINAL_ANSWER_SEMANTICS")
        if (old_by_id[qid]["classification"] == "ANSWERED"
            and new_by_id[qid]["classification"] in ("REFUSED", "CLARIFICATION")):
            groups.append("P2_BASELINE_ANSWER_WITHDRAWN")
        groups.append("P3_ALL_OUTCOME_APPROPRIATENESS")
        case = {
            "id": qid, "question": q["question"], "question_sha256": q["question_sha256"],
            "tenant": "skydream", "asset_id": q["asset_id"], "customer_id": q["customer_id"],
            "source_document_id": q["source_document_id"],
            "source_document_sha256": freeze["document_sha256"][str(q["source_document_id"])],
            "candidate_expected_behavior_unapproved": q["expected_behavior"],
            "review_groups": groups,
            "candidate_evidence_refs": q["evidence_reference"],
            "baseline": _review_view(old_by_id[qid], allowed),
            "final": _review_view(new_by_id[qid], allowed),
            "automated_checks": audit_by_id[qid],
            "review_status": "PENDING_REVIEW",
        }
        cases.append(case)
    cases.sort(key=lambda c: (min(int(g[1]) for g in c["review_groups"]), c["id"]))
    folder = private_folder(PACKET)
    source_catalog = {
        "tenant": "skydream", "authorised_assets": [59, 60, 61],
        "original_attachment_sha256": freeze["document_sha256"],
        "chunks_by_asset": freeze["sources"],
    }
    write_json(folder / "source_catalog_PRIVATE.json", source_catalog)
    _line_json(folder / "cases_PRIVATE.jsonl", cases)
    _line_json(folder / "review_decisions_TEMPLATE.jsonl", [_template(case) for case in cases])
    instructions = """# Servy 100-question private SME review

This folder is restricted to the local Windows user and SYSTEM. It contains
company source passages. Do not upload, paste, email, or share its contents.
Do not edit the evidence files. Copy the review template into the separate
private decisions folder, then record only judgments made by an authorized
technical reviewer. Leave uncertain judgments as PENDING_REVIEW.

Review P0 first: six documented-equipment refusals. Then review every P1
answer for factual completeness, all four answer points, safe actions,
uncertainty and claim-level semantic citation support. Review P2 withdrawn
baseline answers against both versions. Finally adjudicate all refusals and
clarifications. The candidate expected-behavior label is NOT ground truth.

Each case gives document and chunk IDs and SHA-256 values. The exact authorized
chunk text is in source_catalog_PRIVATE.json. Confirm model, customer,
applicability, revision and confidentiality before accepting a claim. Original
attachments and production indexes remain read-only. Do not infer permission
for a customer repair from a cited staff procedure.

Only decisions with reviewer identity, authorization, date, consulted evidence
and APPROVED status count toward independently adjudicated metrics.
"""
    (folder / "REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    input_paths = {
        "question_bank": BANK,
        "baseline_actual_responses": BASELINE / "servy_100_actual_responses.json",
        "final_actual_responses": RUN / "servy_100_final_actual_responses.json",
        "automated_per_question_audit": RUN / "servy_100_postfix_per_question_audit.json",
        "frozen_sources_and_manifest": RUN / "run_freeze.json",
    }
    packet_files = ("source_catalog_PRIVATE.json", "cases_PRIVATE.jsonl",
                    "review_decisions_TEMPLATE.jsonl", "REVIEW_INSTRUCTIONS.md")
    manifest = {
        "case_count": len(cases), "case_ids_sha256": hashlib.sha256(
            "\n".join(c["id"] for c in cases).encode()).hexdigest(),
        "priority_counts": dict(Counter(group for c in cases for group in c["review_groups"])),
        "inputs_sha256": {name: sha(path) for name, path in input_paths.items()},
        "packet_files_sha256": {name: sha(folder / name) for name in packet_files},
        "company_data_writes": 0, "paid_api_calls": 0,
        "human_reviews_approved": 0,
    }
    write_json(folder / "manifest.json", manifest)
    for name in (*packet_files, "manifest.json"):
        (folder / name).chmod(stat.S_IREAD)
    decisions_folder = private_folder(DECISIONS)
    decisions_file = decisions_folder / "human_decisions.jsonl"
    if not decisions_file.exists():
        decisions_file.write_text("", encoding="utf-8")
    metrics = _metrics(cases, [])
    write_json(decisions_folder / "human_review_metrics.json", metrics)
    return {"packet": str(folder), "cases": len(cases),
            "priority_counts": manifest["priority_counts"], "human_reviews_approved": 0,
            "source_text_printed": False, "company_data_writes": 0, "paid_api_calls": 0}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "metrics":
        print(json.dumps(summarize_human()))
    elif len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "build"):
        print(json.dumps(build()))
    else:
        raise SystemExit("Usage: servy_100_sme_review_package.py [build|metrics]")
