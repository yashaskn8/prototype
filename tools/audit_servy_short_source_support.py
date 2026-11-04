"""Check source-anchored refusal overlap locally without emitting source text."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
from tools.servy_500_benchmark import read_only_database

BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
BANK = BASE / "servy_short_questions_20261104_01" / "short_questions_300_candidate.jsonl"
OUT = BASE / "servy_short_answer_candidates_20261104_01"


def main():
    os.environ["SERVY_LLM_PROVIDER"] = "extractive"
    os.environ["SERVY_ALLOW_REMOTE_LLM"] = "0"
    configure_process()
    verify_readonly_identity()
    from core.services.customer_answer import source_chunk
    from core.services.identifiers import extract_error_codes
    from core.services.llm import (
        _grounding_terms, _STOPWORDS, evaluate_chunk_claim_support,
        extract_technical_numbers,
    )
    from core.services.retriever import normalize_domain_terms
    from tools.servy_real_demo import _asset
    rows = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()
            if '"question_kind": "source"' in line]
    if len(rows) != 120:
        raise RuntimeError("Expected 120 source-anchored rows")
    outcomes = Counter()
    without_document_sequence = Counter()
    verified = 0
    with read_only_database():
        for row in rows:
            tenant, customer, asset = _asset(row["asset_id"])
            chunk = source_chunk(tenant=tenant, customer=customer, asset=asset,
                                 document_id=row["source_document_id"],
                                 chunk_id=row["source_chunk_id"])
            if hashlib.sha256(chunk.text.encode()).hexdigest() != row["source_chunk_sha256"]:
                raise RuntimeError("Source passage changed since question generation")
            verified += 1
            q = row["question"]
            terms = _grounding_terms(q, _STOPWORDS, normalize=normalize_domain_terms)
            candidate = SimpleNamespace(title=chunk.document.title, heading=chunk.heading,
                                        text=chunk.text)
            supported, reason = evaluate_chunk_claim_support(
                candidate, terms, extract_error_codes(q), extract_technical_numbers(q),
                q_text=q)
            outcomes[(supported, reason.split(":", 1)[0])] += 1
            document_part = chunk.document.title.split(" - ", 1)[-1][:38]
            clean_part = re.sub(r"^\d{1,3}\s+", "", document_part)
            clean_q = q.replace(document_part, clean_part, 1)
            clean_terms = _grounding_terms(clean_q, _STOPWORDS,
                                           normalize=normalize_domain_terms)
            clean_supported, clean_reason = evaluate_chunk_claim_support(
                candidate, clean_terms, extract_error_codes(clean_q),
                extract_technical_numbers(clean_q), q_text=clean_q)
            without_document_sequence[(clean_supported, clean_reason.split(":", 1)[0])] += 1
    result = {"source_anchors_verified": verified,
              "anchor_support_check": {f"{supported}:{reason}": count
                                       for (supported, reason), count in sorted(outcomes.items())},
              "without_document_sequence_check": {
                  f"{supported}:{reason}": count for (supported, reason), count
                  in sorted(without_document_sequence.items())},
              "semantic_answerability": "NEEDS_EXPERT_REVIEW"}
    (OUT / "source_anchor_support_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
