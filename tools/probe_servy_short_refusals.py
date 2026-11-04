"""Private numeric diagnosis of three source-anchored short-question refusals."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
from tools.servy_500_benchmark import activate_index_copy, read_only_database


BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
BANK = BASE / "servy_short_questions_20261104_01" / "short_questions_300_candidate.jsonl"
OUT = BASE / "servy_short_answer_candidates_20261104_01"


def main():
    corrected = sys.argv[1:] == ["--corrected"]
    if sys.argv[1:] and not corrected:
        raise RuntimeError("Unknown probe option")
    os.environ["SERVY_LLM_PROVIDER"] = "extractive"
    os.environ["SERVY_ALLOW_REMOTE_LLM"] = "0"
    configure_process()
    verify_readonly_identity()
    from django.conf import settings
    if settings.SERVY_LLM_PROVIDER != "extractive" or settings.SERVY_ALLOW_REMOTE_LLM:
        raise RuntimeError("Offline extractive provider required")
    from core.services.rag import ask_rag
    from core.services.customer_answer import build_customer_answer
    from core.services.llm import check_evidence_sufficiency
    from tools.servy_real_demo import _asset, DEMO_USER_ID
    bank = (BANK.with_name("short_questions_300_v2_candidate.jsonl")
            if corrected else BANK)
    rows = [json.loads(line) for line in bank.read_text(encoding="utf-8").splitlines()]
    samples = [next(r for r in rows if r["asset_id"] == asset_id
                    and r["question_kind"] == "source") for asset_id in (59, 60, 61)]
    activate_index_copy(OUT)
    report = []
    with read_only_database():
        for row in samples:
            tenant, customer, asset = _asset(row["asset_id"])
            rag = ask_rag(tenant=tenant, customer=customer, asset=asset,
                          question=row["question"], staff_mode=False,
                          include_confidential=False, service_call=None, top_k=5)
            interaction = SimpleNamespace(id=None, created_by_id=DEMO_USER_ID,
                                          tenant=tenant, customer=customer, asset=asset,
                                          question=row["question"])
            contract = build_customer_answer(interaction, None, rag.get("references", []))
            retrieved = rag.get("retrieved_candidates") or []
            refs = rag.get("references") or []
            evidence = check_evidence_sufficiency(row["question"], asset, rag.get("retrieved") or [])
            target = (row["source_document_id"], row["source_chunk_id"])
            report.append({"id": row["id"], "asset_id": row["asset_id"],
                           "target_in_retrieval_top5": target in [
                               (r.get("document_id"), r.get("chunk_id")) for r in retrieved[:5]],
                           "retrieved_count": len(retrieved), "supported_reference_count": len(refs),
                           "rag_engine": rag.get("engine"), "evidence_reason": evidence.reason,
                           "customer_response_type": contract.get("response_type"),
                           "customer_citation_count": len(contract.get("citations") or [])})
    target = OUT / ("refusal_probe_corrected.json" if corrected else "refusal_probe.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
