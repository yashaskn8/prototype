"""Read-only local replay of saved RAG outputs through the customer formatter.

This measures formatter-only changes; it is not a new end-to-end RAG result.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
from tools.servy_500_benchmark import (classify, load_json, operational_counts,
                                       private_folder, read_only_database, serial, sha,
                                       write_json)


def measurement_claims(contract):
    return [claim.get("text", "") for section in (contract.get("presentation") or {}).get("sections", [])
            for claim in section.get("claims", []) if claim.get("kind") == "observation"
            and re.search(r"\b(?:operating current|current draw|static pressure|line voltage|supply voltage)\b",
                          claim.get("text", ""), re.I)]


def replay(folder):
    from core.models import Asset, CustomerInteraction
    from core.services.customer_diagnostic import (build_structured_diagnostic_contract,
                                                   check_for_safety_hazard, check_for_vague_complaint)
    rows = load_json(folder / "servy_500_retest2_actual_test_results.json")
    if len(rows) != 400:
        raise RuntimeError("The second development retest must complete before formatter replay.")
    output = []
    before = operational_counts()
    assets = {asset.id: asset for asset in Asset.objects.select_related("tenant", "customer", "site", "product")
              .filter(id__in={row["asset_id"] for row in rows})}
    for row in rows:
        if row["status"] != "COMPLETED":
            output.append({"id": row["id"], "status": row["status"], "classification": row["classification"]})
            continue
        asset = assets[row["asset_id"]]
        raw = row.get("raw") or {}
        interaction = CustomerInteraction(tenant=asset.tenant, customer=asset.customer,
                                          asset=asset, site=asset.site, question=row["question"],
                                          answer=raw.get("answer", ""), source_refs=raw.get("references", []))
        hazard, warning = check_for_safety_hazard(row["question"])
        vague, clarification, options = check_for_vague_complaint(row["question"], asset)
        contract = serial(build_structured_diagnostic_contract(
            interaction=interaction, session=None, raw_answer=raw.get("answer", ""),
            retrieved_references=raw.get("references", []),
            clarification_question=clarification if vague and not hazard else None,
            clarification_options=options if vague and not hazard else None,
            is_hazard=hazard, hazard_warning=warning))
        classification, reason = classify({"status": "COMPLETED", "formatted": contract})
        output.append({"id": row["id"], "status": "FORMATTER_REPLAYED", "asset_id": asset.id,
                       "classification": classification, "classification_reason": reason,
                       "formatted": contract, "raw_reference_count": len(raw.get("references") or [])})
        if operational_counts() != before:
            raise RuntimeError("Operational records changed during formatter replay.")
    summary = {"formatting_only": True, "live_rag_calls": 0, "selected": len(rows),
               "before_classifications": dict(Counter(r["classification"] for r in rows)),
               "after_classifications": dict(Counter(r["classification"] for r in output)),
               "measurement_observation_claims_before": sum(len(measurement_claims(r.get("formatted") or {})) for r in rows),
               "measurement_observation_claims_after": sum(len(measurement_claims(r.get("formatted") or {})) for r in output),
               "formatter_code_sha256": sha(ROOT / "core/services/customer_answer.py"),
               "no_database_writes": True,
               "technical_correctness": "NOT_VERIFIED"}
    write_json(folder / "servy_500_formatter_safety_replay.json", output)
    write_json(folder / "servy_500_formatter_safety_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = private_folder(args.data_dir)
    import django
    django.setup()
    with read_only_database():
        summary = replay(folder)
    print(summary)


if __name__ == "__main__":
    main()
