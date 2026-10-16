"""Prepare private source passages and safe clarification requests for captured complaints."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.ml.op23_source_admission import canonical_bytes, sha256

private = (ROOT / "private_training_data").resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--admission", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--suggestions", type=Path)
args = parser.parse_args()
for path in (args.capture, args.admission, args.output, args.suggestions):
    if path and not path.resolve().is_relative_to(private):
        raise ValueError("Reviewer files must remain private")
capture = args.capture
admission = json.loads(args.admission.read_text(encoding="utf-8"))
audit = json.loads((private / "op22_provenance_audit.json").read_text(encoding="utf-8"))
complaints = json.loads((capture / "complaints.json").read_text(encoding="utf-8"))
assets = {a["id"]: a for a in json.loads((capture / "assets.json").read_text(encoding="utf-8"))}
site_assets = json.loads((capture / "site_assets.json").read_text(encoding="utf-8"))
local_links = {a["external_asset_id"]: a for a in audit["asset_links"] if a["code_matches"] and a["model_matches"]}
source_links = {s["external_asset_id"]: [] for s in admission["sources"] if s.get("external_asset_id")}
for source in admission["sources"]:
    if source.get("external_asset_id"):
        source_links[source["external_asset_id"]].append(source)
suggestions = {r["source_complaint_id"]: r for r in
               (json.loads(line) for line in args.suggestions.read_text(encoding="utf-8").splitlines() if line.strip())} if args.suggestions else {}
package = []
for complaint in complaints:
    asset_id = complaint.get("asset")
    asset = assets.get(asset_id)
    if asset_id is not None and asset is None:
        raise ValueError("Complaint asset lacks authenticated catalogue mapping")
    source_list = source_links.get(asset_id, []) if asset else []
    candidate = suggestions.get(complaint["id"], {})
    row = {
        "case_id": f"OP23-COMPLAINT-{complaint['id']}",
        "dataset_role": "HISTORICAL_DIAGNOSTIC", "review_status": "PENDING_REVIEW",
        "reviewer_identity": None, "reviewed_at": None, "approval": None,
        "source_organisation": "273", "tenant_id": 9,
        "query_provenance": {"kind": "starlly_complaint", "complaint_id": complaint["id"],
                             "complaint_sha256": sha256(canonical_bytes(complaint))},
        "original_complaint": complaint,
        "question_text": complaint.get("description") or "",
        "external_asset_id": asset_id,
        "local_asset_id": local_links[asset_id]["local_asset_id"] if asset_id in local_links else None,
        "asset_catalogue": {k: v for k, v in asset.items() if k != "attachments"} if asset else None,
        "matching_site_asset_records": [s for s in site_assets if s.get("asset") == asset_id] if asset else [],
        "source_manuals": [{
            "source_id": s["source_id"], "external_attachment_id": s["external_attachment_id"],
            "source_sha256": s["raw_sha256"], "source_version": s["source_version"],
            "manual_name": s["attachment_name"], "independence_status": s["independence_status"],
            "passages": s["passages"],
        } for s in source_list],
        "machine_suggested_chunk_ids": candidate.get("candidate_passage_chunk_ids", []),
        "machine_suggested_passages": candidate.get("candidate_evidence_passages", []),
        "approved_positive_passage_ids": [], "approved_alternatives": [],
        "answerability": "PENDING_REVIEW" if asset else "NEEDS_ASSET_CLARIFICATION",
        "review_prompt": ("Identify the noise-specific answer-bearing passage, any equivalent passages, and whether the manuals actually support a safe technical answer."
                          if asset else "Ask for an authoritative asset code, serial number, or site-asset record; do not infer equipment from the complaint wording."),
        "clarification_request": None if asset else "Please provide the affected asset code, serial number, or installed site-asset ID.",
    }
    row["review_case_template"] = {
        "query_id": row["case_id"], "question_text": row["question_text"],
        "dataset_role": row["dataset_role"], "review_status": "PENDING_REVIEW",
        "reviewer_identity": None, "reviewed_at": None, "approval": None,
        "source_organisation": admission["source_organisation"], "tenant_id": admission["tenant_id"],
        "query_provenance": row["query_provenance"], "external_asset_id": asset_id,
        "evaluation_task": "TECHNICAL_ANSWERABILITY" if asset else "CLARIFICATION",
        "answerability": "PENDING_REVIEW" if asset else "NEEDS_ASSET_CLARIFICATION",
        "source_id": None, "source_sha256": None, "source_version": None,
        "source_attachment_id": None, "valid_positive_passage_ids": [],
        "valid_alternative_passage_ids": [], "reviewed_negative_passage_ids": [],
        "searched_source_universe": {
            "source_ids": [s["source_id"] for s in source_list],
            "indexed_source_ids": [], "search_complete": False,
        },
    }
    package.append(row)
target = args.output
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as output:
    for row in package:
        output.write(json.dumps(row) + "\n")
print(json.dumps({"private_package": str(target), "cases": len(package),
                  "asset_resolved": sum(r["external_asset_id"] is not None for r in package),
                  "clarification_required": sum(r["external_asset_id"] is None for r in package),
                  "approved_cases": 0}, indent=2))
