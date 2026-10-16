"""Read-only proposals for idempotent MySQL provenance mappings."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

from core.ml.op23_provenance_reconcile import reconcile_document
from core.ml.op23_source_admission import admit_capture, canonical_bytes, sha256
from core.models import Asset, KnowledgeDocument, SourceProvenanceRecord

parser = argparse.ArgumentParser()
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--admission", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
private = (ROOT / "private_training_data").resolve()
for path in (args.capture, args.admission, args.output):
    if not path.resolve().is_relative_to(private):
        raise ValueError("Provenance audit paths must remain private")
baseline = json.loads((private / "op23_review" / "sealed_prior_sources_v2.json").read_text(encoding="utf-8"))
admission = json.loads(args.admission.read_text(encoding="utf-8"))
if canonical_bytes(admission) != canonical_bytes(admit_capture(args.capture, baseline, expected_org="273", tenant_id=9)):
    raise ValueError("Admission manifest differs from authenticated source bytes")
attachments = json.loads((args.capture / "attachments.json").read_text(encoding="utf-8"))
attachment_files = {a["external_attachment_id"]: a for a in attachments}
proposals = []
for source in admission["sources"]:
    if not source.get("source_id"):
        continue
    matches = list(KnowledgeDocument.objects.filter(tenant_id=9, checksum_sha256=source["raw_sha256"])
                   .select_related("asset").prefetch_related("chunks"))
    if not matches:
        proposals.append({"source_id": source["source_id"], "status": "NO_LOCAL_DOCUMENT",
                          "external_attachment_id": source["external_attachment_id"]})
        continue
    for document in matches:
        asset = document.asset
        if asset is None:
            proposals.append({"source_id": source["source_id"], "document_id": document.id,
                              "status": "CONFLICT", "reasons": ["document_has_no_asset"]})
            continue
        record = attachment_files[source["external_attachment_id"]]
        text = (args.capture / record["text_file"]).read_text(encoding="utf-8")
        existing = SourceProvenanceRecord.objects.filter(
            tenant_id=9, source_system="starlly", source_org="273",
            external_object_type="document", external_id=str(source["external_attachment_id"]),
        ).first()
        result = reconcile_document(source, asset, document, text, existing, expected_org="273")
        proposals.append({"source_id": source["source_id"], "document_id": document.id, **result})
result = {"schema_version": "op23-provenance-dryrun-v1", "read_only": True,
          "admission_sha256": sha256(canonical_bytes(admission)),
          "proposals": proposals,
          "other_local_documents_unresolved_by_org_273": KnowledgeDocument.objects.exclude(tenant_id=9).count()}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
summary = {}
for item in proposals:
    summary[item["status"]] = summary.get(item["status"], 0) + 1
print(json.dumps({"read_only": True, "mapping_statuses": summary,
                  "other_local_documents_unresolved_by_org_273": result["other_local_documents_unresolved_by_org_273"]}, indent=2))
