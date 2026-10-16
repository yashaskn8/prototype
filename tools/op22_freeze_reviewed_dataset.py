"""Freeze a new, source-separated test only after authorized human review.

Requires a private reviewer allowlist and newly reconciled source documents.
Will reject today's pending queue and all historical training/test source hashes.
"""
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

from core.ml.op22_review_workflow import ReviewValidationError, freeze_reviewed_dataset
from core.models import KnowledgeChunk, KnowledgeDocument

parser = argparse.ArgumentParser()
parser.add_argument("--reviewed", type=Path, required=True)
parser.add_argument("--reviewer-allowlist", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
private = (ROOT / "private_training_data").resolve()
for path in (args.reviewed, args.reviewer_allowlist, args.output):
    if not path.resolve().is_relative_to(private):
        raise ReviewValidationError("Reviewed data, reviewer allowlist, and output must remain under private_training_data")

rows = [json.loads(line) for line in args.reviewed.read_text(encoding="utf-8").splitlines() if line.strip()]
reviewers = set(json.loads(args.reviewer_allowlist.read_text(encoding="utf-8")))
audit = json.loads((private / "op22_provenance_audit.json").read_text(encoding="utf-8"))
verified = {
    (r["document_id"], r["external_attachment_id"], r["raw_sha256"])
    for r in audit["document_links"] if r["classification"] == "VERIFIED_ORIGINAL_COMPANY_SOURCE"
}
for row in rows:
    identity = (row.get("source_document_id"), row.get("source_attachment_id"), row.get("source_sha256"))
    if identity not in verified:
        raise ReviewValidationError("A reviewed case does not match an independently reconciled source")
    source_document = KnowledgeDocument.objects.get(id=row["source_document_id"])
    if row.get("source_organisation") != audit["source_organisation"] or row.get("tenant_id") != source_document.tenant_id:
        raise ReviewValidationError("Reviewed case organisation or tenant does not match the verified document")
    for chunk_id in row.get("valid_positive_chunk_ids") or []:
        chunk = KnowledgeChunk.objects.filter(id=chunk_id, document_id=source_document.id).first()
        if chunk is None:
            raise ReviewValidationError("An approved positive does not belong to the verified source document")

exclusion = json.loads((private / "op22_review" / "prior_source_exclusion.json").read_text(encoding="utf-8"))
old_sources = set(exclusion["excluded_raw_source_sha256"])
manifest = freeze_reviewed_dataset(
    rows, args.output, authorized_reviewers=reviewers,
    excluded_source_hashes=old_sources,
)
manifest["source_separated_from_prior_train_validation_and_historical_benchmark"] = True
manifest["human_review_status"] = "APPROVED_BY_AUTHORIZED_REVIEWER"
manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
