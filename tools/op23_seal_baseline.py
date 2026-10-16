"""Seal existing development/history source and complaint identities once."""
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

from core.ml.op23_source_admission import prior_source_record, sha256
from core.models import KnowledgeDocument

private = ROOT / "private_training_data"
target = private / "op23_review" / "sealed_prior_sources_v2.json"
target.parent.mkdir(parents=True, exist_ok=True)
old = json.loads((private / "op22_review" / "prior_source_exclusion.json").read_text(encoding="utf-8"))
capture = private / "op23_snapshots" / "capture_20261016T103412Z"
complaints = json.loads((capture / "complaints.json").read_text(encoding="utf-8"))
sources = []
for doc in KnowledgeDocument.objects.select_related("asset").all():
    raw_hash = doc.checksum_sha256 or ""
    if len(raw_hash) != 64:
        raw_hash = ""
    entry = prior_source_record(raw_sha256=raw_hash, text=doc.content_text or "",
                                asset_code=doc.asset.asset_code if doc.asset else "UNSCOPED",
                                title=doc.original_filename or doc.title)
    entry["local_document_id"] = doc.id
    sources.append(entry)
payload = {
    "schema_version": "op23-sealed-baseline-v2",
    "op22_exclusion_sha256": sha256((private / "op22_review" / "prior_source_exclusion.json").read_bytes()),
    "excluded_raw_source_sha256": old["excluded_raw_source_sha256"],
    "historical_complaint_ids": sorted(c["id"] for c in complaints),
    "historical_complaint_sha256": [sha256(json.dumps(c, sort_keys=True).encode()) for c in complaints],
    "sources": sources,
}
with target.open("x", encoding="utf-8") as output:
    json.dump(payload, output, indent=2)
print(json.dumps({"baseline_path": str(target), "sources": len(sources),
                  "excluded_hashes": len(old["excluded_raw_source_sha256"]),
                  "historical_complaints": len(complaints)}, indent=2))
