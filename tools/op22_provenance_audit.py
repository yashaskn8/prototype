"""Read-only Starlly→Servy→chunk→vector→label reconciliation."""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

from core.models import Asset, KnowledgeDocument, SourceProvenanceRecord
from core.services.chroma_store import get_chroma_client

PRIVATE = ROOT / "private_training_data"
SNAPSHOT = PRIVATE / "op22_starlly_snapshot"
OUT = PRIVATE / "op22_provenance_audit.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
assets = json.loads((SNAPSHOT / "assets.json").read_text(encoding="utf-8"))
attachments = json.loads((SNAPSHOT / "attachments.json").read_text(encoding="utf-8"))
if manifest["organisation_id"] != "273" or manifest["tenant_id"] != 9:
    raise RuntimeError("Snapshot organisation/tenant mismatch")
if manifest["assets_sha256"] != sha((SNAPSHOT / "assets.json").read_bytes()):
    raise RuntimeError("Assets snapshot hash changed")
if manifest["complaints_sha256"] != sha((SNAPSHOT / "complaints.json").read_bytes()):
    raise RuntimeError("Complaints snapshot hash changed")

external_by_code = {a["code"]: a for a in assets if a.get("code")}
if len(external_by_code) != len(assets):
    raise RuntimeError("Duplicate or missing external asset code")
attachment_by_hash = {r.get("raw_sha256"): r for r in attachments if r.get("raw_sha256")}
if len(attachment_by_hash) != len(attachments):
    raise RuntimeError("Missing or duplicate attachment hashes")

local_assets = list(Asset.objects.filter(tenant_id=9).select_related("product"))
local_by_code = {a.asset_code: a for a in local_assets}
asset_links = []
for external in assets:
    local = local_by_code.get(external["code"])
    asset_links.append({
        "external_asset_id": external["id"], "local_asset_id": local.id if local else None,
        "code_matches": bool(local and local.asset_code == external["code"]),
        "model_matches": bool(local and (local.model_number or "") == (external.get("model_number") or "")),
    })

client = get_chroma_client()
coll = client.get_collection("servy_dense_op20_bge_ft_v1")
vectors = coll.get(include=["metadatas"])
vector_by_chunk = {int(m["chunk_id"]): m for m in vectors["metadatas"] if m and m.get("chunk_id") is not None}
documents = []
for doc in KnowledgeDocument.objects.filter(tenant_id=9).select_related("asset").prefetch_related("chunks"):
    source = attachment_by_hash.get(doc.checksum_sha256)
    external = external_by_code.get(doc.asset.asset_code) if doc.asset else None
    local_file = Path(doc.file.path) if doc.file else None
    raw_match = bool(source and source.get("download_status") == "DOWNLOADED_AND_EXTRACTED"
                     and local_file and local_file.exists() and sha(local_file.read_bytes()) == source["raw_sha256"])
    source_text = (SNAPSHOT / source["text_file"]).read_text(encoding="utf-8") if source and source.get("text_file") else ""
    text_match = bool(source and sha(doc.content_text.encode("utf-8")) == source.get("text_sha256")
                      and sha(source_text.encode("utf-8")) == source.get("text_sha256"))
    matching_local_asset = local_by_code.get(external["code"]) if external else None
    asset_match = bool(external and source and matching_local_asset
                       and str(source["external_asset_id"]) == str(external["id"])
                       and doc.asset_id == matching_local_asset.id)
    chunks = list(doc.chunks.all())
    chunk_links = []
    for chunk in chunks:
        meta = vector_by_chunk.get(chunk.id)
        vector_identity = bool(meta and meta.get("document_id") == doc.id
                               and meta.get("tenant_id") == doc.tenant_id
                               and meta.get("chunk_index") == chunk.chunk_index)
        chunk_links.append({
            "chunk_id": chunk.id, "source_text_contains_chunk": bool(chunk.text.strip() and chunk.text.strip() in source_text),
            "vector_identity_matches": vector_identity,
            "vector_content_hash_matches_body": bool(meta and meta.get("content_sha256") == sha(chunk.text.encode())[:16]),
        })
    db_provenance = list(SourceProvenanceRecord.objects.filter(tenant_id=9, local_object_type="KnowledgeDocument", local_object_id=doc.id))
    revoked = any(p.verification_status == "REVOKED" or p.revoked_at is not None for p in db_provenance)
    verified_source = bool(raw_match and text_match and asset_match and chunks
                           and not revoked and all(x["source_text_contains_chunk"] for x in chunk_links))
    documents.append({
        "document_id": doc.id, "tenant_id": doc.tenant_id, "local_asset_id": doc.asset_id,
        "external_asset_id": external["id"] if external else None,
        "external_attachment_id": source["external_attachment_id"] if source else None,
        "raw_sha256": source.get("raw_sha256") if source else None,
        "document_version": doc.index_version,
        "classification": "REJECTED_OR_REVOKED" if revoked else ("VERIFIED_ORIGINAL_COMPANY_SOURCE" if verified_source else "UNRESOLVED"),
        "raw_file_matches": raw_match, "extracted_text_matches": text_match,
        "asset_attachment_relationship_matches": asset_match,
        "db_provenance_record_count": len(db_provenance),
        "chunks": chunk_links,
    })

all_docs = KnowledgeDocument.objects.count()
classification_counts = Counter(d["classification"] for d in documents)
other_documents = [
    {"document_id": doc.id, "tenant_id": doc.tenant_id, "classification": "UNRESOLVED",
     "reason": "No authenticated source for this document in organisation 273 snapshot"}
    for doc in KnowledgeDocument.objects.exclude(tenant_id=9).only("id", "tenant_id")
]
payload = {
    "snapshot_sha256": sha((SNAPSHOT / "manifest.json").read_bytes()),
    "source_organisation": manifest["organisation_id"],
    "source_tenant": manifest["tenant_id"],
    "source_asset_count": len(assets), "source_attachment_count": len(attachments),
    "local_total_document_count": all_docs,
    "db_provenance_rows": SourceProvenanceRecord.objects.count(),
    "classification_counts_for_tenant_9": dict(classification_counts),
    "other_local_documents_not_certified_by_this_snapshot": all_docs - len(documents),
    "classification_counts_all_local": dict(classification_counts + Counter({"UNRESOLVED": len(other_documents)})),
    "asset_links": asset_links, "document_links": documents,
    "other_document_classifications": other_documents,
    "limitations": [
        "API snapshot and local source bytes match at capture time; human technical approval is separate.",
        "SourceProvenanceRecord remains empty in this local snapshot; no database mutation occurred.",
        "Operation 20 collection has short content hashes and nine heading/body representation differences.",
        "Non-company tenant documents are not certified by organisation 273's API.",
    ],
}
OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in payload.items() if k not in {"asset_links", "document_links", "other_document_classifications"}}, indent=2))
