"""Capture a new, private, versioned read-only Starlly catalogue snapshot."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.op22_starlly_snapshot import load_existing_env, paged_records


def sha(data):
    return hashlib.sha256(data).hexdigest()


def capture(client, tenant_id: int, organisation_id: str, output: Path) -> dict:
    """Require an empty destination and an authenticated, matching resource catalog."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a source capture")
    catalogue = client._request_get("/api/external/", resource_label="Starlly Resource Catalog").json()
    if not isinstance(catalogue, dict) or str(catalogue.get("organisation_id")) != str(organisation_id):
        raise ValueError("Authenticated catalogue organisation mismatch")
    allowed = {r.get("name") for r in catalogue.get("resources", []) if isinstance(r, dict)}
    if not {"assets", "complaints", "site-assets"}.issubset(allowed):
        raise ValueError("The key lacks required read-only source relationships")
    assets = paged_records(client, "assets")
    complaints = paged_records(client, "complaints")
    site_assets = paged_records(client, "site-assets")
    for asset in assets:
        if str(asset.get("org_id")) != str(organisation_id):
            raise ValueError("Cross-organisation asset in capture")
    for row in site_assets + complaints:
        org = row.get("org_id", row.get("organisation_id", row.get("org")))
        if org is not None and str(org) != str(organisation_id):
            raise ValueError("Cross-organisation relationship in capture")
    output.mkdir(parents=True)
    from core.services.starlly_ingest import fetch_attachment_content
    attachments = []
    clean_assets = []
    for asset in assets:
        clean = {k: v for k, v in asset.items() if k not in {"count", "attachments"}}
        clean["attachments"] = []
        for item in asset.get("attachments") or []:
            if not isinstance(item, dict) or item.get("id") is None:
                raise ValueError("Attachment lacks external ID")
            clean["attachments"].append({k: v for k, v in item.items() if k != "file_url"})
            record = {"external_asset_id": asset["id"], "external_attachment_id": item["id"],
                      "name": item.get("name"), "created_at": item.get("created_at")}
            try:
                raw, raw_hash, text = fetch_attachment_content(item.get("file_url") or "")
                stem = f"asset_{asset['id']}_attachment_{item['id']}_{raw_hash[:16]}"
                raw_file, text_file = f"{stem}.bin", f"{stem}.txt"
                (output / raw_file).write_bytes(raw)
                (output / text_file).write_text(text, encoding="utf-8")
                record.update(download_status="DOWNLOADED_AND_EXTRACTED", raw_sha256=raw_hash,
                              text_sha256=sha(text.encode()), raw_file=raw_file, text_file=text_file,
                              byte_count=len(raw))
            except Exception as exc:
                record.update(download_status="FAILED", failure_type=type(exc).__name__)
            attachments.append(record)
        clean_assets.append(clean)
    for name, rows in (("assets", clean_assets), ("complaints", complaints),
                       ("site_assets", site_assets), ("attachments", attachments)):
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest = {"schema_version": "op23-capture-v1", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "catalog_authenticated": True, "organisation_id": str(organisation_id), "tenant_id": tenant_id,
                "permitted_resources": sorted(allowed), "asset_count": len(assets),
                "complaint_count": len(complaints), "site_asset_count": len(site_assets),
                "attachment_count": len(attachments),
                "downloaded_attachment_count": sum(a.get("download_status") == "DOWNLOADED_AND_EXTRACTED" for a in attachments)}
    for name in ("assets", "complaints", "site_assets", "attachments"):
        manifest[f"{name}_sha256"] = sha((output / f"{name}.json").read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-org", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_existing_env()
    os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    import django
    django.setup()
    from core.models import Tenant
    from core.services.starlly_api import StarllyApiClient, get_authorized_org_for_tenant
    private = (ROOT / "private_training_data").resolve()
    if not args.output.resolve().is_relative_to(private):
        raise ValueError("Source capture must remain private")
    tenant = Tenant.objects.get(slug="skydream")
    if get_authorized_org_for_tenant(tenant) != args.expected_org:
        raise ValueError("Tenant is not mapped to requested organisation")
    result = capture(StarllyApiClient(tenant=tenant), tenant.id, args.expected_org, args.output)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, indent=2))
