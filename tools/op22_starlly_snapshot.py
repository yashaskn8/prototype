"""Read-only, organisation-bound Starlly snapshot for private review preparation.

Credentials come from the existing local .env and never enter output files.
No Servy database, Chroma collection, or upstream record is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_existing_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in {"STARLLY_API_KEY", "STARLLY_API_BASE_URL", "STARLLY_ORGANISATION_ID"}:
            os.environ.setdefault(key, value.strip().strip("\"'"))


def paged_records(client, resource: str, max_records: int = 5000, page_size: int = 50) -> list[dict]:
    """Fetch a complete array response; reject duplicate or inconsistent pages."""
    if resource not in {"assets", "complaints", "site-assets"}:
        raise ValueError("Unsupported paged resource")
    records = []
    seen_ids = set()
    expected_total = None
    for offset in range(0, max_records, page_size):
        response = client._request_get(
            f"/api/external/{resource}/", params={"limit": page_size, "offset": offset},
            resource_label=f"Starlly {resource} snapshot",
        )
        payload = response.json()
        if not isinstance(payload, list) or len(payload) > page_size or not all(isinstance(row, dict) for row in payload):
            raise RuntimeError(f"Malformed {resource} page at offset {offset}")
        if payload and "count" in payload[0]:
            count = payload[0]["count"]
            if not isinstance(count, int) or count < 0 or count > max_records:
                raise RuntimeError(f"Invalid {resource} count")
            if expected_total is not None and count != expected_total:
                raise RuntimeError(f"{resource} count changed during pagination")
            expected_total = count
        for row in payload:
            identity = row.get("id")
            if identity is None or identity in seen_ids:
                raise RuntimeError(f"Missing or duplicate {resource} identity")
            seen_ids.add(identity)
            records.append(row)
        if expected_total is not None:
            if len(records) == expected_total:
                return records
            if len(records) > expected_total or not payload:
                raise RuntimeError(f"Incomplete or overfull {resource} pagination")
        elif len(payload) < page_size:
            return records
    raise RuntimeError(f"{resource} exceeds bounded {max_records}-record snapshot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-org", required=True, help="Explicit authorized upstream organisation ID")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--download-attachments", action="store_true")
    args = parser.parse_args()
    load_existing_env()
    os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    import django
    django.setup()
    from core.models import Tenant
    from core.services.starlly_api import StarllyApiClient, get_authorized_org_for_tenant

    tenant = Tenant.objects.get(slug="skydream")
    mapped_org = get_authorized_org_for_tenant(tenant)
    if mapped_org != args.expected_org:
        raise RuntimeError("Configured tenant-to-organisation mapping differs from explicit authorized organisation")
    client = StarllyApiClient(tenant=tenant, timeout=15)
    response = client._request_get("/api/external/", resource_label="Starlly Resource Catalog")
    catalog = response.json()
    if not isinstance(catalog, dict) or str(catalog.get("organisation_id")) != mapped_org:
        raise RuntimeError("Authenticated catalog organisation does not match tenant mapping")
    resources = {r.get("name") for r in catalog.get("resources", []) if isinstance(r, dict)}
    print(json.dumps({"access": "authenticated", "organisation_matches": True, "resources": sorted(resources)}), flush=True)
    if args.catalog_only:
        return
    if "assets" not in resources:
        raise RuntimeError("API key lacks assets permission; no source snapshot can be certified")

    assets = paged_records(client, "assets")
    complaints = paged_records(client, "complaints") if "complaints" in resources else []
    for asset in assets:
        if str(asset.get("org_id")) != mapped_org:
            raise RuntimeError("Cross-organisation asset returned by API")

    private = ROOT / "private_training_data" / "op22_starlly_snapshot"
    private.mkdir(parents=True, exist_ok=True)
    attachment_records = []
    for asset in assets:
        for attachment in asset.get("attachments") or []:
            if not isinstance(attachment, dict) or attachment.get("id") is None:
                raise RuntimeError("Asset attachment lacks stable external ID")
            record = {
                "external_asset_id": asset["id"], "external_attachment_id": attachment["id"],
                "name": attachment.get("name"), "created_at": attachment.get("created_at"),
                "download_status": "NOT_REQUESTED",
            }
            if args.download_attachments:
                from core.services.starlly_ingest import fetch_attachment_content
                try:
                    raw, raw_hash, text = fetch_attachment_content(attachment.get("file_url") or "")
                    stem = f"asset_{asset['id']}_attachment_{attachment['id']}_{raw_hash[:16]}"
                    (private / f"{stem}.bin").write_bytes(raw)
                    (private / f"{stem}.txt").write_text(text, encoding="utf-8")
                    record.update({
                        "download_status": "DOWNLOADED_AND_EXTRACTED", "raw_sha256": raw_hash,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "byte_count": len(raw), "text_file": f"{stem}.txt",
                    })
                except Exception as exc:
                    record["download_status"] = "FAILED"
                    record["failure_type"] = type(exc).__name__
            attachment_records.append(record)

    # Signed file URLs and secrets are intentionally excluded from saved JSON.
    clean_assets = []
    for asset in assets:
        clean = {k: v for k, v in asset.items() if k not in {"attachments", "count"}}
        clean["attachments"] = [
            {k: v for k, v in attachment.items() if k != "file_url"}
            for attachment in asset.get("attachments") or []
        ]
        clean_assets.append(clean)
    (private / "assets.json").write_text(json.dumps(clean_assets, indent=2), encoding="utf-8")
    (private / "complaints.json").write_text(json.dumps(complaints, indent=2), encoding="utf-8")
    (private / "attachments.json").write_text(json.dumps(attachment_records, indent=2), encoding="utf-8")
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "authenticated_read_only_starlly_external_api",
        "organisation_id": mapped_org, "tenant_id": tenant.id,
        "permitted_resources": sorted(resources),
        "asset_count": len(assets), "complaint_count": len(complaints),
        "attachment_count": len(attachment_records),
        "downloaded_attachment_count": sum(r["download_status"] == "DOWNLOADED_AND_EXTRACTED" for r in attachment_records),
        "failed_attachment_count": sum(r["download_status"] == "FAILED" for r in attachment_records),
        "assets_sha256": hashlib.sha256((private / "assets.json").read_bytes()).hexdigest(),
        "complaints_sha256": hashlib.sha256((private / "complaints.json").read_bytes()).hexdigest(),
    }
    (private / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
