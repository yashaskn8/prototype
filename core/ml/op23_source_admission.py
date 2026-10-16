"""Private, read-only admission of authenticated Starlly attachments for evaluation.

Admission verifies source identity and bytes. Independence is a separate verdict;
neither verdict constitutes a human relevance judgment or changes serving data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class SourceAdmissionError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def text_fingerprint(text: str) -> str:
    return sha256(" ".join(words(text)).encode())


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = words(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def source_family(asset_code: str, title: str) -> str:
    stem = Path(title or "").stem.casefold()
    stem = re.sub(r"\b(?:rev(?:ision)?|version|ver|v)\s*[-_.]?\s*\d+(?:[._-]\d+)*\b", "", stem)
    stem = " ".join(words(stem))
    return f"{asset_code.casefold().strip()}:{stem}"


def prior_source_record(*, raw_sha256: str, text: str, asset_code: str, title: str) -> dict:
    return {
        "raw_sha256": raw_sha256, "text_fingerprint": text_fingerprint(text),
        "shingles": sorted(shingles(text)),
        "source_family": source_family(asset_code, title),
    }


def independence_verdict(candidate: dict, prior: list[dict]) -> tuple[str, str]:
    candidate_shingles = set(candidate["shingles"])
    for old in prior:
        if candidate["raw_sha256"] == old["raw_sha256"]:
            return "HISTORICAL_EXCLUDED", "same_source_bytes"
        if candidate["text_fingerprint"] == old.get("text_fingerprint"):
            return "NEAR_DUPLICATE_QUARANTINED", "same_normalized_text"
        old_shingles = set(old.get("shingles") or [])
        if candidate_shingles and old_shingles:
            overlap = len(candidate_shingles & old_shingles)
            union = len(candidate_shingles | old_shingles)
            if overlap / union >= 0.65 or overlap / min(len(candidate_shingles), len(old_shingles)) >= 0.8:
                return "NEAR_DUPLICATE_QUARANTINED", "technical_text_overlap"
        if candidate["source_family"] and candidate["source_family"] == old.get("source_family"):
            return "REVISION_FAMILY_QUARANTINED", "same_asset_and_manual_family"
    return "INDEPENDENT_SOURCE_CANDIDATE", "no_prior_hash_text_or_family_match"


def extract_passages(text: str, raw_hash: str) -> list[dict]:
    """Keep paragraph boundaries and offsets; never synthesize missing content."""
    spans = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        body = match.group().strip()
        if not body:
            continue
        identity = sha256(f"{raw_hash}:{match.start()}:{match.end()}:{body}".encode())
        spans.append({"passage_id": identity, "start": match.start(), "end": match.end(),
                      "text": body, "text_sha256": sha256(body.encode("utf-8"))})
    if not spans and text.strip():
        body = text.strip()
        spans.append({"passage_id": sha256(f"{raw_hash}:0:{len(text)}:{body}".encode()),
                      "start": 0, "end": len(text), "text": body,
                      "text_sha256": sha256(body.encode("utf-8"))})
    return spans


def admit_capture(capture_dir: Path, baseline: dict, *, expected_org: str, tenant_id: int) -> dict:
    """Recheck versioned capture files and produce a private admission manifest."""
    capture_dir = Path(capture_dir).resolve()
    manifest = json.loads((capture_dir / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("organisation_id")) != str(expected_org) or manifest.get("tenant_id") != tenant_id:
        raise SourceAdmissionError("Authenticated capture organisation or tenant mismatch")
    if not manifest.get("catalog_authenticated"):
        raise SourceAdmissionError("Capture lacks authenticated catalogue evidence")
    for name in ("assets", "complaints", "site_assets", "attachments"):
        file = capture_dir / f"{name}.json"
        if sha256(file.read_bytes()) != manifest.get(f"{name}_sha256"):
            raise SourceAdmissionError(f"{name} capture hash mismatch")
    assets = json.loads((capture_dir / "assets.json").read_text(encoding="utf-8"))
    complaints = json.loads((capture_dir / "complaints.json").read_text(encoding="utf-8"))
    site_assets = json.loads((capture_dir / "site_assets.json").read_text(encoding="utf-8"))
    attachments = json.loads((capture_dir / "attachments.json").read_text(encoding="utf-8"))
    for row in complaints + site_assets:
        organisation = row.get("org_id", row.get("organisation_id", row.get("org")))
        if organisation is not None and str(organisation) != str(expected_org):
            raise SourceAdmissionError("Cross-organisation complaint or site relationship")
    if len({a.get("id") for a in assets}) != len(assets):
        raise SourceAdmissionError("Duplicate external asset ID")
    assets_by_id = {a["id"]: a for a in assets}
    if len({a.get("code") for a in assets}) != len(assets):
        raise SourceAdmissionError("Duplicate external asset code")
    for asset in assets:
        if str(asset.get("org_id")) != str(expected_org):
            raise SourceAdmissionError("Cross-organisation asset in capture")
    seen = set()
    prior = list(baseline.get("sources") or [])
    prior_hashes = set(baseline.get("excluded_raw_source_sha256") or [])
    records = []
    for attachment in attachments:
        asset = assets_by_id.get(attachment.get("external_asset_id"))
        identity = attachment.get("external_attachment_id")
        if asset is None or identity is None or identity in seen:
            raise SourceAdmissionError("Attachment identity or asset relationship invalid")
        seen.add(identity)
        asset_attachments = asset.get("attachments") or []
        catalogue_attachment = next((a for a in asset_attachments if a.get("id") == identity), None)
        if catalogue_attachment is None:
            raise SourceAdmissionError("Attachment absent from authenticated asset catalogue")
        for field in ("name", "created_at"):
            if (catalogue_attachment.get(field) and attachment.get(field)
                    and catalogue_attachment[field] != attachment[field]):
                raise SourceAdmissionError("Attachment metadata differs from authenticated asset catalogue")
        if attachment.get("download_status") != "DOWNLOADED_AND_EXTRACTED":
            records.append({"external_attachment_id": identity, "status": "SOURCE_UNAVAILABLE"})
            continue
        raw_name = attachment.get("raw_file")
        text_name = attachment.get("text_file")
        if not raw_name or not text_name:
            raise SourceAdmissionError("Downloaded attachment lacks private files")
        raw_path = (capture_dir / raw_name).resolve()
        text_path = (capture_dir / text_name).resolve()
        if not raw_path.is_relative_to(capture_dir) or not text_path.is_relative_to(capture_dir):
            raise SourceAdmissionError("Attachment file escapes capture directory")
        raw = raw_path.read_bytes()
        text = text_path.read_text(encoding="utf-8")
        raw_hash = sha256(raw)
        if raw_hash != attachment.get("raw_sha256") or sha256(text.encode()) != attachment.get("text_sha256"):
            raise SourceAdmissionError("Attachment byte or extracted text checksum mismatch")
        title = catalogue_attachment.get("name") or attachment.get("name") or ""
        candidate = prior_source_record(raw_sha256=raw_hash, text=text, asset_code=asset["code"], title=title)
        status, reason = independence_verdict(candidate, prior)
        if raw_hash in prior_hashes:
            status, reason = "HISTORICAL_EXCLUDED", "sealed_prior_source_hash"
        records.append({
            "source_id": f"starlly:{expected_org}:{asset['id']}:{identity}:{raw_hash[:16]}",
            "source_system": "starlly", "source_organisation": str(expected_org),
            "tenant_id": tenant_id, "external_asset_id": asset["id"],
            "external_attachment_id": identity, "asset_code": asset["code"],
            "equipment_model": asset.get("model_number"), "attachment_name": title,
            "source_version": catalogue_attachment.get("created_at") or attachment.get("created_at") or "UNKNOWN_VERSION",
            "raw_sha256": raw_hash, "text_sha256": attachment["text_sha256"],
            "source_family": candidate["source_family"],
            "source_verification": "VERIFIED_ORIGINAL_COMPANY_SOURCE",
            "independence_status": status, "independence_reason": reason,
            "passages": extract_passages(text, raw_hash),
        })
        if status == "INDEPENDENT_SOURCE_CANDIDATE":
            prior.append(candidate)
    return {"schema_version": "op23-admission-v1", "source_organisation": str(expected_org),
            "tenant_id": tenant_id, "capture_manifest_sha256": sha256((capture_dir / "manifest.json").read_bytes()),
            "baseline_sha256": sha256(canonical_bytes(baseline)), "sources": records}
