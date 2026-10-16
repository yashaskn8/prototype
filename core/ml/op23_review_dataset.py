"""Validate source-separated, signed review cases before sealing evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from core.ml.op23_review_authority import ApprovalError, verify_signed_review
from core.ml.op23_source_admission import canonical_bytes, sha256


class DatasetValidationError(ValueError):
    pass


DATASET_ROLES = frozenset({"TRAIN", "DEVELOPMENT", "HISTORICAL_DIAGNOSTIC", "INDEPENDENT_FINAL_TEST"})
ANSWERABILITY = frozenset({"ANSWERABLE", "UNANSWERABLE_FROM_APPROVED_SOURCES", "NEEDS_ASSET_CLARIFICATION"})


def validate_case(case: dict, *, admission: dict, baseline: dict, complaints: list[dict], trusted_keys: dict,
                  dataset_role: str) -> dict:
    if dataset_role not in DATASET_ROLES or case.get("dataset_role") != dataset_role:
        raise DatasetValidationError("Explicit dataset role mismatch")
    if (not isinstance(case.get("answerability"), str) or case["answerability"] not in ANSWERABILITY
            or not isinstance(case.get("query_id"), str) or not case["query_id"]
            or not isinstance(case.get("question_text"), str) or not case["question_text"]):
        raise DatasetValidationError("Question identity and answerability required")
    org, tenant = admission["source_organisation"], admission["tenant_id"]
    try:
        approval = verify_signed_review(case, trusted_keys, expected_org=org, tenant_id=tenant)
    except ApprovalError as exc:
        raise DatasetValidationError(str(exc)) from exc
    provenance = case.get("query_provenance")
    if not isinstance(provenance, dict):
        raise DatasetValidationError("Question lacks authenticated complaint provenance")
    complaint = next((c for c in complaints if c.get("id") == provenance.get("complaint_id")), None)
    if complaint is None or case["query_provenance"].get("kind") != "starlly_complaint":
        raise DatasetValidationError("Question lacks authenticated complaint provenance")
    if sha256(canonical_bytes(complaint)) != case["query_provenance"].get("complaint_sha256"):
        raise DatasetValidationError("Complaint content changed after review")
    if case["question_text"] != (complaint.get("description") or ""):
        raise DatasetValidationError("Reviewed question differs from authenticated complaint")
    if dataset_role == "INDEPENDENT_FINAL_TEST" and complaint["id"] in baseline["historical_complaint_ids"]:
        raise DatasetValidationError("Historical complaint cannot enter independent final test")
    if complaint.get("asset") != case.get("external_asset_id"):
        raise DatasetValidationError("Complaint asset identity changed or was guessed")
    sources = {s["source_id"]: s for s in admission["sources"] if s.get("source_id")}
    positives = case.get("valid_positive_passage_ids") or []
    alternatives = case.get("valid_alternative_passage_ids") or []
    negatives = case.get("reviewed_negative_passage_ids") or []
    if any(not isinstance(rows, list) or any(not isinstance(value, str) for value in rows)
           for rows in (positives, alternatives, negatives)):
        raise DatasetValidationError("Reviewed passages must be lists of exact passage IDs")
    if (len(positives) != len(set(positives)) or len(negatives) != len(set(negatives))
            or not set(alternatives).issubset(positives) or set(negatives) & set(positives)):
        raise DatasetValidationError("Duplicate positive, unlisted alternative or false negative")
    status = case["answerability"]
    if status == "NEEDS_ASSET_CLARIFICATION":
        if (complaint.get("asset") is not None or case.get("evaluation_task") != "CLARIFICATION"
                or positives or alternatives or negatives or any(case.get(key) for key in
                    ("source_id", "source_sha256", "source_version", "source_attachment_id"))):
            raise DatasetValidationError("Clarification case must have no asset and no technical positives")
        return approval
    if case.get("evaluation_task") != "TECHNICAL_ANSWERABILITY" or case.get("external_asset_id") is None:
        raise DatasetValidationError("Technical scoring requires resolved complaint equipment")
    expected_universe = {sid for sid, source in sources.items()
                         if source.get("external_asset_id") == case["external_asset_id"]
                         and source.get("source_verification") == "VERIFIED_ORIGINAL_COMPANY_SOURCE"}
    search = case.get("searched_source_universe") or {}
    if not isinstance(search, dict):
        raise DatasetValidationError("Source universe must be a reviewed object")
    if not expected_universe or set(search.get("source_ids") or []) != expected_universe:
        raise DatasetValidationError("Authorized source universe is incomplete or mismatched")
    if not search.get("search_complete") or set(search.get("indexed_source_ids") or []) != expected_universe:
        raise DatasetValidationError("Incomplete search or index cannot certify answerability")
    if status == "UNANSWERABLE_FROM_APPROVED_SOURCES":
        if positives or alternatives or any(case.get(key) for key in
                                         ("source_id", "source_sha256", "source_version", "source_attachment_id")):
            raise DatasetValidationError("Unanswerable case must not invent a positive source")
        owner = {p["passage_id"] for source in sources.values()
                 if source.get("external_asset_id") == case["external_asset_id"]
                 for p in source.get("passages", [])}
        if not set(negatives).issubset(owner):
            raise DatasetValidationError("Reviewed negatives must be real passages in the searched asset")
        return approval
    if not positives:
        raise DatasetValidationError("Answerable case requires approved positive passages")
    source = sources.get(case.get("source_id"))
    if not source or source.get("external_asset_id") != case["external_asset_id"]:
        raise DatasetValidationError("Positive source is unavailable or wrong equipment")
    if dataset_role == "INDEPENDENT_FINAL_TEST" and source.get("independence_status") != "INDEPENDENT_SOURCE_CANDIDATE":
        raise DatasetValidationError("Final-test source is historical, revised or near-duplicate")
    if (case.get("source_sha256") != source["raw_sha256"]
            or case.get("source_version") != source["source_version"]
            or case.get("source_attachment_id") != source["external_attachment_id"]):
        raise DatasetValidationError("Source identity or document version changed after review")
    owner = {p["passage_id"]: candidate for candidate in sources.values()
             for p in candidate.get("passages", [])}
    if not set(negatives).issubset({p["passage_id"] for candidate in sources.values()
                                   if candidate.get("external_asset_id") == case["external_asset_id"]
                                   for p in candidate.get("passages", [])}):
        raise DatasetValidationError("Reviewed negatives must be real passages in the searched asset")
    approved_versions = case.get("approved_source_versions") or {}
    for passage_id in positives:
        candidate = owner.get(passage_id)
        if not candidate or candidate["external_asset_id"] != case["external_asset_id"]:
            raise DatasetValidationError("Approved positive is not an exact passage for this asset")
        if dataset_role == "INDEPENDENT_FINAL_TEST" and candidate.get("independence_status") != "INDEPENDENT_SOURCE_CANDIDATE":
            raise DatasetValidationError("Alternate positive overlaps a historical or revised source")
        reviewed_version = (case["source_version"] if candidate["source_id"] == case["source_id"]
                            else approved_versions.get(candidate["source_id"]))
        if reviewed_version != candidate["source_version"]:
            raise DatasetValidationError("Approved alternate source version changed after review")
    if not any(p["passage_id"] in positives for p in source["passages"]):
        raise DatasetValidationError("Primary reviewed source lacks a positive passage")
    return approval


def freeze_cases(cases: list[dict], destination: Path, *, admission: dict, baseline: dict,
                 complaints: list[dict], trusted_keys: dict, dataset_role: str,
                 existing_role_cases: list[dict] | None = None) -> dict:
    if not cases:
        raise DatasetValidationError("Cannot freeze an empty dataset")
    ids = [c.get("query_id") for c in cases]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("Duplicate query ID")
    sources = {s["source_id"]: s for s in admission["sources"] if s.get("source_id")}
    def families(case):
        ids = {case.get("source_id")}
        owner = {p["passage_id"]: source["source_id"] for source in sources.values()
                 for p in source.get("passages", [])}
        ids.update(owner.get(p) for p in case.get("valid_positive_passage_ids") or [])
        return {sources[sid].get("source_family") for sid in ids if sid in sources}

    for existing in existing_role_cases or []:
        for case in cases:
            same_query = case.get("query_provenance") == existing.get("query_provenance")
            same_source_family = bool(families(case) & families(existing))
            if same_query or same_source_family:
                raise DatasetValidationError("Source or query family overlaps a frozen dataset")
    verified_approvals = [validate_case(c, admission=admission, baseline=baseline, complaints=complaints,
                                        trusted_keys=trusted_keys, dataset_role=dataset_role) for c in cases]
    destination = Path(destination)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise DatasetValidationError("Refusing to overwrite frozen dataset or manifest")
    payload = "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
    manifest = {"schema_version": "op23-reviewed-dataset-v1", "dataset_role": dataset_role,
                "case_count": len(cases), "dataset_sha256": sha256(payload.encode()),
                "admission_sha256": sha256(canonical_bytes(admission)),
                "baseline_sha256": sha256(canonical_bytes(baseline)),
                "approval_content_sha256": [a["signed_content_sha256"] for a in verified_approvals],
                "technical_case_count": sum(c.get("evaluation_task") == "TECHNICAL_ANSWERABILITY" for c in cases),
                "clarification_case_count": sum(c.get("evaluation_task") == "CLARIFICATION" for c in cases)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    return manifest
